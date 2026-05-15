# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert (
            (not self.use_distributed)
            or is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        self._maybe_compile_model()

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        if self.use_distributed:
            dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32
            if cuda_conf.allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.use_distributed = world_size > 1
        if not self.use_distributed:
            logging.info("WORLD_SIZE=1 detected; skipping distributed process-group initialization.")
            self.rank = 0
            return

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_model_state_with_shape_filter(self, model_state_dict: Mapping[str, torch.Tensor]):
        current_state = self.model.state_dict()
        filtered_state = {}
        skipped_shape_mismatch = {}
        unexpected_input_keys = []

        for key, value in model_state_dict.items():
            if key not in current_state:
                unexpected_input_keys.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped_shape_mismatch[key] = {
                    "checkpoint_shape": tuple(value.shape),
                    "model_shape": tuple(current_state[key].shape),
                }
                continue
            filtered_state[key] = value

        missing, unexpected_loaded = self.model.load_state_dict(filtered_state, strict=False)
        unexpected = list(dict.fromkeys(list(unexpected_input_keys) + list(unexpected_loaded)))
        return missing, unexpected, skipped_shape_mismatch

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected, skipped_shape_mismatch = self._load_model_state_with_shape_filter(model_state_dict)
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
            if skipped_shape_mismatch:
                preview = {
                    key: value
                    for key, value in list(skipped_shape_mismatch.items())[:16]
                }
                logging.info(
                    "Model state skipped %d shape-mismatched key(s). Preview: %s",
                    len(skipped_shape_mismatch),
                    preview,
                )

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint and getattr(self, "optims", None) is not None:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            optimizer_wrappers = (
                list(self.optims)
                if isinstance(self.optims, (list, tuple))
                else [self.optims]
            )
            optimizer_states = checkpoint["optimizer"]
            if not isinstance(optimizer_states, list):
                optimizer_states = [optimizer_states]

            if len(optimizer_wrappers) == len(optimizer_states):
                for optim_wrapper, optimizer_state in zip(optimizer_wrappers, optimizer_states):
                    optimizer = getattr(optim_wrapper, "optimizer", optim_wrapper)
                    try:
                        optimizer.load_state_dict(optimizer_state)
                    except ValueError as exc:
                        logging.warning(
                            "Skipping optimizer state load because the resumed optimizer param groups "
                            "do not match the current model structure: %s",
                            exc,
                        )
            elif len(optimizer_wrappers) == 1 and len(optimizer_states) == 1:
                optimizer = getattr(optimizer_wrappers[0], "optimizer", optimizer_wrappers[0])
                try:
                    optimizer.load_state_dict(optimizer_states[0])
                except ValueError as exc:
                    logging.warning(
                        "Skipping optimizer state load because the resumed optimizer param groups "
                        "do not match the current model structure: %s",
                        exc,
                    )
            else:
                logging.warning(
                    "Skipping optimizer state load because checkpoint stores %d optimizer state(s) "
                    "but trainer initialized %d optimizer wrapper(s).",
                    len(optimizer_states),
                    len(optimizer_wrappers),
                )

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        elif "prev_epoch" in checkpoint:
            self.epoch = int(checkpoint["prev_epoch"]) + 1
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _maybe_compile_model(self):
        compile_conf = getattr(self.optim_conf, "compile", None)
        if compile_conf is None or not bool(getattr(compile_conf, "enabled", False)):
            return
        if self.device.type != "cuda":
            logging.info("torch.compile requested on non-CUDA device; skipping.")
            return
        if not hasattr(torch, "compile"):
            logging.warning("torch.compile requested but not available in this PyTorch build; skipping.")
            return

        backend = str(getattr(compile_conf, "backend", "inductor"))
        mode = getattr(compile_conf, "mode", None)
        dynamic = bool(getattr(compile_conf, "dynamic", False))
        fullgraph = bool(getattr(compile_conf, "fullgraph", False))
        suppress_errors = bool(getattr(compile_conf, "suppress_errors", False))
        compile_kwargs = {
            "backend": backend,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
        }
        if mode not in (None, "", "null"):
            compile_kwargs["mode"] = str(mode)
        if suppress_errors:
            try:
                import importlib

                dynamo = importlib.import_module("torch._dynamo")
                dynamo.config.suppress_errors = True
                logging.info("torch.compile suppress_errors enabled; backend failures will fall back to eager.")
            except Exception as exc:
                logging.warning("Failed to enable torch.compile suppress_errors: %s", exc)

        logging.info(
            "Compiling model with torch.compile backend=%s mode=%s dynamic=%s fullgraph=%s suppress_errors=%s",
            backend,
            mode,
            dynamic,
            fullgraph,
            suppress_errors,
        )
        start_time = time.time()
        self.model = torch.compile(self.model, **compile_kwargs)
        logging.info("torch.compile finished in %.2fs", time.time() - start_time)

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        if not self.use_distributed:
            logging.info("Single-process mode enabled; leaving model unwrapped by DDP.")
            return

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        if not bool(getattr(self.checkpoint_conf, "enabled", True)):
            logging.info("Checkpoint saving disabled by config; skipping save_checkpoint.")
            return

        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            current_where = float(exact_epoch) / self.max_epochs
            batch["train_progress"] = current_where

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter < limit_train_batches
            self.where = current_where
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", "foreground_masks", "conf_depth_point_masks",
            "depth_conf_maps",
            "smpl_prior_masks",
            "smpl_prior_feature_maps",
            "smpl_vertex_feature_maps",
            "prior_maps",
            "prior_mask",
            "prior_depths",
            "prior_points",
            "prior_normals",
            "teacher_mask",
            "smplx_bodyhand_anchor_mask",
            "smplx_body_anchor_mask",
            "smplx_hand_anchor_mask",
            "smplx_left_hand_anchor_mask",
            "smplx_right_hand_anchor_mask",
            "smplx_native_visible_mask",
            "human_prior_completion_masks",
            "human_prior_completion_depths", "human_prior_completion_world_points", "human_prior_completion_point_masks",
            "head_hair_region_masks",
            "head_hair_detail_masks",
            "selection_anchor_quality_score",
            "selection_sample_manifest_applied",
        ]        
        string_keys = ["seq_name", "selection_anchor_camera", "selection_sample_manifest_label"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                flipped_tensor = (
                    torch.flip(original_tensor, dims=[1])
                    if original_tensor.ndim > 1
                    else original_tensor
                )
                batch[key] = torch.concatenate([original_tensor, flipped_tensor], dim=0)

        if "smpl_summary_tokens" in batch:
            original_tokens = batch["smpl_summary_tokens"]
            batch["smpl_summary_tokens"] = torch.concatenate([original_tokens, original_tokens], dim=0)
        
        for key in string_keys:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key] * 2

        if "selection_anchor_view_index" in batch:
            original_index = batch["selection_anchor_view_index"]
            if "images" in batch and batch["images"].ndim > 1:
                view_count = int(batch["images"].shape[1])
            else:
                view_count = 0
            if view_count > 0:
                flipped_index = torch.where(
                    original_index >= 0,
                    (view_count - 1) - original_index,
                    original_index,
                )
            else:
                flipped_index = original_index
            batch["selection_anchor_view_index"] = torch.concatenate([original_index, flipped_index], dim=0)
        
        return batch

    def _compute_batch_scene_normalization_stats(self, batch: Mapping):
        extrinsics = batch["extrinsics"]
        world_points = batch["world_points"]
        point_masks = batch["point_masks"]

        first_cam_rotation = extrinsics[:, 0, :3, :3]
        first_cam_translation = extrinsics[:, 0, :3, 3]
        normalized_world_points = (
            world_points @ first_cam_rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
        ) + first_cam_translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        dist = normalized_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1, 2, 3])
        valid_count = point_masks.sum(dim=[1, 2, 3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        return first_cam_rotation, first_cam_translation, avg_scale

    def _normalize_human_prior_completion_batch_tensors(self, batch: Mapping) -> None:
        first_cam_rotation, first_cam_translation, avg_scale = self._compute_batch_scene_normalization_stats(batch)

        prior_point_masks = batch.get("human_prior_completion_point_masks", None)
        prior_world_points = batch.get("human_prior_completion_world_points", None)
        prior_depths = batch.get("human_prior_completion_depths", None)
        if prior_point_masks is not None and (prior_world_points is not None or prior_depths is not None):
            if prior_world_points is not None:
                normalized_prior_world_points = (
                    prior_world_points @ first_cam_rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
                ) + first_cam_translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                normalized_prior_world_points = normalized_prior_world_points / avg_scale.view(-1, 1, 1, 1, 1)
                batch["human_prior_completion_world_points"] = torch.where(
                    prior_point_masks.unsqueeze(-1),
                    normalized_prior_world_points,
                    torch.zeros_like(normalized_prior_world_points),
                )

            if prior_depths is not None:
                normalized_prior_depths = prior_depths / avg_scale.view(-1, 1, 1, 1)
                batch["human_prior_completion_depths"] = torch.where(
                    prior_point_masks,
                    normalized_prior_depths,
                    torch.zeros_like(normalized_prior_depths),
                )

        native_prior_mask = batch.get("prior_mask", None)
        native_prior_points = batch.get("prior_points", None)
        native_prior_depths = batch.get("prior_depths", None)
        if native_prior_mask is None or (native_prior_points is None and native_prior_depths is None):
            return

        if native_prior_points is not None:
            normalized_prior_world_points = (
                native_prior_points @ first_cam_rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
            ) + first_cam_translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            normalized_prior_world_points = normalized_prior_world_points / avg_scale.view(-1, 1, 1, 1, 1)
            batch["prior_points"] = torch.where(
                native_prior_mask.unsqueeze(-1),
                normalized_prior_world_points,
                torch.zeros_like(normalized_prior_world_points),
            )

        if native_prior_depths is not None:
            normalized_prior_depths = native_prior_depths / avg_scale.view(-1, 1, 1, 1)
            batch["prior_depths"] = torch.where(
                native_prior_mask,
                normalized_prior_depths,
                torch.zeros_like(normalized_prior_depths),
            )

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        self._normalize_human_prior_completion_batch_tensors(batch)

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        model_kwargs = {"images": batch["images"]}
        if "smpl_vertex_feature_maps" in batch and batch["smpl_vertex_feature_maps"] is not None:
            model_kwargs["human_prior_feature_maps"] = batch["smpl_vertex_feature_maps"]
        if "smpl_summary_tokens" in batch and batch["smpl_summary_tokens"] is not None:
            model_kwargs["human_prior_summary_tokens"] = batch["smpl_summary_tokens"]
        y_hat = model(**model_kwargs)
        
        # Loss computation
        loss_batch = dict(batch)
        loss_batch["_loss_phase"] = phase
        loss_dict = self.loss(y_hat, loss_batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            data_key = key
            if key not in data and key == "loss_objective" and "objective" in data:
                data_key = "objective"

            if data_key in data:
                value = data[data_key].item() if torch.is_tensor(data[data_key]) else data[data_key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
