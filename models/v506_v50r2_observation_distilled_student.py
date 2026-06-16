from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORBIDDEN_INFERENCE_KEY_TOKENS = (
    "teacher",
    "v50r2",
    "kinect",
    "v591",
    "raw_depth",
    "rgbd",
    "direct_fusion",
)

CONTROL_MODES = (
    "true",
    "vggt_baseline",
    "no_smpl",
    "shuffled_semantic",
    "same_topology_no_semantic",
    "shell_only",
    "observation_only",
    "visible_only",
)


@dataclass(frozen=True)
class V506ObservationDistilledStudentConfig:
    vggt_feature_dim: int = 12
    smplx_feature_dim: int = 16
    local_frame_dim: int = 9
    camera_feature_dim: int = 6
    hidden_dim: int = 192
    max_residual: float = 0.035
    max_rgb_delta: float = 0.04
    max_normal_delta: float = 0.18
    part_count: int = 8


def _normalized_key(key: object) -> str:
    return str(key).lower().replace("-", "_").replace(" ", "_")


def _collect_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return []
    bad: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        normalized = _normalized_key(key)
        compact = "".join(ch for ch in normalized if ch.isalnum())
        for token in FORBIDDEN_INFERENCE_KEY_TOKENS:
            if token in normalized or token.replace("_", "") in compact:
                bad.append(path)
        if isinstance(item, Mapping):
            bad.extend(_collect_forbidden_keys(item, path))
    return bad


def reject_forbidden_inference_keys(batch: Mapping[str, Any]) -> None:
    bad = sorted(set(_collect_forbidden_keys(batch)))
    if bad:
        raise ValueError(f"forbidden teacher/reference/depth inference keys present: {bad}")


def _require(batch: Mapping[str, Tensor], key: str) -> Tensor:
    if key not in batch:
        raise KeyError(key)
    return batch[key]


def _xyz(name: str, value: Tensor) -> Tensor:
    if value.shape[-1] != 3:
        raise ValueError(f"{name} must end with xyz dimension 3")
    return value.float()


def _feature(name: str, value: Tensor, ref: Tensor, dim: int) -> Tensor:
    if value.shape[:-1] != ref.shape[:-1]:
        raise ValueError(f"{name} leading shape {tuple(value.shape[:-1])} must match VGGT point shape {tuple(ref.shape[:-1])}")
    if value.shape[-1] != dim:
        raise ValueError(f"{name} last dim must be {dim}, got {value.shape[-1]}")
    return value.float()


def _scalar_feature(name: str, value: Tensor, ref: Tensor) -> Tensor:
    if value.shape == ref.shape[:-1]:
        value = value.unsqueeze(-1)
    if value.shape != ref.shape[:-1] + (1,):
        raise ValueError(f"{name} must match leading point shape and have optional scalar channel")
    return value.float()


def _part_ids(value: Tensor | None, ref: Tensor, part_count: int) -> Tensor:
    if value is None:
        return torch.zeros(ref.shape[:-1], device=ref.device, dtype=torch.long)
    if value.shape == ref.shape[:-1] + (1,):
        value = value.squeeze(-1)
    if value.shape != ref.shape[:-1]:
        raise ValueError("body_part_id must match VGGT point leading shape")
    return value.to(device=ref.device, dtype=torch.long).clamp(0, part_count - 1)


def _norm(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1, eps=1.0e-6)


class V506V50R2ObservationDistilledStudent(nn.Module):
    """Model-owned VGGT+SMPL-X student for V50R2 observation distillation.

    V50R2 is deliberately absent from the forward inputs. Training code may use
    the teacher bank to form losses outside this module, while inference gets
    only VGGT observation, SMPL-X graph/local-frame features, masks, camera
    context, and environment points.
    """

    def __init__(self, cfg: V506ObservationDistilledStudentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or V506ObservationDistilledStudentConfig()
        self.part_embedding = nn.Embedding(self.cfg.part_count, 24)
        in_dim = (
            3
            + 3
            + 1
            + self.cfg.vggt_feature_dim
            + self.cfg.smplx_feature_dim
            + 3
            + self.cfg.local_frame_dim
            + self.cfg.camera_feature_dim
            + 1
            + self.cfg.part_count
            + 24
        )
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.SiLU(),
        )
        self.residual_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.rgb_delta_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.normal_delta_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.occupancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.confidence_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.part_logits_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count)
        self.control_margin_head = nn.Linear(self.cfg.hidden_dim, 1)

    @staticmethod
    def reject_forbidden(batch: Mapping[str, Any]) -> None:
        reject_forbidden_inference_keys(batch)

    def forward(self, batch: Mapping[str, Tensor], *, control: str = "true") -> dict[str, Tensor]:
        self.reject_forbidden(batch)
        if control not in CONTROL_MODES:
            raise ValueError(f"unknown control mode: {control}")

        vggt_points = _xyz("vggt_world_points", _require(batch, "vggt_world_points"))
        vggt_rgb = _feature("vggt_rgb", _require(batch, "vggt_rgb"), vggt_points, 3).clamp(0.0, 1.0)
        vggt_conf = _scalar_feature("vggt_confidence", _require(batch, "vggt_confidence"), vggt_points).clamp_min(0.0)
        vggt_features = _feature("vggt_features", _require(batch, "vggt_features"), vggt_points, self.cfg.vggt_feature_dim)
        smplx_features = _feature("smplx_graph_features", _require(batch, "smplx_graph_features"), vggt_points, self.cfg.smplx_feature_dim)
        smplx_normal = _norm(_feature("smplx_normal", _require(batch, "smplx_normal"), vggt_points, 3))
        smplx_local_frame = _feature("smplx_local_frame", _require(batch, "smplx_local_frame"), vggt_points, self.cfg.local_frame_dim)
        camera_features = _feature("camera_features", _require(batch, "camera_features"), vggt_points, self.cfg.camera_feature_dim)
        human_mask = _scalar_feature("human_mask", _require(batch, "human_mask"), vggt_points).clamp(0.0, 1.0)
        part_ids = _part_ids(batch.get("body_part_id"), vggt_points, self.cfg.part_count)

        if control == "vggt_baseline":
            smplx_features = torch.zeros_like(smplx_features)
            smplx_normal = torch.zeros_like(smplx_normal)
            smplx_local_frame = torch.zeros_like(smplx_local_frame)
            part_ids = torch.zeros_like(part_ids)
        elif control == "no_smpl":
            smplx_features = torch.zeros_like(smplx_features)
            smplx_normal = torch.zeros_like(smplx_normal)
            smplx_local_frame = torch.zeros_like(smplx_local_frame)
            part_ids = torch.zeros_like(part_ids)
        elif control == "shuffled_semantic":
            flat = smplx_features.reshape(-1, smplx_features.shape[-1])
            perm = torch.randperm(flat.shape[0], device=flat.device)
            smplx_features = flat[perm].reshape_as(smplx_features)
            part_flat = part_ids.reshape(-1)
            part_perm = torch.randperm(part_flat.shape[0], device=part_flat.device)
            part_ids = part_flat[part_perm].reshape_as(part_ids)
        elif control == "same_topology_no_semantic":
            smplx_features = smplx_features[..., :1].expand_as(smplx_features) * 0.0
            part_ids = torch.zeros_like(part_ids)
        elif control == "shell_only":
            vggt_features = torch.zeros_like(vggt_features)
            vggt_rgb = torch.zeros_like(vggt_rgb)
            vggt_conf = torch.zeros_like(vggt_conf)
        elif control == "observation_only":
            smplx_features = torch.zeros_like(smplx_features)
            smplx_local_frame = torch.zeros_like(smplx_local_frame)
        elif control == "visible_only":
            human_mask = torch.ones_like(human_mask)

        part_one_hot = F.one_hot(part_ids, num_classes=self.cfg.part_count).float()
        part_embedding = self.part_embedding(part_ids)
        x = torch.cat(
            [
                vggt_points,
                vggt_rgb,
                vggt_conf,
                vggt_features,
                smplx_features,
                smplx_normal,
                smplx_local_frame,
                camera_features,
                human_mask,
                part_one_hot,
                part_embedding,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        visibility = torch.sigmoid(self.visibility_head(h)) * human_mask
        occupancy = torch.sigmoid(self.occupancy_head(h)) * human_mask
        confidence = torch.sigmoid(self.confidence_head(h)) * vggt_conf.clamp(0.0, 1.0)
        residual = torch.tanh(self.residual_head(h)) * self.cfg.max_residual * visibility
        student_points = vggt_points + residual
        student_rgb = (vggt_rgb + torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta * visibility).clamp(0.0, 1.0)
        student_normal = _norm(smplx_normal + torch.tanh(self.normal_delta_head(h)) * self.cfg.max_normal_delta)

        out: dict[str, Tensor] = {
            "student_human_points": student_points,
            "student_human_rgb": student_rgb,
            "residual": residual,
            "surfel_occupancy": occupancy,
            "surfel_visibility": visibility,
            "student_confidence": confidence,
            "student_normal": student_normal,
            "part_logits": self.part_logits_head(h),
            "control_margin_logits": self.control_margin_head(h),
            "model_owned_student_output": torch.tensor(True, device=vggt_points.device),
            "no_teacher_points_inference": torch.tensor(True, device=vggt_points.device),
            "no_v50r2_inference": torch.tensor(True, device=vggt_points.device),
            "no_kinect_depth_inference": torch.tensor(True, device=vggt_points.device),
        }

        if "environment_points" in batch:
            out["environment_points"] = _xyz("environment_points", batch["environment_points"])
        if "environment_rgb" in batch:
            env = batch["environment_rgb"].float().clamp(0.0, 1.0)
            if env.shape[-1] != 3:
                raise ValueError("environment_rgb must end with RGB dimension 3")
            out["environment_rgb"] = env
        return out


def build_smoke_batch(cfg: V506ObservationDistilledStudentConfig, *, batch: int = 2, nodes: int = 256) -> dict[str, Tensor]:
    gen = torch.Generator().manual_seed(506)
    return {
        "vggt_world_points": torch.randn(batch, nodes, 3, generator=gen) * 0.25,
        "vggt_rgb": torch.rand(batch, nodes, 3, generator=gen),
        "vggt_confidence": torch.rand(batch, nodes, generator=gen),
        "vggt_features": torch.randn(batch, nodes, cfg.vggt_feature_dim, generator=gen) * 0.1,
        "smplx_graph_features": torch.randn(batch, nodes, cfg.smplx_feature_dim, generator=gen) * 0.1,
        "smplx_normal": F.normalize(torch.randn(batch, nodes, 3, generator=gen), dim=-1),
        "smplx_local_frame": torch.randn(batch, nodes, cfg.local_frame_dim, generator=gen) * 0.05,
        "camera_features": torch.randn(batch, nodes, cfg.camera_feature_dim, generator=gen) * 0.1,
        "human_mask": (torch.rand(batch, nodes, generator=gen) > 0.15).float(),
        "body_part_id": torch.randint(0, cfg.part_count, (batch, nodes), generator=gen),
        "environment_points": torch.randn(batch, 64, 3, generator=gen) * 0.5,
        "environment_rgb": torch.rand(batch, 64, 3, generator=gen),
    }


def smoke_test() -> dict[str, Any]:
    cfg = V506ObservationDistilledStudentConfig()
    model = V506V50R2ObservationDistilledStudent(cfg)
    batch = build_smoke_batch(cfg)
    out = model(batch)
    loss = (
        out["student_human_points"].square().mean()
        + out["student_human_rgb"].mean()
        + out["surfel_occupancy"].mean()
        + out["surfel_visibility"].mean()
        + out["student_confidence"].mean()
        + out["part_logits"].square().mean()
        + out["control_margin_logits"].square().mean()
    )
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)

    forbidden_rejected = {}
    for key in ("teacher_points", "v50r2_teacher_bank", "kinect_depth", "raw_depth"):
        bad = dict(batch)
        bad[key] = torch.zeros_like(batch["vggt_world_points"])
        try:
            model(bad)
            forbidden_rejected[key] = False
        except ValueError:
            forbidden_rejected[key] = True

    control_shapes: dict[str, list[int]] = {}
    with torch.no_grad():
        for control in CONTROL_MODES:
            control_shapes[control] = list(model(batch, control=control)["student_human_points"].shape)

    return {
        "config": asdict(cfg),
        "output_shapes": {k: list(v.shape) for k, v in out.items() if isinstance(v, Tensor) and v.ndim > 0},
        "grad_norm_positive": grad_norm > 0,
        "forbidden_keys_rejected": forbidden_rejected,
        "all_forbidden_rejected": all(forbidden_rejected.values()),
        "control_modes": list(CONTROL_MODES),
        "control_shapes": control_shapes,
        "model_owned_student_output": bool(out["model_owned_student_output"].item()),
        "no_teacher_points_inference": bool(out["no_teacher_points_inference"].item()),
        "no_v50r2_inference": bool(out["no_v50r2_inference"].item()),
        "no_kinect_depth_inference": bool(out["no_kinect_depth_inference"].item()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
