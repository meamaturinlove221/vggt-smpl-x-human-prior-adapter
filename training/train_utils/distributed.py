# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import torch


def _read_rank_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    world_size = _read_rank_env("WORLD_SIZE", 1)
    if world_size <= 1:
        return 0, 0

    local_rank = _read_rank_env("LOCAL_RANK", 0)
    distributed_rank = _read_rank_env("RANK", 0)
    return local_rank, distributed_rank
