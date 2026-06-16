from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.v184_canonical_surfel_graph_occupancy_student import (  # noqa: E402
    CanonicalSurfelGraphOccupancyConfig,
    CanonicalSurfelGraphOccupancyStudent,
)


class ProposalConditionedDecoder(CanonicalSurfelGraphOccupancyStudent):
    """Canonical surfel decoder conditioned on learned local edit proposals.

    V206 used proposal scores only in losses and post-decode selection. This
    model injects proposal, lock, seed, weak, and body-part context before the
    occupancy/residual decisions are made.
    """

    def __init__(self, cfg: CanonicalSurfelGraphOccupancyConfig | None = None) -> None:
        super().__init__(cfg)
        cfg = self.cfg
        proposal_dim = 4 + cfg.part_count
        self.proposal_encoder = nn.Sequential(
            nn.Linear(proposal_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
            nn.GELU(),
        )
        self.conditioned_occupancy_head = nn.Linear(cfg.hidden_dim, 1)
        self.conditioned_residual_head = nn.Linear(cfg.hidden_dim, 3)
        self.nonregression_head = nn.Linear(cfg.hidden_dim, 1)

    @staticmethod
    def _proposal_feature(batch: Mapping[str, Tensor], body_one_hot: Tensor) -> Tensor:
        xyz = next(iter(batch.values()))
        bsz, points = body_one_hot.shape[:2]
        device = xyz.device
        def optional(name: str) -> Tensor:
            value = batch.get(name)
            if value is None:
                return torch.zeros(bsz, points, 1, device=device)
            value = value.float()
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            return value

        proposal = optional("proposal_score").clamp(0, 1)
        lock = optional("visible_lock").clamp(0, 1)
        seed = optional("proposal_seed").clamp(0, 1)
        weak = optional("weak_score").clamp(0, 1)
        return torch.cat([proposal, lock, seed, weak, body_one_hot.float()], dim=-1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.reject_forbidden(batch)
        surfel_xyz = self._first(batch, "surfel_xyz", "canonical_surfel_xyz").float()
        surfel_rgb = self._first(batch, "surfel_rgb", "anchor_rgb").float().clamp(0.0, 1.0)
        normal = F.normalize(self._first(batch, "surfel_normal", "normal").float(), dim=-1, eps=1e-6)
        tangent = F.normalize(self._first(batch, "surfel_tangent", "tangent").float(), dim=-1, eps=1e-6)
        binormal = F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1, eps=1e-6)
        confidence = self._first(batch, "vggt_confidence", "confidence").float()
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        bsz, points, _ = surfel_xyz.shape
        surfel_features = self._optional(batch, "surfel_features", (bsz, points, self.cfg.surfel_feature_dim), surfel_xyz.device).float()
        vggt_features = self._optional(batch, "vggt_features", (bsz, points, self.cfg.vggt_feature_dim), surfel_xyz.device).float()
        token_context = self._first(batch, "vggt_token_context", "token_context").float()
        if token_context.ndim == 2:
            token_context = token_context.unsqueeze(1).expand(-1, points, -1)
        elif token_context.shape[1] != points:
            token_context = F.interpolate(token_context.transpose(1, 2), size=points, mode="linear", align_corners=False).transpose(1, 2)
        if "body_part_id" in batch:
            body = batch["body_part_id"].long().clamp(0, self.cfg.part_count - 1)
            part_one_hot = F.one_hot(body, num_classes=self.cfg.part_count).float()
        else:
            part_one_hot = self._optional(batch, "body_part_one_hot", (bsz, points, self.cfg.part_count), surfel_xyz.device).float()
        base_x = torch.cat(
            [
                surfel_xyz,
                surfel_rgb,
                normal,
                tangent,
                binormal,
                confidence,
                surfel_features,
                vggt_features,
                token_context,
                part_one_hot,
            ],
            dim=-1,
        )
        h = self.trunk(base_x)
        p = self.proposal_encoder(self._proposal_feature(batch, part_one_hot))
        conditioned = h + p
        nonregression = torch.sigmoid(self.nonregression_head(conditioned))
        proposal = batch.get("proposal_score")
        if proposal is None:
            proposal = torch.zeros(bsz, points, 1, device=surfel_xyz.device)
        elif proposal.ndim == 2:
            proposal = proposal.unsqueeze(-1)
        lock = batch.get("visible_lock")
        if lock is None:
            lock = torch.zeros(bsz, points, 1, device=surfel_xyz.device)
        elif lock.ndim == 2:
            lock = lock.unsqueeze(-1)

        occupancy_logits = self.occupancy_head(h) + self.conditioned_occupancy_head(conditioned) * (0.35 + proposal)
        occupancy_logits = occupancy_logits - 1.8 * lock * nonregression
        occupancy = torch.sigmoid(occupancy_logits)
        visibility_logits = self.visibility_head(h)
        visibility = torch.sigmoid(visibility_logits)
        local_base = self.residual_local_head(h)
        local_cond = self.conditioned_residual_head(conditioned)
        local = torch.tanh(local_base + local_cond * (0.25 + proposal)) * self.cfg.max_residual
        local = local * (1.0 - lock * nonregression)
        residual = tangent * local[..., 0:1] + binormal * local[..., 1:2] + normal * local[..., 2:3]
        thickness = torch.sigmoid(self.thickness_head(conditioned)) * self.cfg.max_thickness
        normal_delta = torch.tanh(self.normal_delta_head(conditioned)) * 0.22
        refined_normal = F.normalize(normal + normal_delta, dim=-1, eps=1e-6)
        front_shell = surfel_xyz + residual + refined_normal * thickness[..., 0:1]
        back_shell = surfel_xyz + residual - refined_normal * thickness[..., 1:2]
        student_xyz = 0.5 * (front_shell + back_shell)
        rgb_delta = torch.tanh(self.rgb_delta_head(conditioned)) * self.cfg.max_rgb_delta * visibility * (1.0 - lock)
        student_rgb = (surfel_rgb + rgb_delta).clamp(0.0, 1.0)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
            "occupancy_logits": occupancy_logits,
            "occupancy": occupancy,
            "visibility_logits": visibility_logits,
            "visibility": visibility,
            "residual_xyz": residual,
            "front_shell": front_shell,
            "back_shell": back_shell,
            "thickness": thickness,
            "normal": refined_normal,
            "part_graph_logits": self.part_graph_head(conditioned),
            "part_exclusion_logits": self.exclusion_head(conditioned),
            "nonregression": nonregression,
            "model_owned_student_output": torch.tensor(True, device=surfel_xyz.device),
            "no_teacher_points_inference": torch.tensor(True, device=surfel_xyz.device),
            "no_raw_kinect_depth_inference": torch.tensor(True, device=surfel_xyz.device),
        }


def smoke_test() -> dict[str, Any]:
    cfg = CanonicalSurfelGraphOccupancyConfig()
    model = ProposalConditionedDecoder(cfg)
    gen = torch.Generator()
    gen.manual_seed(207)
    body = torch.randint(0, cfg.part_count, (1, 256), generator=gen)
    batch: dict[str, Tensor] = {
        "surfel_xyz": torch.randn(1, 256, 3, generator=gen) * 0.15,
        "surfel_rgb": torch.rand(1, 256, 3, generator=gen),
        "surfel_normal": F.normalize(torch.randn(1, 256, 3, generator=gen), dim=-1),
        "surfel_tangent": F.normalize(torch.randn(1, 256, 3, generator=gen), dim=-1),
        "vggt_confidence": torch.rand(1, 256, generator=gen),
        "surfel_features": torch.randn(1, 256, cfg.surfel_feature_dim, generator=gen) * 0.1,
        "vggt_features": torch.randn(1, 256, cfg.vggt_feature_dim, generator=gen) * 0.1,
        "vggt_token_context": torch.randn(1, cfg.token_dim, generator=gen) * 0.1,
        "body_part_id": body,
        "proposal_score": torch.rand(1, 256, 1, generator=gen),
        "visible_lock": torch.rand(1, 256, 1, generator=gen).gt(0.82).float(),
        "proposal_seed": torch.rand(1, 256, 1, generator=gen).gt(0.88).float(),
        "weak_score": torch.rand(1, 256, 1, generator=gen),
    }
    out = model(batch)
    loss = out["student_xyz"].square().mean() + out["occupancy"].mean() + out["nonregression"].mean()
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    bad = dict(batch)
    bad["teacher_points"] = torch.zeros_like(batch["surfel_xyz"])
    forbidden_rejected = False
    try:
        model(bad)
    except ValueError:
        forbidden_rejected = True
    return {
        "student_xyz_shape": list(out["student_xyz"].shape),
        "occupancy_shape": list(out["occupancy"].shape),
        "nonregression_shape": list(out["nonregression"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_teacher_points_rejected": forbidden_rejected,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
