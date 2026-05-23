from __future__ import annotations

import torch
from torch import nn


class SurfaceTokenEncoder(nn.Module):
    def __init__(self, semantic_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.net(semantic)


class ObservationEncoder(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class SupportMaskOnly(nn.Module):
    def forward(self, support: torch.Tensor) -> torch.Tensor:
        # Support is allowed to provide only reliability/mask. Dense content channels are ignored.
        mask = support[..., :1].sigmoid()
        return mask


class CameraBoundTransportDecoder(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.delta_point = nn.Linear(hidden, 3)
        self.delta_normal = nn.Linear(hidden, 3)
        self.confidence = nn.Linear(hidden, 1)

    def forward(self, semantic_tokens: torch.Tensor, observation_tokens: torch.Tensor, support_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = semantic_tokens + 0.35 * observation_tokens
        attended, _ = self.attn(tokens, tokens, tokens, key_padding_mask=None)
        gated = attended * support_mask.clamp(0.0, 1.0)
        normal = torch.nn.functional.normalize(self.delta_normal(gated), dim=-1, eps=1e-6)
        return {
            "delta_point": self.delta_point(gated),
            "learned_delta_normal": normal,
            "confidence": self.confidence(gated).sigmoid(),
        }


class CameraBoundPointTransformer(nn.Module):
    def __init__(self, semantic_dim: int = 81, observation_dim: int = 9, support_dim: int = 5, hidden: int = 96) -> None:
        super().__init__()
        self.semantic = SurfaceTokenEncoder(semantic_dim, hidden)
        self.observation = ObservationEncoder(observation_dim, hidden)
        self.support = SupportMaskOnly()
        self.decoder = CameraBoundTransportDecoder(hidden)

    def forward(self, semantic: torch.Tensor, observation: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic_tokens = self.semantic(semantic)
        observation_tokens = self.observation(observation)
        support_mask = self.support(support)
        return self.decoder(semantic_tokens, observation_tokens, support_mask)


def smoke() -> dict[str, object]:
    torch.manual_seed(401)
    model = CameraBoundPointTransformer()
    n = 2048
    semantic = torch.randn(2, n, 81)
    observation = torch.randn(2, n, 9)
    support = torch.randn(2, n, 5)
    out = model(semantic, observation, support)
    support2 = support.clone()
    support2[..., 1:] = torch.randn_like(support2[..., 1:]) * 100.0
    out2 = model(semantic, observation, support2)
    semantic2 = semantic.clone()
    semantic2[..., :8] = torch.randn_like(semantic2[..., :8])
    out3 = model(semantic2, observation, support)
    return {
        "delta_point_shape": list(out["delta_point"].shape),
        "learned_delta_normal_shape": list(out["learned_delta_normal"].shape),
        "confidence_shape": list(out["confidence"].shape),
        "support_dense_content_delta": float((out["delta_point"] - out2["delta_point"]).abs().max().item()),
        "semantic_swap_delta": float((out["delta_point"] - out3["delta_point"]).abs().mean().item()),
        "normal_nonzero_ratio": float((out["learned_delta_normal"].abs().sum(dim=-1) > 0.1).float().mean().item()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke(), indent=2, sort_keys=True))
