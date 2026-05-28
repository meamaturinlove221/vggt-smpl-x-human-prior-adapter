from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


IMAGE_SIZE = 518


@dataclass(frozen=True)
class ViewChoice:
    axis_x: int = 0
    axis_y: int = 1
    flip_x: bool = False
    flip_y: bool = False


@dataclass
class MorphologyRegion:
    name: str
    point_mask: np.ndarray
    image_crop: tuple[int, int, int, int]
    point_count: int


class True3DMorphologyAdapter:
    """Small helper for selecting mentor-readable 3D views and local crops.

    This module intentionally stays light. The route script feeds it real
    prediction and detail-source arrays and uses the returned view/crop choices
    to build the final mentor evidence.
    """

    @staticmethod
    def choose_upright_view(points: np.ndarray, *, prefer_xy: bool = True) -> ViewChoice:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return ViewChoice()
        # The current dataset is aligned closely enough that the x/y plane is
        # usually the most human-readable mentor view. Keep a small fallback
        # search so obvious inverted cases can be corrected without touching the
        # model output.
        candidates = [
            ViewChoice(0, 1, False, False),
            ViewChoice(0, 1, False, True),
            ViewChoice(0, 1, True, False),
            ViewChoice(0, 1, True, True),
            ViewChoice(0, 2, False, False),
            ViewChoice(1, 2, False, False),
        ]
        if not prefer_xy:
            candidates = candidates[4:] + candidates[:4]
        best = candidates[0]
        best_score = -1e9
        for cand in candidates:
            xy = True3DMorphologyAdapter.project_xy(pts, cand)
            lo = xy.min(axis=0)
            hi = xy.max(axis=0)
            span = np.maximum(hi - lo, 1e-6)
            aspect = float(span[1] / span[0])
            area = float(span[0] * span[1])
            center = (lo + hi) * 0.5
            # Favor an upright, legible, compact human-main view with enough
            # area to keep environment visible around the subject.
            score = 0.0
            score += 2.0 * (1.0 - abs(aspect - 1.15))
            score += 0.8 * np.tanh(area * 1.2)
            score -= 0.15 * abs(center[0])
            score -= 0.10 * abs(center[1])
            if score > best_score:
                best_score = score
                best = cand
        return best

    @staticmethod
    def project_xy(points: np.ndarray, view: ViewChoice | None = None) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        view = view or ViewChoice()
        axes = pts[:, [view.axis_x, view.axis_y]].copy()
        if view.flip_x:
            axes[:, 0] *= -1.0
        if view.flip_y:
            axes[:, 1] *= -1.0
        return axes

    @staticmethod
    def center_and_span(points: np.ndarray, view: ViewChoice | None = None) -> tuple[np.ndarray, np.ndarray]:
        xy = True3DMorphologyAdapter.project_xy(points, view)
        if len(xy) == 0:
            return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        return (lo + hi) * 0.5, np.maximum(hi - lo, 1e-6)

    @staticmethod
    def limit_box(
        human_points: np.ndarray,
        environment_points: np.ndarray,
        *,
        view: ViewChoice | None = None,
        human_margin: float = 0.72,
        env_hint_count: int = 1200,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        h_xy = True3DMorphologyAdapter.project_xy(human_points, view)
        e_xy = True3DMorphologyAdapter.project_xy(environment_points, view)
        if len(h_xy) == 0:
            return (-1.0, 1.0), (-1.0, 1.0)
        h_lo = h_xy.min(axis=0)
        h_hi = h_xy.max(axis=0)
        center = (h_lo + h_hi) * 0.5
        radius = max(float((h_hi - h_lo).max()) * human_margin, 0.20)
        if len(e_xy):
            env_near = e_xy[np.argsort(np.linalg.norm(e_xy - center[None], axis=1))[: min(env_hint_count, len(e_xy))]]
            lo = np.minimum(center - radius, np.percentile(env_near, 4, axis=0))
            hi = np.maximum(center + radius, np.percentile(env_near, 96, axis=0))
            center = (lo + hi) * 0.5
            radius = max(float((hi - lo).max()) * 0.52, radius)
        xlim = (float(center[0] - radius), float(center[0] + radius))
        ylim = (float(center[1] - radius), float(center[1] + radius))
        return xlim, ylim

    @staticmethod
    def local_crop_from_points(points: np.ndarray, *, view: ViewChoice | None = None, margin: float = 0.10) -> tuple[int, int, int, int]:
        xy = True3DMorphologyAdapter.project_xy(points, view)
        if len(xy) == 0:
            return (0, 0, IMAGE_SIZE, IMAGE_SIZE)
        lo = np.percentile(xy, 3, axis=0)
        hi = np.percentile(xy, 97, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        pad = np.maximum(span * margin, 10.0)
        x0 = int(max(0, np.floor(lo[0] - pad[0])))
        y0 = int(max(0, np.floor(lo[1] - pad[1])))
        x1 = int(min(IMAGE_SIZE, np.ceil(hi[0] + pad[0])))
        y1 = int(min(IMAGE_SIZE, np.ceil(hi[1] + pad[1])))
        if x1 <= x0:
            x1 = min(IMAGE_SIZE, x0 + 32)
        if y1 <= y0:
            y1 = min(IMAGE_SIZE, y0 + 32)
        return (x0, y0, x1, y1)

    @staticmethod
    def region_from_mask(
        name: str,
        point_mask: np.ndarray,
        points: np.ndarray,
        *,
        view: ViewChoice | None = None,
    ) -> MorphologyRegion:
        mask = np.asarray(point_mask, dtype=bool)
        pts = np.asarray(points, dtype=np.float32)
        selected = pts[mask] if len(pts) else pts
        crop = True3DMorphologyAdapter.local_crop_from_points(selected, view=view)
        return MorphologyRegion(name=name, point_mask=mask, image_crop=crop, point_count=int(mask.sum()))

    @staticmethod
    def summarize_region(region: MorphologyRegion) -> dict[str, Any]:
        return {
            "name": region.name,
            "point_count": region.point_count,
            "image_crop": list(region.image_crop),
        }
