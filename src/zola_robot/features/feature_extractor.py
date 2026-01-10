"""Feature extraction from pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class FeatureExtractor:
    """Extracts numeric features from pose landmarks.

    Landmarks should be shaped (N, 2) or (N, 3) with normalized coordinates.
    """

    distance_pairs: Sequence[tuple[int, int]] = field(
        default_factory=lambda: (
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
            (11, 12),
            (23, 24),
            (11, 23),
            (12, 24),
        )
    )
    angle_triplets: Sequence[tuple[int, int, int]] = field(
        default_factory=lambda: (
            (11, 13, 15),
            (12, 14, 16),
            (23, 25, 27),
            (24, 26, 28),
        )
    )

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """Return a 1D feature vector from landmarks."""

        if landmarks is None:
            raise ValueError("Landmarks are required for feature extraction.")

        points = np.asarray(landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Landmarks must have shape (N, 2) or (N, 3).")

        xy = points[:, :2]
        scale = self._compute_scale(xy)

        features: list[float] = []
        for a, b in self.distance_pairs:
            if a >= len(xy) or b >= len(xy):
                features.append(0.0)
                continue
            dist = np.linalg.norm(xy[a] - xy[b]) / scale
            features.append(float(dist))

        for a, b, c in self.angle_triplets:
            if a >= len(xy) or b >= len(xy) or c >= len(xy):
                features.append(0.0)
                continue
            angle = self._angle_between(xy[a], xy[b], xy[c])
            features.append(float(angle))

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _compute_scale(points: np.ndarray) -> float:
        """Use bounding box diagonal to normalize distances."""

        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        scale = float(np.linalg.norm(max_xy - min_xy))
        return scale if scale > 1e-6 else 1.0

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point b formed by a-b-c, in radians."""

        ba = a - b
        bc = c - b
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0))
        return float(np.arccos(cosine))
