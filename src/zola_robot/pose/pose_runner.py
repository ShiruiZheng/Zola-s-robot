"""Pose runner entry point for YOLO11 Pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .yolo_pose_runner import YoloPoseRunner


@dataclass
class PoseResult:
    """Holds pose keypoints and an annotated frame."""

    landmarks: Optional[np.ndarray]
    annotated_frame: np.ndarray
    bbox: Optional[list[float]] = None
    score: Optional[float] = None


class PoseRunner:
    """Lightweight wrapper around a YOLO11 Pose estimator."""

    def __init__(
        self,
        weights_path: str = "yolo11n-pose.pt",
    ) -> None:
        self._yolo = YoloPoseRunner(weights_path=weights_path)

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> PoseResult:
        """Run pose estimation on a single frame."""

        payload = self._yolo.run_frame(frame)
        annotated = self._yolo.annotate_frame(frame) if draw else frame.copy()
        landmarks = None
        if payload["keypoints"] is not None:
            landmarks = np.asarray(payload["keypoints"], dtype=np.float32)
        return PoseResult(
            landmarks=landmarks,
            annotated_frame=annotated,
            bbox=payload["bbox"],
            score=payload["score"],
        )

    def close(self) -> None:
        """Release resources."""

        return None

    def __enter__(self) -> "PoseRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["PoseResult", "PoseRunner", "YoloPoseRunner"]
