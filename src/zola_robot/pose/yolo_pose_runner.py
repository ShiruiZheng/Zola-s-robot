"""YOLO11 Pose runner for extracting keypoints from video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ultralytics import YOLO


class YoloPoseRunner:
    """Wrapper around a YOLO11 Pose model."""

    def __init__(self, weights_path: str | Path = "yolo11n-pose.pt") -> None:
        resolved = self._resolve_model_path(Path(weights_path))
        self._model = YOLO(str(resolved))
        self._last_result: Optional[Any] = None

    def run_frame(self, frame) -> dict:
        """Run pose estimation on a single frame and return 24x3 keypoints."""

        result = self._model(frame, verbose=False)[0]
        self._last_result = result

        payload = {
            "keypoints": None,
            "bbox": None,
            "score": None,
        }

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return payload

        best_index = 0
        if result.boxes is not None and len(result.boxes.conf) > 0:
            best_index = int(result.boxes.conf.argmax().item())
            payload["score"] = float(result.boxes.conf[best_index].item())
            payload["bbox"] = result.boxes.xyxy[best_index].cpu().tolist()

        xy = result.keypoints.xy[best_index].cpu().tolist()
        conf = result.keypoints.conf[best_index].cpu().tolist()
        keypoints = [[x, y, c] for (x, y), c in zip(xy, conf)]
        payload["keypoints"] = keypoints

        return payload

    def annotate_frame(self, frame):
        """Return an annotated frame using the last inference result."""

        if self._last_result is None:
            return frame
        return self._last_result.plot()

    @staticmethod
    def _resolve_model_path(path: Path) -> Path:
        if path.is_absolute() or path.exists():
            return path
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / path
        return candidate if candidate.exists() else path
