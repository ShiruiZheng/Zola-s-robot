"""Video source helpers for camera or local files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass(frozen=True)
class VideoSource:
    """Represents a supported video source.

    source_type: "camera" or "file".
    path: Local file path if source_type == "file".
    """

    source_type: str
    path: Optional[Path] = None


def parse_video_source(source: str, camera_index: int = 0) -> tuple[VideoSource, int]:
    """Parse a user-provided source string into a VideoSource.

    Supported inputs:
    - "camera": built-in Mac webcam or iPhone Continuity Camera via cv2.VideoCapture(0)
    - Local file path
    """

    if source.lower() == "camera":
        return VideoSource(source_type="camera"), camera_index

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Video source is not a file: {path}")

    return VideoSource(source_type="file", path=path), camera_index


def open_video_capture(source: str, camera_index: int = 0) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture from a supported source.

    The only supported camera option is index 0 (Mac webcam or Continuity Camera).
    """

    video_source, index = parse_video_source(source, camera_index=camera_index)
    if video_source.source_type == "camera":
        capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(str(video_source.path))

    if not capture.isOpened():
        raise RuntimeError("Unable to open video source. Check camera permissions or file path.")

    return capture
