"""Run YOLO11 Pose inference and export keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict YOLO11 Pose and save keypoints.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/pose/train/weights/best.pt",
        help="Path to weights file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path, '0', or 'camera' for webcam.",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save overlay video to outputs/overlay/.",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source == "0" or source.lower() == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Unable to open video source.")
    return cap


def get_output_paths(source: str) -> tuple[Path, Path, str]:
    if source == "0" or source.lower() == "camera":
        name = "camera"
    else:
        name = Path(source).stem

    keypoint_path = Path("outputs/keypoints") / f"{name}.jsonl"
    overlay_path = Path("outputs/overlay") / f"{name}.mp4"
    return keypoint_path, overlay_path, name


def main() -> int:
    args = parse_args()

    model = YOLO(args.weights)
    cap = open_capture(args.source)

    keypoint_path, overlay_path, _ = get_output_paths(args.source)
    keypoint_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_overlay:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_index = 0
    with keypoint_path.open("w", encoding="utf-8") as handle:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame, verbose=False)[0]
            detections = []

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                for idx in range(len(result.keypoints.xy)):
                    xy = result.keypoints.xy[idx].cpu().tolist()
                    conf = result.keypoints.conf[idx].cpu().tolist()
                    keypoints = [[x, y, c] for (x, y), c in zip(xy, conf)]
                    bbox = None
                    score = None
                    if result.boxes is not None and len(result.boxes.xyxy) > idx:
                        bbox = result.boxes.xyxy[idx].cpu().tolist()
                        score = float(result.boxes.conf[idx].item())
                    detections.append({"keypoints": keypoints, "bbox": bbox, "score": score})

            payload = {"frame_index": frame_index, "detections": detections}
            handle.write(json.dumps(payload) + "\n")

            if writer is not None:
                writer.write(result.plot())

            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"Saved keypoints to {keypoint_path}")
    if writer is not None:
        print(f"Saved overlay video to {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
