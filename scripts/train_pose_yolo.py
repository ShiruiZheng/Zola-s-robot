"""Train YOLO11 Pose on Dog-Pose dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = "configs/datasets/dog-pose.yaml"
DEFAULT_MODEL = "yolo11n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 Pose baseline.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")


    model = YOLO(args.model)
    model.train(
         data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project="runs/pose",
        name="train",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
