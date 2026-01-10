"""Download or extract the Stanford Dog-Pose dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_ROOT = Path("data/dog-pose")
DEFAULT_ZIP_NAME = "dog-pose.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Dog-Pose dataset.")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=str,
        default=None,
        help="Path to a downloaded zip file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional URL to download the dataset zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset directory if present.",
    )
    return parser.parse_args()


def download_zip(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {target} ...")
    urllib.request.urlretrieve(url, target)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Dataset directory is not empty: {output_dir}. Use --force to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def normalize_layout(root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "Expected 'images/' and 'labels/' directories under data/dog-pose."
        )

    val_dir = images_dir / "val"
    test_dir = images_dir / "test"
    if not val_dir.exists() and test_dir.exists():
        try:
            val_dir.symlink_to(test_dir, target_is_directory=True)
            (labels_dir / "val").symlink_to(labels_dir / "test", target_is_directory=True)
        except OSError:
            shutil.copytree(test_dir, val_dir)
            shutil.copytree(labels_dir / "test", labels_dir / "val")


def main() -> int:
    args = parse_args()
    dataset_dir = DATASET_ROOT

    if dataset_dir.exists() and args.force:
        shutil.rmtree(dataset_dir)

    zip_path = Path(args.zip_path) if args.zip_path else None

    if args.url and not zip_path:
        zip_path = dataset_dir.parent / DEFAULT_ZIP_NAME
        download_zip(args.url, zip_path)

    if not zip_path:
        print(
            "No download URL provided. Please download the Dog-Pose dataset zip manually, "
            "then re-run with --zip /path/to/file.zip."
        )
        return 1

    if not zip_path.exists():
        print(f"Zip file not found: {zip_path}")
        return 1

    extract_zip(zip_path, dataset_dir)
    normalize_layout(dataset_dir)

    meta = {
        "dataset_dir": str(dataset_dir.resolve()),
        "images": str((dataset_dir / "images").resolve()),
        "labels": str((dataset_dir / "labels").resolve()),
    }
    print("Dataset ready:")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
