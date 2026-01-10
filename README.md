# Zola's Robot

Dog action recognition baseline focused on pose-based features. This repo avoids any human identification and only produces keypoints and numeric features.

## Project Structure

```text
Zola-s-robot/
├── notebooks/
│   ├── 01_input_sources.ipynb
│   ├── 02_pose_demo.ipynb
│   └── 03_feature_extraction.ipynb
├── data/
│   └── dog-pose/
│       ├── images/{train,val}
│       └── labels/{train,val}
├── configs/
│   └── datasets/
│       └── dog-pose.yaml
├── scripts/
│   ├── download_dogpose.py
│   ├── train_pose_yolo.py
│   └── predict_pose_yolo.py
├── src/
│   └── zola_robot/
│       ├── io/
│       │   └── video_sources.py
│       ├── pose/
│       │   ├── pose_runner.py
│       │   └── yolo_pose_runner.py
│       └── features/
│           └── feature_extractor.py
├── requirements.txt
└── README.md
```

## Supported Input Sources

- Mac webcam or iPhone Continuity Camera via `cv2.VideoCapture(0)`
- Local video file path

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Baseline Pipeline

1) YOLO11 Pose detects keypoints.
2) `zola_robot.features.feature_extractor.FeatureExtractor` computes the feature vector.
3) Features are ready for downstream classification or clustering.

## YOLO11 Pose + Dog-Pose baseline

Baseline: YOLO11 Pose + Stanford Dog-Pose (24 keypoints with x, y, visibility).

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare dataset:

- If you already downloaded a zip file:

```bash
python scripts/download_dogpose.py --zip /path/to/dog-pose.zip
```

- If you have an official URL:

```bash
python scripts/download_dogpose.py --url https://example.com/dog-pose.zip
```

Dataset layout (matches Ultralytics Dog-Pose guidance):

```text
data/dog-pose/
  images/{train,val}
  labels/{train,val}
```

Train:

```bash
python scripts/train_pose_yolo.py \
  --model models/yolo11/yolo11n-pose.pt \
  --data configs/datasets/dog-pose.yaml
```

Predict (save keypoints to `outputs/keypoints/*.jsonl`):

```bash
python scripts/predict_pose_yolo.py --weights runs/pose/train/weights/best.pt --source /path/to/video.mp4
```

Predict from webcam with overlay:

```bash
python scripts/predict_pose_yolo.py --weights runs/pose/train/weights/best.pt --source 0 --save_overlay
```

Explanation:
- YOLO11 Pose outputs keypoints.
- Feature vectors are computed by `zola_robot.features.feature_extractor.FeatureExtractor`.

Note: Model weights (.pt files) are not tracked in GitHub.
After training, the best checkpoint is saved locally under:
`runs/pose/train*/weights/best.pt`


## Step-by-Step Run Instructions

1) Open Jupyter Lab or Notebook

```bash
jupyter lab
```

2) Run input sources demo

- Open `notebooks/01_input_sources.ipynb`
- Set `SOURCE = 'camera'` or a local video file path
- Run all cells to verify capture

3) Run pose demo

- Open `notebooks/02_pose_demo.ipynb`
- Set `SOURCE = 'camera'` or a local video file path
- Run all cells to see YOLO11 pose keypoints overlay

4) Extract features

- Open `notebooks/03_feature_extraction.ipynb`
- Set `SOURCE = 'camera'` or a local video file path
- Run all cells to print the feature vector

## Notes

- Pose backend is YOLO11.
- The pose model is trained once and saved locally under:
`runs/pose/train*/weights/best.pt`.
- All demos and feature extraction scripts load a pretrained checkpoint
and do not retrain the pose model.
- Feature vectors are computed by `zola_robot.features.feature_extractor.FeatureExtractor`.
- Current baseline assumes the Stanford Dog-Pose dataset for training/evaluation.


## Project Log / Milestones 
### 7-9 Jan 2026
Completed fine-tuning YOLO11-Pose on the Ultralytics Stanford Dog-Pose dataset.
The system can now robustly detect 24 dog keypoints from images and videos.
### 10 Jan 2026
- Begin pose-based feature engineering:
Start with frame-level pose features for interpretability and debugging
Then aggregate over short temporal windows for robust action recognition
- Long-term plan:
Compare classical temporal features with LSTM / Transformer
Explore end-to-end video models as advanced baselines

### Current status:
- YOLO11-Pose successfully fine-tuned on Stanford Dog-Pose
- Pose inference from image/video is complete
- Feature extraction pipeline is implemented and under active development


## Overview:
### Pipeline Overview
Video / Image
→ YOLO11 Pose (24 keypoints)
→ Frame-level pose features
→ Temporal aggregation (next)
→ Action recognition (future)

### What’s Next
- Align keypoints with semantic names
- Frame-level pose features
- Temporal window features
- Sequence models (LSTM / Transformer)
