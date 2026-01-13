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

## Feature Schema (v1)
Frame-level features extracted from 24 dog keypoints (Ultralytics Dog-Pose).
All distances are normalized by scale = ||nose - tail_start|| when available.

| Feature              | Type  | Range   | Meaning                                                       |                                                     |                                  |   |         |
| -------------------- | ----- | ------- | ------------------------------------------------------------- | --------------------------------------------------- | -------------------------------- | - | ------- |
| visible_ratio        | float | [0,1]   | Fraction of keypoints with v>0                                |                                                     |                                  |   |         |
| head_visible_ratio   | float | [0,1]   | Fraction of head keypoints with v>0                           |                                                     |                                  |   |         |
| limbs_visible_ratio  | float | [0,1]   | Fraction of limb keypoints with v>0                           |                                                     |                                  |   |         |
| v_mean               | float | [0,2]   | Mean COCO visibility value over 24 kpts                       |                                                     |                                  |   |         |
| v_min                | float | {0,1,2} | Minimum visibility over 24 kpts                               |                                                     |                                  |   |         |
| v_head_min           | float | {0,1,2} | Minimum visibility over head kpts                             |                                                     |                                  |   |         |
| v_limbs_min          | float | {0,1,2} | Minimum visibility over limb kpts                             |                                                     |                                  |   |         |
| front_span           | float | ~[0,+]  |                                                               |                                                     | front_left_paw - front_right_paw |   | / scale |
| rear_span            | float | ~[0,+]  |                                                               |                                                     | rear_left_paw - rear_right_paw   |   | / scale |
| torso_len            | float | ~[0,+]  |                                                               |                                                     | withers - tail_start             |   | / scale |
| tail_len             | float | ~[0,+]  |                                                               |                                                     | tail_start - tail_end            |   | / scale |
| body_axis_angle      | float | [0,π]   | Angle between body axis (withers→tail_start) and horizontal   |                                                     |                                  |   |         |
| head_body_angle      | float | [0,π]   | Angle between (withers→nose) and (withers→tail_start)         |                                                     |                                  |   |         |
| front_leg_fold_angle | float | [0,π]   | Mean knee angle for both front legs (paw-knee-elbow)          |                                                     |                                  |   |         |
| withers_to_paws_y    | float | ~[0,+]  | Vertical distance between withers and mean paw y, normalized  |                                                     |                                  |   |         |
| front_symmetry       | float | ~[0,+]  |                                                               | dist(withers, FL paw) - dist(withers, FR paw)       | / scale                          |   |         |
| rear_symmetry        | float | ~[0,+]  |                                                               | dist(tail_start, RL paw) - dist(tail_start, RR paw) | / scale                          |   |         |
| head_yaw             | float | ~[-,+]  | Signed horizontal nose offset vs mid-eye, normalized by scale |                                                     |                                  |   |         |

Missing data handling: If required keypoints are not available (v==0), features are returned as NaN.
Fixed ordering for ML: See FeatureExtractor.FEATURE_NAMES.

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

### 11 Jan 2026 Pose Feature Engineering (v1)
Designed and implemented a frame-level pose feature extraction pipeline
Features are derived from normalized distances, angles, symmetry, and visibility
Introduced:
Fixed-order feature vectors (ML-ready)
Explicit feature schema for interpretability and debugging
Verified feature behavior on short video clips
Established pose → feature separation (clean interface for future models)

### Current status:
- YOLO11-Pose successfully fine-tuned on Stanford Dog-Pose
- Pose inference from image/video is complete
- Frame-level pose feature extractor implemented
🔄 Feature set under active iteration and validation


## Overview:
### Pipeline Overview
Video / Image
→ YOLO11 Pose (24 keypoints)
→ Frame-level pose features
→ Temporal aggregation (next)
→ Action recognition (future)

### What’s Next
- Temporal window features
- Sequence models (LSTM / Transformer)
