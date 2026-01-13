"""Feature extraction from dog pose keypoints."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

KEYPOINT_NAMES: list[str] = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
]

_KEYPOINT_INDEX = {name: index for index, name in enumerate(KEYPOINT_NAMES)}

_HEAD_KEYPOINTS = (
    "left_ear_base",
    "right_ear_base",
    "left_ear_tip",
    "right_ear_tip",
    "nose",
    "chin",
    "left_eye",
    "right_eye",
    "throat",
)
_LIMB_KEYPOINTS = (
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
)
_PAW_KEYPOINTS = (
    "front_left_paw",
    "front_right_paw",
    "rear_left_paw",
    "rear_right_paw",
)

_HEAD_INDICES = tuple(_KEYPOINT_INDEX[name] for name in _HEAD_KEYPOINTS)
_LIMB_INDICES = tuple(_KEYPOINT_INDEX[name] for name in _LIMB_KEYPOINTS)
_PAW_INDICES = tuple(_KEYPOINT_INDEX[name] for name in _PAW_KEYPOINTS)

_HORIZONTAL = np.array([1.0, 0.0], dtype=np.float32)
_MIN_SCALE = 1e-6


def dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_product = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_product < _MIN_SCALE:
        return float(np.nan)
    cosine = float(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0))
    return float(np.arccos(cosine))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return vector_angle(a - b, c - b)


def nan_if_missing(visible_flags: Iterable[bool], value: float) -> float:
    return float(value) if all(visible_flags) else float(np.nan)


class FeatureExtractor:
    """Extracts frame-level features from 24 dog pose keypoints."""

    # fixed order of features
    FEATURE_NAMES = [
        "visible_ratio",
        "head_visible_ratio",
        "limbs_visible_ratio",
        "v_mean",
        "v_min",
        "v_head_min",
        "v_limbs_min",
        "front_span",
        "rear_span",
        "torso_len",
        "tail_len",
        "body_axis_angle",
        "head_body_angle",
        "front_leg_fold_angle",
        "withers_to_paws_y",
        "front_symmetry",
        "rear_symmetry",
        "head_yaw",
    ]

    def extract_frame_features(
        self, keypoints: Sequence[Sequence[float]]
    ) -> Dict[str, float]:

        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (len(KEYPOINT_NAMES), 3):
            raise ValueError(
                f"keypoints must have shape (24, 3); got {points.shape}."
            )

        xy = points[:, :2]
        raw_visibility = points[:, 2]
        visible = raw_visibility > 0
        idx = _KEYPOINT_INDEX

        # --- Visibility / quality features ---
        visible_ratio = float(np.mean(visible))
        head_visible_ratio = float(np.mean(visible[list(_HEAD_INDICES)]))
        limbs_visible_ratio = float(np.mean(visible[list(_LIMB_INDICES)]))

        v_mean = float(np.mean(raw_visibility))
        v_min = float(np.min(raw_visibility))
        v_head_min = float(np.min(raw_visibility[list(_HEAD_INDICES)]))
        v_limbs_min = float(np.min(raw_visibility[list(_LIMB_INDICES)]))

        # --- Scale (nose ↔ tail_start) ---
        nose_idx = idx["nose"]
        tail_start_idx = idx["tail_start"]
        scale = (
            dist(xy[nose_idx], xy[tail_start_idx])
            if visible[nose_idx] and visible[tail_start_idx]
            else float("nan")
        )
        scale_valid = bool(np.isfinite(scale) and scale > _MIN_SCALE)

        def normalized(value: float, required_indices: Sequence[int]) -> float:
            if not scale_valid:
                return float(np.nan)
            return nan_if_missing((visible[i] for i in required_indices), value / scale)

        # --- Size / geometry ---
        front_span = normalized(
            dist(xy[idx["front_left_paw"]], xy[idx["front_right_paw"]]),
            [idx["front_left_paw"], idx["front_right_paw"]],
        )
        rear_span = normalized(
            dist(xy[idx["rear_left_paw"]], xy[idx["rear_right_paw"]]),
            [idx["rear_left_paw"], idx["rear_right_paw"]],
        )
        torso_len = normalized(
            dist(xy[idx["withers"]], xy[idx["tail_start"]]),
            [idx["withers"], idx["tail_start"]],
        )
        tail_len = normalized(
            dist(xy[idx["tail_start"]], xy[idx["tail_end"]]),
            [idx["tail_start"], idx["tail_end"]],
        )

        # --- Orientation ---
        body_axis_angle = nan_if_missing(
            (visible[idx["withers"]], visible[idx["tail_start"]]),
            vector_angle(
                xy[idx["tail_start"]] - xy[idx["withers"]],
                _HORIZONTAL,
            ),
        )
        head_body_angle = nan_if_missing(
            (
                visible[idx["nose"]],
                visible[idx["withers"]],
                visible[idx["tail_start"]],
            ),
            vector_angle(
                xy[idx["nose"]] - xy[idx["withers"]],
                xy[idx["tail_start"]] - xy[idx["withers"]],
            ),
        )

        # --- Limb articulation ---
        front_leg_required = [
            idx["front_left_paw"],
            idx["front_left_knee"],
            idx["front_left_elbow"],
            idx["front_right_paw"],
            idx["front_right_knee"],
            idx["front_right_elbow"],
        ]
        if all(visible[i] for i in front_leg_required):
            left_angle = angle(
                xy[idx["front_left_paw"]],
                xy[idx["front_left_knee"]],
                xy[idx["front_left_elbow"]],
            )
            right_angle = angle(
                xy[idx["front_right_paw"]],
                xy[idx["front_right_knee"]],
                xy[idx["front_right_elbow"]],
            )
            front_leg_fold_angle = float(np.mean([left_angle, right_angle]))
        else:
            front_leg_fold_angle = float(np.nan)

        # --- Height / symmetry ---
        withers_idx = idx["withers"]
        paws_y = xy[list(_PAW_INDICES), 1]
        withers_to_paws_y = normalized(
            float(abs(np.mean(paws_y) - xy[withers_idx, 1])),
            [withers_idx, *_PAW_INDICES],
        )

        front_symmetry = normalized(
            abs(
                dist(xy[withers_idx], xy[idx["front_left_paw"]])
                - dist(xy[withers_idx], xy[idx["front_right_paw"]])
            ),
            [withers_idx, idx["front_left_paw"], idx["front_right_paw"]],
        )
        rear_symmetry = normalized(
            abs(
                dist(xy[idx["tail_start"]], xy[idx["rear_left_paw"]])
                - dist(xy[idx["tail_start"]], xy[idx["rear_right_paw"]])
            ),
            [idx["tail_start"], idx["rear_left_paw"], idx["rear_right_paw"]],
        )

        # --- Head yaw (signed, normalized) ---
        if scale_valid and all(
            visible[i] for i in (idx["nose"], idx["left_eye"], idx["right_eye"])
        ):
            mid_eye_x = 0.5 * (
                xy[idx["left_eye"], 0] + xy[idx["right_eye"], 0]
            )
            head_yaw = float((xy[idx["nose"], 0] - mid_eye_x) / scale)
        else:
            head_yaw = float(np.nan)

        return {
            "visible_ratio": visible_ratio,
            "head_visible_ratio": head_visible_ratio,
            "limbs_visible_ratio": limbs_visible_ratio,
            "v_mean": v_mean,
            "v_min": v_min,
            "v_head_min": v_head_min,
            "v_limbs_min": v_limbs_min,
            "front_span": front_span,
            "rear_span": rear_span,
            "torso_len": torso_len,
            "tail_len": tail_len,
            "body_axis_angle": body_axis_angle,
            "head_body_angle": head_body_angle,
            "front_leg_fold_angle": front_leg_fold_angle,
            "withers_to_paws_y": withers_to_paws_y,
            "front_symmetry": front_symmetry,
            "rear_symmetry": rear_symmetry,
            "head_yaw": head_yaw,
        }
def extract_frame_vector(self, keypoints) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (D,) aligned with FEATURE_NAMES.
            Missing values are NaN.
        """
        feats = self.extract_frame_features(keypoints)
        return np.asarray([feats[name] for name in self.FEATURE_NAMES], dtype=np.float32)

def get_feature_names(self) -> list[str]:
        return list(self.FEATURE_NAMES)