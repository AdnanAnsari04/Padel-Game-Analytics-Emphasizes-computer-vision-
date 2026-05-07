"""
classifier.py
-------------
Rule-based and pose-based shot classifier for padel.

Shot types recognised:
  - forehand
  - backhand
  - serve / smash
  - volley
  - unknown

Strategy
--------
1. Detect racket/wrist position relative to the player bounding box.
2. Use MediaPipe Pose (if available) for precise wrist & elbow angles.
3. Fall back to a geometric heuristic using bbox positions.
"""

import math
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ShotEvent:
    shot_type: str
    confidence: float
    frame: int
    timestamp_sec: float
    player_id: Optional[int]
    player_bbox: Optional[Tuple]
    racket_bbox: Optional[Tuple]
    ball_center: Optional[Tuple]


class PoseEstimator:
    """Wraps MediaPipe Pose. Gracefully degrades if unavailable."""

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
            print("[PoseEstimator] MediaPipe Pose loaded.")
        except Exception as e:
            print(f"[PoseEstimator] MediaPipe unavailable ({e}). Using bbox heuristics.")
            self.available = False

    def get_landmarks(self, frame, player_bbox):
        """Returns landmark dict or None."""
        if not self.available:
            return None
        import mediapipe as mp
        x1, y1, x2, y2 = player_bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        rgb = crop[:, :, ::-1].copy()
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None

        h, w = crop.shape[:2]
        lm = results.pose_landmarks.landmark
        # Return absolute pixel coords for key joints
        def to_abs(idx):
            pt = lm[idx]
            return (x1 + pt.x * w, y1 + pt.y * h, pt.visibility)

        return {
            "left_wrist":  to_abs(mp.solutions.pose.PoseLandmark.LEFT_WRIST),
            "right_wrist": to_abs(mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
            "left_elbow":  to_abs(mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
            "right_elbow": to_abs(mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
            "left_shoulder":  to_abs(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
            "right_shoulder": to_abs(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
            "left_hip":  to_abs(mp.solutions.pose.PoseLandmark.LEFT_HIP),
            "right_hip": to_abs(mp.solutions.pose.PoseLandmark.RIGHT_HIP),
        }


def _angle_deg(a, b, c) -> float:
    """Angle at point b formed by a-b-c (2-D)."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    cos_val = (ax * cx + ay * cy) / (
        math.sqrt(ax**2 + ay**2) * math.sqrt(cx**2 + cy**2) + 1e-9
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


class ShotClassifier:
    """
    Classifies padel shots from per-frame pose / bbox features.

    Internal state machine:
      - 'idle'   : no swing in progress
      - 'swing'  : swing detected, accumulating frames
    """

    SWING_COOLDOWN = 15  # minimum frames between shots

    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self._last_shot_frame = -999
        self._wrist_history = []   # recent wrist y positions (for smash detection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        frame,
        frame_idx: int,
        fps: float,
        player_bbox: Optional[Tuple],
        racket_bbox: Optional[Tuple],
        ball_center: Optional[Tuple],
        player_id: Optional[int] = None,
    ) -> Optional[ShotEvent]:
        """
        Returns a ShotEvent if a shot is detected, else None.
        """
        if player_bbox is None:
            return None
        if frame_idx - self._last_shot_frame < self.SWING_COOLDOWN:
            return None

        shot_type, conf = self._classify_shot(
            frame, frame_idx, player_bbox, racket_bbox, ball_center
        )
        if shot_type == "unknown":
            return None

        self._last_shot_frame = frame_idx
        return ShotEvent(
            shot_type=shot_type,
            confidence=conf,
            frame=frame_idx,
            timestamp_sec=frame_idx / fps,
            player_id=player_id,
            player_bbox=player_bbox,
            racket_bbox=racket_bbox,
            ball_center=ball_center,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_shot(self, frame, frame_idx, player_bbox, racket_bbox, ball_center):
        """Returns (shot_type, confidence)."""
        landmarks = self.pose_estimator.get_landmarks(frame, player_bbox)

        if landmarks:
            return self._pose_based(landmarks, player_bbox, racket_bbox, ball_center)
        else:
            return self._bbox_heuristic(player_bbox, racket_bbox, ball_center)

    def _pose_based(self, lm, player_bbox, racket_bbox, ball_center):
        """Use joint angles to classify shot."""
        px1, py1, px2, py2 = player_bbox
        player_cx = (px1 + px2) / 2

        rw = lm["right_wrist"]
        lw = lm["left_wrist"]
        rs = lm["right_shoulder"]
        ls = lm["left_shoulder"]
        re = lm["right_elbow"]
        le = lm["left_elbow"]

        # Determine dominant (swing) side
        r_vis, l_vis = rw[2], lw[2]
        if r_vis > l_vis:
            wrist, elbow, shoulder = rw, re, rs
            side = "right"
        else:
            wrist, elbow, shoulder = lw, le, ls
            side = "left"

        # Wrist height relative to shoulder → smash / serve
        wrist_above_shoulder = wrist[1] < shoulder[1] - 20
        elbow_angle = _angle_deg(shoulder, elbow, wrist)

        # Wrist x relative to player centre → forehand vs backhand
        wrist_x = wrist[0]

        if wrist_above_shoulder and elbow_angle > 120:
            return "smash", 0.80

        # Forehand: dominant wrist on same side as racket
        if racket_bbox:
            rc_x = (racket_bbox[0] + racket_bbox[2]) / 2
            if abs(rc_x - wrist_x) < 80:
                if wrist_x > player_cx and side == "right":
                    return "forehand", 0.75
                elif wrist_x < player_cx and side == "left":
                    return "forehand", 0.70
                else:
                    return "backhand", 0.70
        else:
            if wrist_x > player_cx:
                return "forehand", 0.65
            else:
                return "backhand", 0.65

        return "volley", 0.55

    def _bbox_heuristic(self, player_bbox, racket_bbox, ball_center):
        """
        Fallback: use bounding box geometry only.
        """
        px1, py1, px2, py2 = player_bbox
        player_cx = (px1 + px2) / 2
        player_cy = (py1 + py2) / 2
        player_h = py2 - py1

        if racket_bbox:
            rx1, ry1, rx2, ry2 = racket_bbox
            racket_cx = (rx1 + rx2) / 2
            racket_cy = (ry1 + ry2) / 2

            # Racket high above player → smash / serve
            if racket_cy < py1 + player_h * 0.2:
                return "smash", 0.72

            # Racket close to net (low, forward) → volley
            if ry2 > py1 + player_h * 0.55 and abs(racket_cx - player_cx) < 60:
                return "volley", 0.60

            # Side heuristic
            if racket_cx > player_cx + 20:
                return "forehand", 0.65
            elif racket_cx < player_cx - 20:
                return "backhand", 0.65
            else:
                return "forehand", 0.50

        if ball_center:
            bx, by = ball_center
            if by < py1 + player_h * 0.3:
                return "smash", 0.55
            if bx > player_cx:
                return "forehand", 0.50
            else:
                return "backhand", 0.50

        return "unknown", 0.0
