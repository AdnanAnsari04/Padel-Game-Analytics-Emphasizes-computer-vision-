"""
detector.py
-----------
Ball, Racket, and Player detection using YOLOv8.
Falls back to a simple background-subtraction tracker if YOLO is unavailable.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class Detection:
    label: str          # 'ball', 'racket', 'player'
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 for person + sports-ball detection."""

    COCO_LABELS = {0: "person", 32: "sports ball", 38: "tennis racket"}

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.3):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf = conf
            self.available = True
            print(f"[YOLODetector] Loaded model: {model_path}")
        except Exception as e:
            print(f"[YOLODetector] YOLO not available ({e}). Using fallback detector.")
            self.available = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.COCO_LABELS:
                continue
            label_raw = self.COCO_LABELS[cls_id]
            # Normalise labels
            if label_raw == "person":
                label = "player"
            elif label_raw == "sports ball":
                label = "ball"
            else:
                label = "racket"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(Detection(label=label, bbox=(x1, y1, x2, y2), confidence=conf))
        return detections


class FallbackDetector:
    """
    Simple MOG2 background-subtraction detector used when YOLO is unavailable.
    Detects moving blobs as 'ball' / 'player' by size heuristic.
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if area < 500:
                label = "ball"
            else:
                label = "player"
            detections.append(
                Detection(label=label, bbox=(x, y, x + w, y + h), confidence=0.6)
            )
        return detections


class MultiObjectTracker:
    """
    Lightweight IoU-based tracker that assigns persistent IDs to detections.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter_area
        return inter_area / union if union > 0 else 0.0

    def update(self, detections: List[Detection]) -> Dict[int, Detection]:
        """Returns {track_id: Detection}"""
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        assigned = {}
        unmatched = list(detections)

        for tid, track in self.tracks.items():
            best_iou = 0.0
            best_det = None
            for det in unmatched:
                if det.label != track["label"]:
                    continue
                iou = self._iou(track["bbox"], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det is not None and best_iou >= self.iou_threshold:
                self.tracks[tid]["bbox"] = best_det.bbox
                self.tracks[tid]["age"] = 0
                assigned[tid] = best_det
                unmatched.remove(best_det)

        for det in unmatched:
            self.tracks[self.next_id] = {"label": det.label, "bbox": det.bbox, "age": 0}
            assigned[self.next_id] = det
            self.next_id += 1

        return assigned
