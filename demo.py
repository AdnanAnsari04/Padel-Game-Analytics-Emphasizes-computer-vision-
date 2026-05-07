"""
demo.py
-------
Runs the pipeline on a synthetic (generated) padel video so you can
verify the installation and see sample output without needing the real
match footage.

Usage:
    python demo.py
    python demo.py --frames 300 --output output/
"""

import argparse
import os
import sys
import cv2
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from classifier import ShotClassifier
from analytics import ShotAnalytics
from visualizer import Visualizer


# ── synthetic video generator ───────────────────────────────────────────────

def _make_padel_frame(w, h, frame_idx):
    """Renders a minimal padel-court frame with a bouncing 'ball'."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Court background
    frame[:] = (30, 80, 30)

    # Court lines
    margin_x, margin_y = 60, 40
    cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y),
                  (200, 200, 200), 2)
    # Net
    mid_y = h // 2
    cv2.line(frame, (margin_x, mid_y), (w - margin_x, mid_y), (220, 220, 220), 3)
    # Service box
    cx = w // 2
    cv2.line(frame, (cx, margin_y), (cx, h - margin_y), (200, 200, 200), 1)

    # "Player" bounding box
    p1_x, p1_y = cx - 80, int(h * 0.65)
    p_w, p_h = 60, 120
    cv2.rectangle(frame, (p1_x, p1_y), (p1_x + p_w, p1_y + p_h), (0, 140, 255), -1)
    cv2.putText(frame, "P1", (p1_x + 10, p1_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    p2_x, p2_y = cx + 20, int(h * 0.15)
    cv2.rectangle(frame, (p2_x, p2_y), (p2_x + p_w, p2_y + p_h), (255, 80, 80), -1)
    cv2.putText(frame, "P2", (p2_x + 10, p2_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # "Racket" next to P1
    r_x = p1_x + p_w + 5
    r_y = p1_y + 20
    cv2.ellipse(frame, (r_x + 15, r_y + 10), (20, 25), 0, 0, 360, (180, 130, 60), 2)
    cv2.line(frame, (r_x + 15, r_y + 35), (r_x + 15, r_y + 70), (180, 130, 60), 3)

    # Bouncing ball
    t = frame_idx / 30.0
    bx = int(cx + (cx - margin_x - 20) * 0.7 * math.sin(t * 1.3))
    by = int(h * 0.5 + h * 0.3 * math.sin(t * 2.1))
    cv2.circle(frame, (bx, by), 8, (0, 220, 255), -1)
    cv2.circle(frame, (bx, by), 8, (255, 255, 255), 1)

    return frame, (p1_x, p1_y, p1_x + p_w, p1_y + p_h), \
                  (r_x, r_y, r_x + 30, r_y + 70), (bx, by)


# ── demo runner ─────────────────────────────────────────────────────────────

def run_demo(n_frames: int = 300, fps: float = 30.0, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    W, H = 640, 480

    classifier = ShotClassifier()
    analytics  = ShotAnalytics()
    viz        = Visualizer()

    from detector import Detection, MultiObjectTracker
    tracker = MultiObjectTracker()

    out_video_path = os.path.join(output_dir, "demo_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    print(f"[Demo] Generating {n_frames} synthetic frames…")

    for fidx in range(1, n_frames + 1):
        frame, p_bbox, r_bbox, b_center = _make_padel_frame(W, H, fidx)

        # Synthetic detections (no real YOLO needed)
        px1, py1, px2, py2 = p_bbox
        rx1, ry1, rx2, ry2 = r_bbox
        bx, by = b_center

        dets = [
            Detection("player", (px1, py1, px2, py2), 0.95),
            Detection("racket", (rx1, ry1, rx2, ry2), 0.88),
            Detection("ball",   (bx-8, by-8, bx+8, by+8), 0.92),
        ]
        tracked = tracker.update(dets)

        # Pick the first player track
        p_id = next((tid for tid, d in tracked.items() if d.label == "player"), None)

        shot_event = classifier.classify(
            frame, fidx, fps,
            p_bbox, r_bbox, b_center, p_id
        )
        if shot_event:
            analytics.add(shot_event)

        annotated = viz.draw(frame, tracked, shot_event, fidx, fps)
        writer.write(annotated)

    writer.release()

    analytics.print_summary()
    analytics.to_json(os.path.join(output_dir, "shot_predictions.json"))
    analytics.to_csv(os.path.join(output_dir,  "shot_predictions.csv"))
    viz.draw_analytics_dashboard(viz.shot_counts,
                                 os.path.join(output_dir, "analytics_dashboard.png"))

    print(f"\n[Demo] Finished. Files saved to: {output_dir}/")
    print(f"       Annotated video : {out_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    run_demo(n_frames=args.frames, output_dir=args.output)
