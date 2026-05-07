"""
main.py
-------
Padel Game Analytics — Shot Classification Pipeline
Entry point. Ties together detection, tracking, classification, and output.

Usage:
    python main.py --input path/to/video.mp4
    python main.py --input path/to/video.mp4 --output output/ --no-display
    python main.py --input path/to/video.mp4 --yolo yolov8n.pt --conf 0.35
"""

import argparse
import os
import sys
import cv2
from collections import Counter

# Add src to path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from detector import YOLODetector, FallbackDetector, MultiObjectTracker
from classifier import ShotClassifier
from analytics import ShotAnalytics
from visualizer import Visualizer


def parse_args():
    p = argparse.ArgumentParser(description="Padel Shot Analytics Pipeline")
    p.add_argument("--input",   required=True, help="Path to input video file")
    p.add_argument("--output",  default="output", help="Output directory")
    p.add_argument("--yolo",    default="yolov8n.pt", help="YOLO weights path")
    p.add_argument("--conf",    type=float, default=0.30, help="Detection confidence threshold")
    p.add_argument("--no-display", action="store_true", help="Skip live preview window")
    p.add_argument("--save-video", action="store_true", help="Save annotated output video")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Process every N-th frame (1 = all frames)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Open video                                                          #
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Video: {width}x{height} @ {fps:.1f}fps — {total_f} frames")

    # ------------------------------------------------------------------ #
    #  Initialise components                                               #
    # ------------------------------------------------------------------ #
    yolo_det   = YOLODetector(model_path=args.yolo, conf=args.conf)
    detector   = yolo_det if yolo_det.available else FallbackDetector()
    tracker    = MultiObjectTracker(iou_threshold=0.3, max_age=12)
    classifier = ShotClassifier()
    analytics  = ShotAnalytics()
    viz        = Visualizer(show_trail=True)

    # Optional video writer
    writer = None
    if args.save_video:
        out_video_path = os.path.join(args.output, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Will save annotated video → {out_video_path}")

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx % args.skip_frames) != 0:
            continue
        processed += 1

        # 1. Detect
        detections = detector.detect(frame)

        # 2. Track
        tracked = tracker.update(detections)

        # 3. Identify primary player + racket + ball
        player_bbox = None
        racket_bbox = None
        ball_center = None
        player_id   = None

        for tid, det in tracked.items():
            if det.label == "player" and player_bbox is None:
                player_bbox = det.bbox
                player_id   = tid
            elif det.label == "racket" and racket_bbox is None:
                racket_bbox = det.bbox
            elif det.label == "ball" and ball_center is None:
                ball_center = det.center

        # 4. Classify shot
        shot_event = classifier.classify(
            frame, frame_idx, fps,
            player_bbox, racket_bbox, ball_center, player_id
        )
        if shot_event:
            analytics.add(shot_event)
            print(f"  [Shot] frame={frame_idx:05d}  type={shot_event.shot_type:<10}"
                  f"  conf={shot_event.confidence:.2f}  player={player_id}")

        # 5. Visualise
        annotated = viz.draw(frame, tracked, shot_event, frame_idx, fps)

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Padel Analytics", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Pipeline] Quit by user.")
                break

        if processed % 100 == 0:
            pct = 100 * frame_idx / max(total_f, 1)
            print(f"  Progress: frame {frame_idx}/{total_f} ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    #  Cleanup & outputs                                                   #
    # ------------------------------------------------------------------ #
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    analytics.print_summary()

    json_path = os.path.join(args.output, "shot_predictions.json")
    csv_path  = os.path.join(args.output, "shot_predictions.csv")
    analytics.to_json(json_path)
    analytics.to_csv(csv_path)

    dashboard_path = os.path.join(args.output, "analytics_dashboard.png")
    viz.draw_analytics_dashboard(viz.shot_counts, dashboard_path)

    print(f"\n[Pipeline] Done. Outputs written to: {args.output}/")


if __name__ == "__main__":
    main()
