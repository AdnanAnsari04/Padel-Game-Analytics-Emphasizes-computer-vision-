"""
visualizer.py
-------------
Draws detection boxes, track IDs, shot labels, and analytics overlay
onto video frames.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque, Counter


# Colour palette per label
COLOURS = {
    "player":  (0, 200, 50),
    "ball":    (0, 220, 255),
    "racket":  (255, 140, 0),
    "forehand":  (50, 220, 50),
    "backhand":  (50, 50, 255),
    "smash":     (255, 50, 50),
    "volley":    (200, 50, 200),
    "unknown":   (180, 180, 180),
}
SHOT_COLOURS = {k: v for k, v in COLOURS.items()
                if k in ("forehand", "backhand", "smash", "volley", "unknown")}


class Visualizer:
    def __init__(self, show_trail: bool = True, trail_len: int = 30):
        self.ball_trail: deque = deque(maxlen=trail_len)
        self.show_trail = show_trail
        self.shot_flash: Optional[dict] = None   # {type, colour, ttl}
        self.shot_counts: Counter = Counter()

    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        tracked: Dict,
        shot_event=None,
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> np.ndarray:
        out = frame.copy()

        # ---- update ball trail ----
        for tid, det in tracked.items():
            if det.label == "ball":
                self.ball_trail.append(det.center)

        # ---- draw trail ----
        if self.show_trail and len(self.ball_trail) > 1:
            pts = list(self.ball_trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = (int(0 * alpha), int(220 * alpha), int(255 * alpha))
                cv2.line(out, (int(pts[i-1][0]), int(pts[i-1][1])),
                         (int(pts[i][0]), int(pts[i][1])), c, 2)

        # ---- draw detections ----
        for tid, det in tracked.items():
            colour = COLOURS.get(det.label, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label_text = f"{det.label} #{tid}"
            cv2.putText(out, label_text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        # ---- shot flash ----
        if shot_event:
            self.shot_counts[shot_event.shot_type] += 1
            colour = SHOT_COLOURS.get(shot_event.shot_type, (255, 255, 255))
            self.shot_flash = {"type": shot_event.shot_type, "colour": colour, "ttl": 25}

        if self.shot_flash and self.shot_flash["ttl"] > 0:
            sf = self.shot_flash
            h, w = out.shape[:2]
            overlay = out.copy()
            cv2.rectangle(overlay, (w//2 - 150, h//2 - 40), (w//2 + 150, h//2 + 40),
                          sf["colour"], -1)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.putText(out, sf["type"].upper(), (w//2 - 130, h//2 + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            sf["ttl"] -= 1

        # ---- HUD ----
        out = self._draw_hud(out, frame_idx, fps)
        return out

    def _draw_hud(self, frame, frame_idx, fps):
        h, w = frame.shape[:2]
        # Semi-transparent panel
        panel = frame.copy()
        cv2.rectangle(panel, (8, 8), (230, 30 + 22 * (len(self.shot_counts) + 2)),
                      (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

        ts = frame_idx / fps
        cv2.putText(frame, f"Frame: {frame_idx}  T: {ts:.1f}s", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        y = 52
        cv2.putText(frame, "Shot Counts:", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        y += 22
        total = sum(self.shot_counts.values())
        for stype, cnt in sorted(self.shot_counts.items()):
            col = SHOT_COLOURS.get(stype, (200, 200, 200))
            cv2.putText(frame, f"  {stype}: {cnt}", (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            y += 20
        cv2.putText(frame, f"  Total: {total}", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def draw_analytics_dashboard(self, shot_counts: Counter, output_path: str):
        """Save a static analytics chart image using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = list(shot_counts.keys())
            values = list(shot_counts.values())
            colours_hex = ["#32dc32", "#3232ff", "#ff3232", "#c832c8", "#aaaaaa"]

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor("#1a1a2e")

            # Bar chart
            ax1 = axes[0]
            ax1.set_facecolor("#16213e")
            bars = ax1.bar(labels, values,
                           color=colours_hex[:len(labels)], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                         str(val), ha="center", va="bottom", color="white", fontsize=11)
            ax1.set_title("Shots by Type", color="white", fontsize=13)
            ax1.set_xlabel("Shot Type", color="white")
            ax1.set_ylabel("Count", color="white")
            ax1.tick_params(colors="white")
            for spine in ax1.spines.values():
                spine.set_edgecolor("#444")

            # Pie chart
            ax2 = axes[1]
            ax2.set_facecolor("#16213e")
            wedges, texts, autotexts = ax2.pie(
                values, labels=labels, colors=colours_hex[:len(labels)],
                autopct="%1.1f%%", startangle=90,
                textprops={"color": "white"}
            )
            for at in autotexts:
                at.set_color("white")
            ax2.set_title("Shot Distribution", color="white", fontsize=13)

            plt.suptitle("Padel Shot Analytics", color="white", fontsize=15, y=1.02)
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
            print(f"[Visualizer] Dashboard saved → {output_path}")
        except Exception as e:
            print(f"[Visualizer] Could not generate dashboard: {e}")
