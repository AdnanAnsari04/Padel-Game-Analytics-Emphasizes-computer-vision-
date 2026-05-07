"""
analytics.py
------------
Aggregates ShotEvents and exports JSON / CSV reports.
Also provides shot-count summary statistics.
"""

import json
import csv
import os
from collections import defaultdict, Counter
from typing import List, Optional
from dataclasses import asdict

from classifier import ShotEvent


class ShotAnalytics:
    def __init__(self):
        self.events: List[ShotEvent] = []

    def add(self, event: ShotEvent):
        self.events.append(event)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        total = len(self.events)
        if total == 0:
            return {"total_shots": 0}

        type_counts = Counter(e.shot_type for e in self.events)
        player_counts = defaultdict(Counter)
        for e in self.events:
            pid = e.player_id if e.player_id is not None else "unknown"
            player_counts[str(pid)][e.shot_type] += 1

        durations = []
        for i in range(1, len(self.events)):
            durations.append(self.events[i].timestamp_sec - self.events[i-1].timestamp_sec)
        avg_interval = sum(durations) / len(durations) if durations else 0

        return {
            "total_shots": total,
            "shot_type_counts": dict(type_counts),
            "shots_per_player": {k: dict(v) for k, v in player_counts.items()},
            "avg_interval_between_shots_sec": round(avg_interval, 2),
            "first_shot_frame": self.events[0].frame,
            "last_shot_frame": self.events[-1].frame,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "summary": self.summary(),
            "shots": [
                {
                    "frame": e.frame,
                    "timestamp_sec": round(e.timestamp_sec, 3),
                    "shot_type": e.shot_type,
                    "confidence": round(e.confidence, 3),
                    "player_id": e.player_id,
                    "ball_center": e.ball_center,
                }
                for e in self.events
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Analytics] JSON saved → {path}")

    def to_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            "frame", "timestamp_sec", "shot_type",
            "confidence", "player_id", "ball_center_x", "ball_center_y"
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.events:
                bc = e.ball_center or (None, None)
                writer.writerow({
                    "frame": e.frame,
                    "timestamp_sec": round(e.timestamp_sec, 3),
                    "shot_type": e.shot_type,
                    "confidence": round(e.confidence, 3),
                    "player_id": e.player_id,
                    "ball_center_x": bc[0],
                    "ball_center_y": bc[1],
                })
        print(f"[Analytics] CSV saved → {path}")

    def print_summary(self):
        s = self.summary()
        print("\n" + "="*50)
        print("  PADEL SHOT ANALYTICS SUMMARY")
        print("="*50)
        print(f"  Total shots detected : {s.get('total_shots', 0)}")
        for stype, cnt in s.get("shot_type_counts", {}).items():
            print(f"    {stype:<12}: {cnt}")
        print(f"  Avg interval         : {s.get('avg_interval_between_shots_sec', 0):.2f}s")
        print("="*50 + "\n")
