# 🎾 Padel Game Analytics — Shot Classification System

> **Layman AI Internship Assignment** | Computer Vision & ML Prototype

---

## 📌 Overview

This project builds a **computer vision pipeline** that analyses padel match footage to:

1. **Detect & track** the ball, racket, and players frame-by-frame  
2. **Classify shots** — Forehand, Backhand, Smash/Serve, Volley  
3. **Output structured results** — JSON + CSV with shot type, timestamp, frame, and player ID  
4. **Visualise** — annotated output video + analytics dashboard  

---

## 🗂️ Project Structure

```
padel_analytics/
│
├── main.py              ← Full pipeline (real video input)
├── demo.py              ← Synthetic demo (no video required)
├── requirements.txt
│
├── src/
│   ├── detector.py      ← YOLOv8 / fallback MOG2 detector + IoU tracker
│   ├── classifier.py    ← MediaPipe pose + bbox heuristic shot classifier
│   ├── analytics.py     ← Shot aggregation, JSON/CSV export, summary stats
│   └── visualizer.py    ← Bounding-box overlay, shot flash, HUD, dashboard
│
├── output/              ← Generated outputs (gitignored)
│   ├── shot_predictions.json
│   ├── shot_predictions.csv
│   ├── annotated_output.mp4
│   └── analytics_dashboard.png
│
├── models/              ← Place downloaded YOLO weights here
│   └── (yolov8n.pt, etc.)
│
└── data/
    └── (sample videos)
```

---

## ⚙️ Setup

### 1. Clone / unzip the project

```bash
git clone https://github.com/YOUR_USERNAME/padel-analytics.git
cd padel-analytics
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` auto-downloads `yolov8n.pt` on first run.  
> For MediaPipe: `pip install mediapipe` (optional — system gracefully degrades).

---

## 🚀 Usage

### Run the synthetic demo (no video needed)

```bash
python demo.py --frames 300 --output output/
```

### Run on a real padel video

```bash
python main.py --input data/match.mp4 --output output/ --save-video
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to input video |
| `--output` | `output/` | Output directory |
| `--yolo` | `yolov8n.pt` | YOLO weights file |
| `--conf` | `0.30` | Detection confidence threshold |
| `--no-display` | off | Disable live preview (headless servers) |
| `--save-video` | off | Write annotated video to output dir |
| `--skip-frames` | `1` | Process every N-th frame (speed vs accuracy) |

---

## 🧠 Methodology

### 1. Object Detection & Tracking

**Primary:** YOLOv8n (Ultralytics) — detects COCO classes `person` (→ player), `sports ball` (→ ball), `tennis racket` (→ racket).

**Fallback:** MOG2 background subtraction — when YOLO weights are unavailable, moving blobs are classified by area (small = ball, large = player).

**Tracking:** A lightweight IoU-based tracker assigns persistent IDs across frames without needing deep-sort or a GPU.

### 2. Shot Classification

Shot detection uses a **two-tier system**:

**Tier 1 — MediaPipe Pose (if available):**
- Extracts wrist, elbow, and shoulder landmarks from the player crop
- Computes elbow angle and wrist height relative to shoulder
- Rules:
  - Wrist above shoulder + elbow angle > 120° → **Smash**
  - Dominant wrist on same side as racket → **Forehand**
  - Cross-body wrist position → **Backhand**
  - Low racket + close to net → **Volley**

**Tier 2 — Bounding-box heuristics (fallback):**
- Racket centroid above top 20% of player box → **Smash**
- Racket centroid right of player centre → **Forehand**
- Racket centroid left of player centre → **Backhand**
- Ball above player → **Smash**

A **15-frame cooldown** prevents duplicate shot events from a single swing.

### 3. Analytics & Output

Each detected shot is recorded as:

```json
{
  "frame": 145,
  "timestamp_sec": 4.833,
  "shot_type": "forehand",
  "confidence": 0.75,
  "player_id": 2,
  "ball_center": [318, 241]
}
```

A summary with shot counts per type and per player is prepended to the JSON.

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `shot_predictions.json` | Full shot log + summary statistics |
| `shot_predictions.csv` | Tabular version for spreadsheet analysis |
| `annotated_output.mp4` | Video with bounding boxes, trail, and shot labels |
| `analytics_dashboard.png` | Bar + pie chart of shot distribution |

---

## ⚡ Shot Types Classified

| Shot | Detection Method |
|------|-----------------|
| Forehand | Racket/wrist on dominant side |
| Backhand | Racket/wrist crossing body midline |
| Smash / Serve | Wrist/racket elevated above shoulder |
| Volley | Low racket near net mid-frame |

---

## 🚧 Challenges Faced

1. **Ball detection** — padel balls are small and fast; YOLO `yolov8n` is trained on COCO `sports ball` which works reasonably well but a fine-tuned model would improve accuracy significantly.
2. **Shot boundaries** — distinguishing the exact frame of contact is non-trivial; the cooldown heuristic works but misses rapid back-to-back exchanges.
3. **Player identification** — with 4 players on court, reliably assigning a shot to the correct player requires more robust tracking (e.g., DeepSORT with re-ID embeddings).
4. **Camera angles** — wide-angle overhead footage versus side-on footage requires different heuristics; the current system assumes a broadcast-style side view.

---

## 🔮 Future Improvements

- Fine-tune YOLOv8 on a padel-specific dataset for better ball/racket detection
- Add DeepSORT for robust multi-player re-identification
- Train a temporal CNN / LSTM classifier on labelled shot clips for higher accuracy
- Add ball-bounce detection (velocity inversion in Y-axis)
- Compute shot direction / court zone using homography
- Build a real-time web dashboard with Flask + WebSocket

---

## 📦 Models

Download links for pre-trained weights:

- **YOLOv8n (COCO):** Auto-downloaded by `ultralytics` on first run, or manually from [Ultralytics](https://github.com/ultralytics/ultralytics).
- Place custom fine-tuned weights in the `models/` folder and pass `--yolo models/your_model.pt`.

---

## 👤 Adnan Ansari (Author)

Built as part of the **Layman AI Internship Assignment**.  
Stack: Python · OpenCV · YOLOv8 · MediaPipe · NumPy · Matplotlib

link of Input Data video : https://drive.google.com/file/d/1eoF6gou59wTBGgL54spxaBgN3sUPd3yh/view?usp=sharing

link of Output annotated video : https://drive.google.com/file/d/1bzMj-wUpKuFasNg_77Ir623t9nvjn7pT/view?usp=sharing
