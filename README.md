# 🦖 MOT-DINOv3

### Multi-Object Tracking with a Gated Appearance Model and Motion Prediction

This project's goal is to perform multi-object tracking by combining object detection with appearance embeddings. It focuses on maintaining stable track identities through common challenges like partial occlusion. This is addressed with two primary mechanisms: a Kalman filter to predict object motion and a dual-model appearance gallery to store and reference multiple viewpoints of a tracked object.

To manage computational load, a budget-aware scheduler selectively computes new appearance embeddings based on track stability and a configurable time budget.

-----

## ✨ Features

- **Detector**: Ultralytics YOLO (configurable via command line and config.toml).
- **Embedder**: DINOv3 (gated) or DINOv2 (open) via 🤗 Transformers.
- **Tracker**: A tracker designed for persistence through occlusions, featuring:
  - **Kalman Filter Motion Model**: Uses a linear Kalman filter to predict track locations and smooth velocity estimates, improving matching with predicted boxes.
  - **Gated Appearance Gallery**: Maintains a long-term gallery of high-quality, diverse embeddings for each track. This provides a robust reference for re-identification when a track's most recent appearance is corrupted by partial occlusion.
  - **Velocity-Informed Gating**: Uses the track's last known velocity to define a dynamic search area for re-identification, making the search more efficient.

- **Embedding Scheduler**: Selectively computes embeddings based on track stability, crowdedness, and a configurable time budget to reduce computational load. A force_compute_all flag is available to disable scheduling for baseline testing.
- **Diagnostic Visualization**: An optional debug mode that generates a video with detailed diagnostic panels for analyzing tracker state and performance frame-by-frame.

-----

## 🚀 Quickstart

Get up and running in a few simple steps.

#### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/mot-dinov3.git
cd mot-dinov3
```

#### 2\. (Optional) Set Hugging Face Token for Gated Models

To use gated models like DINOv3, create a `.env` file with your token.

```bash
echo 'HF_TOKEN=hf_xxx_your_access_token' > .env
```

> **Note:** You must also visit the [DINOv3 model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) and accept the terms to gain access.

#### 3\. Run the Setup Script

For a typical CUDA-enabled environment (e.g., NVIDIA T4/L4 GPU):

```bash
# This script creates a .venv/ and installs dependencies
bash setup.sh t4_gpu --yes
```

For CPU or Mac:

```bash
bash setup.sh cpu --yes
```

#### 4\. Run the Demo

Activate the virtual environment and run the tracker on a sample video.

```bash
source .venv/bin/activate

python cli.py \
  --source data/input.mp4 \
  --output outputs/tracked_video.mp4 \
  --det yolov8n.pt \
  --embed-model facebook/dinov2-base
```

-----

## 🧠 The Smart Scheduler

The scheduler is the core component for achieving high performance. It categorizes every detection to decide if an expensive embedding calculation is truly necessary.

The decision process for each detected object is as follows:

```
Detection
    └--> Is it a stable track (high IoU match & not crowded)?
          |
          ├─--> YES: Is the track's embedding stale (due for a refresh)?
          |     |
          |     ├─-> YES: Add to REFRESH backlog (processed if budget allows).
          |     └--> NO: REUSE the cached embedding (almost zero cost).
          |
          └--> NO: This is a CRITICAL detection (new object, crowded scene, etc.).
                    Compute the embedding immediately (ignores budget).
```

This ensures that compute resources are spent where they matter most: establishing new tracks and resolving ambiguity in crowded scenes.

#### Tuning the Scheduler

You can control the scheduler's behavior with a few key parameters.

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--embed-budget-ms` | `0.0` | Caps the time spent on **refresh** work per frame. `0` means unlimited. A value like `20` ensures refreshes don't cause frame drops. |
| `--embed-refresh` | `5` | How often to refresh a stable track's embedding (every N frames). Lower is more frequent. |
| `--embed-iou-gate` | `0.6` | An object must have an IoU of at least this value with a prior track to be considered "stable" for reuse. |
| `--embed-overlap-thr`| `0.2` | If two detections have an IoU greater than this, they are marked as "crowded," triggering a critical compute. |

> **Visual Cue:** In the output video, any track ID with a trailing `*` (e.g., `ID 42*`) indicates its embedding was re-computed in that frame. No star means it was reused.

-----

## 🧰 Command-Line Options

Below is a summary of the most important CLI arguments. For a full list, run `python cli.py --help`.

| Argument | Default | Description |
| :--- | :--- | :--- |
| **I/O** | | |
| `--source` | **(required)** | Path to the input video file. |
| `--output` | `outputs/tracked.mp4` | Path to save the output video. |
| **Detector** | | |
| `--det` | `yolov8n.pt` | Ultralytics YOLO model to use for detection. |
| `--conf` | `0.3` | Minimum confidence threshold for detections. |
| `--imgsz` | `960` | Input image size for the detector. |
| `--classes` | `""` | Comma-separated list of class IDs to track (e.g., `"0,2,5"`). |
| **Embedding Scheduler** | | |
| `--embed-model` | `facebook/dinov3...` | Hugging Face model ID for the embedder. |
| `--dinov3` | *(deprecated)* | Legacy argument for `--embed-model`. |
| `--fallback-open` | `false` | If a gated model fails, automatically try an open model (`dinov2-base`). |
| `--embed-budget-ms` | `0.0` | Max milliseconds per frame for **refresh** work. `0` = unlimited. |
| **Tracker** | | |
| `--reid-sim-thr` | `0.6` | Appearance similarity threshold needed to revive a `LOST` track. |
| `--max-age` | `30` | Number of frames a track can be unmatched before being marked `LOST`. |
| `--reid-max-age` | `60` | Number of frames a `LOST` track is kept for re-identification before being deleted. |
| `--no-hungarian`| `false` | Use faster greedy assignment instead of the optimal Hungarian algorithm. |
| **System** | | |
| `--cpu` | `false` | Force all computations to run on the CPU. |
| `--debug` | `false` | Show full stack traces on error. |

-----

## 🗂️ Project Structure

```
.
├─ README.md
├─ pyproject.toml
├─ setup.sh
├─ clean.sh
├─ .gitignore
├─ .env.example          # .env not committed
├─ cli.py
├─ config.toml
├── src
│   └── mot_dinov3
│      ├── __init__.py
│      ├── compat.py      # small shims (torch.compiler, numpy guard, HF token bridge)
│      ├── detector.py    # Ultralytics detection
│       ├── embedders
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── dino.py
│       │   ├── osnet.py
│       │   └── transreid.py
│       ├── features      # pluggable embedding backends (factory.py, dino.py, clip.py, ...)
│       │   ├── base.py
│       │   ├── dino.py
│       │   ├── factory.py
│       │   ├── osnet.py
│       │   └── transreid.py
│       ├── scheduler.py  # embedding reuse/refresh + backlog + budget
│       ├── tracker.py    # IoU + cosine(DINO) + Hungarian (EMA embeddings)
│       ├── utils.py
│       └── viz.py        # colored boxes, labels; ‘*’ when embedding was computed this frame         
├─ input/                # input videos here (ignored; keep .gitkeep)
└─ outputs/              # results (ignored; keep .gitkeep)
```

---

## 🧩 Troubleshooting
<details>
<summary><h3>Gated model message (clean, no traceback)</h3></summary>

```
Model 'facebook/dinov3-vitb16-pretrain-lvd1689m' is gated on Hugging Face.
To use it:
  1) Visit the model page while logged in and click “Agree/Request access”.
  2) Provide an access token (recommended via .env):
       HF_TOKEN=hf_xxx
     Or log in once: `huggingface-cli login`.
  3) Or choose an open fallback, e.g.: --dinov3 facebook/dinov2-base
```

* Ensure `.env` has `HF_TOKEN=...` (we also bridge legacy token env vars)
* Ensure the same HF account has accepted access on the model page
</details>
<details>
<summary><h3>“Transformers does not recognize `dinov3_vit`”</h3></summary>

```bash
pip install -U transformers huggingface_hub
# or bleeding-edge:
pip install -U "git+https://github.com/huggingface/transformers.git"
```

Or run with `--fallback-open` / `--dinov3 facebook/dinov2-base`.
</details>
<details>
<summary><h3>NumPy 2.x ABI errors (Torch 2.2.x CPU)</h3></summary>

```bash
pip install "numpy<2" "scipy<1.13"
```
</details>

---

## 📜 License

This repository includes references to third-party models and datasets.
Please follow their respective licenses and gating terms (e.g., Meta’s DINOv3).

---

## 🙏 Acknowledgements

* Meta AI: DINOv3 / DINOv2
* Ultralytics: YOLO
* Hugging Face: Transformers & Hub
