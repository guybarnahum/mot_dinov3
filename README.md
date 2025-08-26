# Multi-Object Tracker · DINOv3/DINOv2

DINOv3/DINOv2 embeddings + YOLO detection + **re-identification (re-ID) across occlusions/out-of-frame**.  
Progress bar + per-stage timing stats built-in.

## ✨ What’s new

- **Longer-term re-ID:** tracks survive occlusions/out-of-frame gaps via an ACTIVE → LOST → REMOVED lifecycle.
- **Two-stage association:** ACTIVE↔detections matched twice (high-conf, then low-conf) with **IoU + appearance**.
- **Re-ID stage:** LOST↔detections matched with **appearance-only** (cosine) + **loose center gate** that grows with time.
- **Stable appearance:** per-track **EMA** embedding + a small **gallery** (last few embeddings) for robust matching.
- **Embedding experimentation:** try different DINO variants, crop padding, and square crops to improve invariance.
- **Tunable tracker:** weights, thresholds, ages and gates exposed as flags for quick iteration.
- **Clean gated-model UX:** helpful message if DINOv3 access isn’t granted; optional auto-fallback to DINOv2.

---

## 📦 Install

Clone the repo, then choose one of the variants below.

### 0) (Optional) Hugging Face token

Create a `.env` in the repo root:

```bash
echo 'HF_TOKEN=hf_xxx_your_access_token' > .env
# Legacy names also supported: HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN
````

> You must also **request/accept access** on the DINOv3 model page (gated) with the same HF account tied to your token.

### 1) CPU

Recommended for Macs or CPU-only Linux:

```bash
bash setup.sh cpu --dinov3-edge --yes
# --dinov3-edge installs Transformers from GitHub (latest, supports DINOv3)
# use --dinov3-stable for PyPI release if it already supports DINOv3
```

### 2) T4/L4 GPU (CUDA)

```bash
TORCH_CHANNEL=cu124 bash setup.sh t4_gpu --dinov3-edge --yes
# override TORCH_CHANNEL if needed: cu121 / cu122 / cu124
```

> The setup script:
>
> * pins `numpy<2` and `scipy<1.13` (PyTorch 2.2 CPU ABI-friendly)
> * installs extras from `pyproject.toml` (`[cpu]` or `[t4_gpu]`)
> * upgrades/validates Transformers for DINOv3 (stable or edge, with auto-fallback to edge if needed)

---

## ▶️ Quick start

Basic run (DINOv3 if available):

```bash
source .venv/bin/activate
python cli.py \
  --source data/input.mp4 \
  --output outputs/tracked.mp4 \
  --dinov3 facebook/dinov3-vitb16-pretrain-lvd1689m
```

If you don’t have DINOv3 access or support yet, either:

* **Auto-fallback to open DINOv2:**

  ```bash
  python cli.py --source data/input.mp4 --output outputs/tracked.mp4 --fallback-open
  ```
* **Explicitly use DINOv2:**

  ```bash
  python cli.py --source data/input.mp4 --output outputs/tracked.mp4 --dinov3 facebook/dinov2-base
  ```

Filter classes (e.g., COCO `0` = person) and set detector size:

```bash
python cli.py \
  --source data/input.mp4 \
  --output outputs/tracked.mp4 \
  --classes "0" \
  --imgsz 1280 \
  --conf 0.25
```

---

## 🧪 Embedding experimentation (for better re-ID)

* **Switch DINO variants** (stronger backbones or v2 vs v3):

  ```bash
  # DINOv2 base (open)
  python cli.py ... --dinov3 facebook/dinov2-base
  # DINOv3 ViT-B/16 (gated)
  python cli.py ... --dinov3 facebook/dinov3-vitb16-pretrain-lvd1689m
  ```
* **Crop padding / square crops** (more invariance to small shifts/aspect):

  ```bash
  # Increase context around boxes (default ~0.12). Try 0.2 for small objects:
  python cli.py ... --crop-pad 0.20

  # Square crops (on by default in code; pass --no-crop-square if you prefer tight boxes)
  # (If your CLI exposes --crop-square as a flag, enable/disable to compare.)
  ```
* **Class-consistent association** (avoid cross-class switches):

  ```bash
  python cli.py ... --class-consistent
  ```

> Tip: For crowded scenes, try **higher crop padding** and increase `--reid-sim` (0.65–0.75).

---

## ⚙️ Tuning the tracker

These flags let you balance **continuity** vs **ID stability**:

* **Association weights**
  `--iou-w 0.3` and `--app-w 0.7` (default) — bias toward appearance to reduce switches.
* **Two-stage IoU gates**
  `--iou-thresh 0.3` (high-conf pass) and `--iou-thresh-low 0.2` (low-conf pass).
* **Detector confidence bands**
  `--conf-high 0.5` and `--conf-low 0.1` — ByteTrack-style ordering.
* **Re-ID similarity**
  `--reid-sim 0.6` — cosine dot needed to revive a LOST track.
* **Track ages**
  `--max-age 30` frames before ACTIVE becomes LOST;
  `--reid-max-age 60` frames to keep LOST before removal.
* **Re-ID center gate**
  `--center-gate-base 50 --center-gate-slope 10` (pixels) — allowable center jump grows each missed frame.
* **Embedding stability**
  `--ema-alpha 0.9 --gallery-size 10` — EMA + small gallery per track.

**Example (people only, stronger re-ID):**

```bash
python cli.py \
  --source data/pedestrians.mp4 \
  --output outputs/tracked.mp4 \
  --classes "0" \
  --conf 0.25 --imgsz 1280 \
  --iou-w 0.25 --app-w 0.75 \
  --conf-high 0.55 --conf-low 0.10 \
  --reid-sim 0.68 --max-age 20 --reid-max-age 90 \
  --center-gate-base 60 --center-gate-slope 15 \
  --crop-pad 0.20 --class-consistent
```

---

## 🧰 CLI Options

```
python cli.py --help

--source PATH                 (required) input video
--output PATH                 output video (default: outputs/tracked.mp4)
--det MODEL_OR_PATH           Ultralytics model (default: yolov8n.pt)

# Embeddings (DINO)
--dinov3 MODEL_ID             HF model id (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
--fallback-open               if gated/unsupported, auto-switch to facebook/dinov2-base
--crop-pad FLOAT              padding ratio around box for embedding crops (default: 0.12)
--crop-square / --no-crop-square  use square crops for embeddings (on by default)

# Detector
--conf FLOAT                  detector confidence (default: 0.3)
--imgsz INT                   detector input size (default: 960)
--classes "CSV"               keep only these class ids (e.g., "0" for person)
--fps FLOAT                   override output FPS (default: source FPS)
--cpu                         force CPU
--no-hungarian                use greedy assignment (debug)

# Tracker / re-ID
--iou-w FLOAT                 weight of IoU in cost (default: 0.3)
--app-w FLOAT                 weight of appearance in cost (default: 0.7)
--iou-thresh FLOAT            IoU gate for high-conf pass (default: 0.3)
--iou-thresh-low FLOAT        IoU gate for low-conf pass (default: 0.2)
--conf-high FLOAT             high-confidence threshold (default: 0.5)
--conf-low FLOAT              low-confidence lower bound (default: 0.1)
--reid-sim FLOAT              min cosine dot to re-identify (default: 0.6)
--max-age INT                 frames to keep ACTIVE without match (default: 30)
--reid-max-age INT            frames to keep LOST for re-ID (default: 60)
--center-gate-base FLOAT      base pixels allowed for re-ID center shift (default: 50)
--center-gate-slope FLOAT     extra pixels per missed frame (default: 10)
--ema-alpha FLOAT             EMA factor for per-track embedding (default: 0.9)
--gallery-size INT            embeddings kept per track for re-ID (default: 10)
--class-consistent            enforce class-consistent matching

--debug                       show full tracebacks
```

At the end of a run you’ll see a summary:

* effective FPS
* mean / p50 / p95 per-frame latency (ms)
* mean per-stage times + %: read / detect / embed / track / draw / write

---

## 🗂️ Project Structure

```
.
├─ pyproject.toml
├─ setup.sh
├─ clean.sh
├─ .gitignore
├─ .env                  # not committed (token/env)
├─ cli.py
├─ src/
│  └─ mot_dinov3/
│     ├─ __init__.py
│     ├─ compat.py       # small shims (torch.compiler, numpy guard, HF token bridge)
│     ├─ embedder.py     # DINOv3/DINOv2 embeddings (+ manual preproc fallback)
│     ├─ detector.py     # Ultralytics detection
│     ├─ tracker.py      # Two-stage assoc + re-ID + EMA/gallery + class-consistency
│     ├─ utils.py
│     └─ viz.py
├─ data/                 # put your videos here (ignored; keep .gitkeep)
└─ outputs/              # results (ignored; keep .gitkeep)
```

---

## 🧪 Troubleshooting

**Gated model message (clean, no traceback):**

```
Model 'facebook/dinov3-vitb16-pretrain-lvd1689m' is gated on Hugging Face.
To use it:
  1) Visit the model page while logged in and click “Agree/Request access”.
  2) Provide an access token (recommended via .env):
       HF_TOKEN=hf_xxx
     Or log in once: `huggingface-cli login`.
  3) Or choose an open fallback, e.g.: --dinov3 facebook/dinov2-base
```

* Ensure your `.env` has `HF_TOKEN=...` (legacy names bridged).
* Ensure the same HF account has accepted model access.

**“Transformers does not recognize \`dinov3\_vit\`”**

```bash
pip install -U transformers huggingface_hub
# or bleeding-edge (recommended if stable fails verification):
pip install -U "git+https://github.com/huggingface/transformers.git"
```

Or run with `--fallback-open` / `--dinov3 facebook/dinov2-base`.

**NumPy 2.x ABI errors (Torch 2.2 CPU):**

```bash
pip install "numpy<2" "scipy<1.13"
```

---

## 📜 License

This repository includes references to third-party models and datasets.
Please follow their respective licenses and gating terms (e.g., Meta’s DINOv3).

---

## 🙏 Acknowledgements

* Meta AI: DINOv3 / DINOv2
* Ultralytics: YOLO
* Hugging Face: Transformers & Hub


