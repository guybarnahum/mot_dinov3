# Multi-Object Tracker DINOv3

DINOv3/DINOv2 embeddings + YOLO detection + light MOT with cosine appearance matching.
Progress bar + per-stage timing stats built-in. **Embedding scheduler** reduces compute by reusing features and honoring a **per-frame time budget**.

## ✨ Features

* **Detector:** Ultralytics YOLO (configurable)
* **Embedder:** DINOv3 (gated) or DINOv2 (open), via 🤗 Transformers (or your own via `features/`)
* **Tracker:** IoU + cosine(DINO) cost, Hungarian or greedy assignment
* **Embedding Scheduler (smart & budget-aware):**

  * Reuses cached track embeddings for **stable, non-crowded** objects
  * Computes **critical** embeddings immediately (new/weak-IoU/crowded)
  * Computes **refresh** embeddings **only within a per-frame budget**
  * Maintains a **backlog** of refreshes and serves oldest first
* **UI cues:** Tracks whose embedding was computed **this frame** are labeled with a trailing `*`
* **CLI:** clean errors for gated models, optional fallback to DINOv2
* **Progress & Profiling:** `tqdm` bar, FPS, mean/p50/p95 latency, stage breakdown
* **Setup script:** CPU / T4 GPU variants, optional install of latest Transformers for DINOv3

---

## 🔄 How embedding compute adapts to detection load

The embedding step automatically **scales with scene complexity**:

* **Fewer detections ⇒ fewer candidates.**
  Only current detections are considered for embeddings. With fewer boxes, there are naturally fewer critical needs and fewer refresh candidates, so the embedding time drops.

* **Critical vs. Refresh (scheduler behavior).**

  * **Critical** = must compute **now** (ignores budget): new objects, weak IoU continuity, or crowded overlaps (see `--embed-iou-gate`, `--embed-overlap-thr`).
  * **Refresh** = nice-to-have, done **within `--embed-budget-ms`** to keep features fresh (default every `--embed-refresh` frames). When the budget is tight, refreshes defer to a **backlog** (oldest first).

* **Backlog drains in easy frames.**
  If previous frames deferred refreshes, the scheduler uses “extra” time in later, lighter frames (fewer detections) to **catch up**—still honoring the per-frame budget.

* **Strict budgeting for refresh.**
  Refresh work is admitted **incrementally** (one-by-one or tiny chunks) against the **live** per-crop EMA. CUDA is synchronized around timing so the step won’t overshoot `--embed-budget-ms`.

* **What you’ll see in practice.**

  * Sparse scenes: many detections **reuse** cached embeddings → `emb` time hovers near **0–few ms**; occasional `*` marks when a refresh fits the budget.
  * Busy scenes: more **critical** work → `emb` can exceed the budget (by design) to keep IDs stable, while refreshes are throttled.

**Tuning knobs**

* Budget: `--embed-budget-ms 20` (caps **refresh** work per frame; 0 = unlimited)
* Refresh rate: `--embed-refresh 5` (lower = more frequent refreshes)
* Reuse strictness: `--embed-iou-gate 0.6` (higher = reuse only with stronger IoU continuity)
* Crowding sensitivity: `--embed-overlap-thr 0.2` (lower = more scenes treated as crowded ⇒ more critical embeds)

> Tip: Tracks that had a **real embedding computed this frame** are labeled `ID <tid> *` in the video, so you can visually verify when compute happens.

---
## 📦 Install

Clone the repo, then choose one of the variants below.

### 0) (Optional) Hugging Face token

Create a `.env` in the repo root:

```bash
echo 'HF_TOKEN=hf_xxx_your_access_token' > .env
# Legacy names also supported: HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN
```

> You must also **request/accept access** on the DINOv3 model page (gated) with the same HF account tied to your token.

### 1) CPU (PyTorch 2.2.2)

Recommended for Macs or CPUs:

```bash
bash setup.sh cpu --dinov3-edge --yes
# --dinov3-edge installs Transformers from GitHub (latest, supports DINOv3)
# use --dinov3-stable for PyPI release if it already supports DINOv3
```

### 2) T4/L4 GPU (PyTorch 2.4.x CUDA)

```bash
TORCH_CHANNEL=cu124 bash setup.sh t4_gpu --dinov3-edge --yes
# override TORCH_CHANNEL if needed: cu121 / cu122 / cu124
```

> The setup script pins `numpy<2` and `scipy<1.13`, installs extras from `pyproject.toml`, and can upgrade Transformers to a DINOv3-capable version.

---

## ▶️ Run

Basic run (uses DINOv3 if available):

```bash
source .venv/bin/activate
python cli.py \
  --source data/input.mp4 \
  --output outputs/tracked.mp4 \
  --dinov3 facebook/dinov3-vitb16-pretrain-lvd1689m
```

If you don’t have DINOv3 access or support yet:

```bash
# auto-fallback to open DINOv2
python cli.py --source data/input.mp4 --output outputs/tracked.mp4 --fallback-open

# explicitly use DINOv2
python cli.py --source data/input.mp4 --output outputs/tracked.mp4 --dinov3 facebook/dinov2-base
```

Filter classes (e.g., COCO `0` = person) and change detector size:

```bash
python cli.py --source data/input.mp4 --output outputs/tracked.mp4 --classes "0" --imgsz 1280 --conf 0.25
```

### Embedding Scheduler & Budget examples

```bash
# Refresh embedding every 5 frames (default), but cap refresh work at 20 ms per frame.
python cli.py --source data/input.mp4 --output outputs/tracked.mp4 \
  --embed-refresh 5 --embed-budget-ms 20

# More aggressive reuse: require high IoU continuity to reuse (0.7), treat >0.15 IoU between detections as crowded:
python cli.py --source data/input.mp4 --output outputs/tracked.mp4 \
  --embed-iou-gate 0.7 --embed-overlap-thr 0.15 --embed-refresh 8 --embed-budget-ms 15
```

**Visual cue:** any track that had its **embedding computed** this frame is annotated as `ID <tid> *`. (No star = reused.)

---

## 🧠 How Embeddings Are Scheduled (Behavior & Insights)

The scheduler separates work into **critical** vs **refresh**:

* **Critical = compute now (ignores budget)**
  We always compute embeddings when:

  * The detection has **weak IoU continuity** with any active track (possible new or identity risk).
  * The detection is in a **crowded** region (det-det IoU above `--embed-overlap-thr`), where appearance helps disambiguate.
  * There are **no active tracks** yet (bootstrapping).

* **Refresh = nice-to-have (budget-limited)**
  For **stable** detections (good IoU continuity and not crowded), we:

  * **Reuse** last cached/EMA embedding by default.
  * **Periodically refresh** the embedding every `--embed-refresh N` frames to correct drift.
  * Refreshes are **deferred** when the per-frame budget would be exceeded.
  * Deferred refreshes go to a **backlog** and are served **oldest-first** when time allows.

* **Budgeting (`--embed-budget-ms`)**

  * Applies **only** to **refresh** work; **critical** work always runs.
  * Uses a smoothed estimate (EMA) of **ms per crop** to decide how many refreshes fit in the remaining time this frame.
  * Prevents refresh storms and stabilizes FPS.

* **Backlog management**

  * The scheduler maintains a small FIFO of **track IDs that are due** for refresh.
  * Each frame, it prioritizes backlog (stale first) within the remaining budget.
  * When a refresh is done, the track leaves the backlog; otherwise it remains for the next frame.

* **Label `*`**

  * When a track’s embedding is (re)computed this frame (critical or refresh), its label shows a trailing `*`.
  * This gives you an at-a-glance sense of how often embeddings are computed vs reused.

---

## 🧪 Experimentation & Re-ID Options

* **Embedding backends:** via `mot_dinov3/features/factory.py` you can plug in

  * `dino` (default: DINOv3/DINOv2 via Transformers, config-based manual preprocess fallback)
  * others, e.g., CLIP/ResNet or your custom extractor
* **Crop policy:** `--crop-pad` and `--crop-square` to tune context and shape
* **Tracker knobs:** switch to greedy (`--no-hungarian`) to test cost behavior; tune IoU / appearance weights in `tracker.py`
* **Longer-term re-ID:** the scheduler reduces compute cost while keeping periodic refreshes; for scenes with long occlusions, consider:

  * extending `max_misses` / keeping a small gallery per track,
  * stronger motion model (Kalman) or gate by motion during reuse,
  * optional memory of **multiple** historical embeddings per track with decay.

---

## 🧰 CLI Options

```
python cli.py --help

--source PATH                (required) input video
--output PATH                output video (default: outputs/tracked.mp4)
--det MODEL_OR_PATH          Ultralytics model (default: yolov8n.pt)
--dinov3 MODEL_ID            HF model id (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
--fallback-open              if gated/unsupported, auto-switch to facebook/dinov2-base
--conf FLOAT                 detector confidence (default: 0.3)
--imgsz INT                  detector input size (default: 960)
--classes "CSV"              keep only these class ids (e.g., "0" for person)
--fps FLOAT                  override output FPS (default: source FPS)
--cpu                        force CPU
--no-hungarian               use greedy assignment (debug)

--embedder STR               embedding backend: dino (default), clip, resnet, ...
--embed-model STR            embedding model id/name for the chosen embedder
--crop-pad FLOAT             padding ratio around det for crops (default: 0.12)
--crop-square                square crops for embeddings (default: on)

--embed-refresh INT          refresh real embedding every N frames (default: 5)
--embed-iou-gate FLOAT       IoU ≥ gate → eligible to reuse (default: 0.6)
--embed-overlap-thr FLOAT    det-det IoU > thr marks crowded (default: 0.2)
--embed-budget-ms FLOAT      per-frame ms cap for refresh work (0 = unlimited)

--draw-lost                  also draw LOST tracks
--line-thickness INT         bbox line thickness
--font-scale FLOAT           label font scale

--debug                      show full tracebacks
```

At the end of a run you’ll see a summary:

* effective FPS
* mean / p50 / p95 per-frame latency (ms)
* mean per-stage times + %: read / detect / embed / track / draw / write
* **embedding stats**: total computed, per-frame rate, reused count

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
│     ├─ embedder.py     # DINOv3/DINOv2 embeddings (gated handling + manual preproc fallback)
│     ├─ detector.py     # Ultralytics detection
│     ├─ features/       # pluggable embedding backends (factory.py, dino.py, clip.py, ...)
│     ├─ scheduler.py    # embedding reuse/refresh + backlog + budget
│     ├─ tracker.py      # IoU + cosine(DINO) + Hungarian (EMA embeddings)
│     ├─ utils.py
│     └─ viz.py          # colored boxes, labels; ‘*’ when embedding was computed this frame
├─ data/                 # put your videos here (ignored; keep .gitkeep)
└─ outputs/              # results (ignored; keep .gitkeep)
```

---

## 🧩 Troubleshooting

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

* Ensure `.env` has `HF_TOKEN=...` (we also bridge legacy token env vars)
* Ensure the same HF account has accepted access on the model page

**“Transformers does not recognize `dinov3_vit`”**

```bash
pip install -U transformers huggingface_hub
# or bleeding-edge:
pip install -U "git+https://github.com/huggingface/transformers.git"
```

Or run with `--fallback-open` / `--dinov3 facebook/dinov2-base`.

**NumPy 2.x ABI errors (Torch 2.2.x CPU)**

```bash
pip install "numpy<2" "scipy<1.13"
```

**Mac/CPU “torch.compiler.is\_compiling”**

* Covered by the shim in `compat.py`; no action needed.

---

## 📜 License

This repository includes references to third-party models and datasets.
Please follow their respective licenses and gating terms (e.g., Meta’s DINOv3).

---

## 🙏 Acknowledgements

* Meta AI: DINOv3 / DINOv2
* Ultralytics: YOLO
* Hugging Face: Transformers & Hub
