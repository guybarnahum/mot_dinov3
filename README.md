# mot-dinov3 / Multi-Object Tracker DINOv3

DINOv3/DINOv2 embeddings + YOLO detection + simple MOT (Hungarian) with cosine appearance matching.  
Progress bar + per-stage timing stats built-in.

## ✨ Features

- **Detector:** Ultralytics YOLO (configurable)
- **Embedder:** DINOv3 (gated) or DINOv2 (open), via 🤗 Transformers
- **Tracker:** IoU + cosine(DINO) cost, Hungarian assignment
- **CLI:** clean errors for gated models, optional fallback to DINOv2
- **Progress & Profiling:** `tqdm` bar, FPS, mean/p50/p95 latency, stage breakdown
- **Setup script:** CPU / T4 GPU variants, optional install of latest Transformers for DINOv3

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

### 1) CPU (PyTorch 2.2.2)

Recommended for Macs or CPUs:

```bash
bash setup.sh cpu --dinov3-edge --yes
# --dinov3-edge installs Transformers from GitHub (latest, supports DINOv3)
# use --dinov3-stable for PyPI release if it already supports DINOv3
```

### 2) T4 GPU (PyTorch 2.4.2 CUDA)

```bash
TORCH_CHANNEL=cu124 bash setup.sh t4_gpu --dinov3-edge --yes
# override TORCH_CHANNEL if needed: cu121 / cu122 / cu124
```

> The setup script:
>
> * pins `numpy<2` and `scipy<1.13` (PyTorch 2.2 CPU ABI-friendly)
> * installs extras from `pyproject.toml` (`[cpu]` or `[t4_gpu]`)
> * upgrades Transformers to support DINOv3 (stable or edge)
> * verifies DINOv3 support; auto-offers fallback to edge if stable is too old

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

If you don’t have DINOv3 access or support yet, you can either:

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
--debug                      show full tracebacks
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
│     ├─ embedder.py     # DINOv3/DINOv2 embeddings (gated handling + manual preproc fallback)
│     ├─ detector.py     # Ultralytics detection
│     ├─ tracker.py      # IoU + cosine(DINO) + Hungarian
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

* Ensure your `.env` has `HF_TOKEN=...` (we also bridge legacy names).
* Ensure the same HF account has accepted access on the model page.

**“Transformers does not recognize `dinov3_vit`”**

* Install latest Transformers:

  ```bash
  pip install -U transformers huggingface_hub
  # or bleeding-edge:
  pip install -U "git+https://github.com/huggingface/transformers.git"
  ```
* Or run with `--fallback-open` / `--dinov3 facebook/dinov2-base`.

**NumPy 2.x ABI errors (Torch 2.2.x CPU):**

* The setup script already pins these; if you changed deps:

  ```bash
  pip install "numpy<2" "scipy<1.13"
  ```

**Mac/CPU “torch.compiler.is\_compiling” error:**

* We ship a small shim in `compat.py` to satisfy newer Transformers; no action needed.

---

## 📜 License

This repository includes references to third-party models and datasets.
Please follow their respective licenses and gating terms (e.g., Meta’s DINOv3).

---

## 🙏 Acknowledgements

* Meta AI: DINOv3 / DINOv2
* Ultralytics: YOLO
* Hugging Face: Transformers & Hub

```

