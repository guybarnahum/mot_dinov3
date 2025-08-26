# cli.py
import os
from dotenv import load_dotenv
load_dotenv()  # pick up HUGGINGFACE_HUB_TOKEN from .env

import argparse, sys, time
from time import perf_counter
import numpy as np, cv2, torch
from tqdm.auto import tqdm

from mot_dinov3 import compat
compat.apply(strict_numpy=False, quiet=True)

from mot_dinov3.detector import Detector
from mot_dinov3.embedder import DinoV3Embedder, GatedModelAccessError
from mot_dinov3.tracker import SimpleTracker
from mot_dinov3.viz import draw_tracks


def parse_args():
    ap = argparse.ArgumentParser(description="DINOv3-based MOT (tracking-by-detection)")
    ap.add_argument("--source", type=str, required=True, help="Path to input video")
    ap.add_argument("--output", type=str, default="outputs/tracked.mp4", help="Output video path")
    ap.add_argument("--det", type=str, default="yolov8n.pt", help="Ultralytics model path or name")
    ap.add_argument("--dinov3", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                    help="HF model id (e.g., facebook/dinov3-..., or open: facebook/dinov2-base)")
    ap.add_argument("--fallback-open", action="store_true",
                    help="If gated access is denied, automatically fall back to facebook/dinov2-base")
    ap.add_argument("--conf", type=float, default=0.3, help="Detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=960, help="Detector input size")
    ap.add_argument("--classes", type=str, default="", help="Comma-separated class ids to keep (optional)")
    ap.add_argument("--fps", type=float, default=0.0, help="Force output FPS (0 uses source FPS)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--no-hungarian", action="store_true", help="Use greedy assignment instead of Hungarian")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks for debugging")
    return ap.parse_args()


def build_embedder(model_id: str, device: str, autocast: bool, allow_fallback: bool):
    try:
        return DinoV3Embedder(model_id, device=device, use_autocast=autocast)
    except GatedModelAccessError as e:
        # Print only the friendly message
        print(str(e))
        if allow_fallback:
            fb = "facebook/dinov2-base"
            print(f"\n⚠️  Falling back to open model: {fb}\n")
            return DinoV3Embedder(fb, device=device, use_autocast=autocast)
        sys.exit(2)


def _fmt_ms(s):  # seconds -> "xx.x ms"
    return f"{s * 1000:.1f} ms"


def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), f"Could not open {args.source}"

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or total_frames > 1_000_000_000:
        total_frames = None  # unknown or bogus

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer for {args.output}")

    # Init modules
    detector = Detector(args.det, device=device, imgsz=args.imgsz)
    embedder = build_embedder(args.dinov3, device=device, autocast=(device == "cuda"),
                              allow_fallback=args.fallback_open)
    tracker = SimpleTracker(use_hungarian=not args.no_hungarian)

    keep_ids = None
    if args.classes.strip():
        keep_ids = set(int(x) for x in args.classes.split(",") if x.strip().isdigit())

    # Progress bar
    pbar = tqdm(
        total=total_frames,
        unit="frame",
        dynamic_ncols=True,
        desc=f"Tracking [{device}]",
        disable=not sys.stderr.isatty(),
        leave=True,
    )

    # Timers / stats
    t_start = perf_counter()
    frames = 0
    ema_fps = None
    alpha = 0.15  # smoothing for FPS

    # Stage accumulators (seconds)
    acc_read = acc_det = acc_emb = acc_track = acc_draw = acc_write = 0.0
    acc_frame = 0.0
    frame_times = []  # per-frame total time, for p50/p95

    while True:
        f0 = perf_counter()
        # --- READ ---
        t = perf_counter()
        ok, frame = cap.read()
        t_read = perf_counter() - t
        if not ok:
            break

        # --- DETECT ---
        t = perf_counter()
        boxes, confs, clses = detector.detect(frame, conf_thres=args.conf)
        t_det = perf_counter() - t

        # Optional class filtering
        if keep_ids is not None and len(boxes) > 0:
            m = np.array([c in keep_ids for c in clses], dtype=bool)
            boxes, confs, clses = boxes[m], confs[m], clses[m]

        # --- EMBED ---
        t = perf_counter()
        embs = embedder.embed_crops(frame, boxes)
        t_emb = perf_counter() - t

        # --- TRACK + DRAW ---
        t = perf_counter()
        tracks = tracker.update(boxes, embs)
        t_track = perf_counter() - t

        t = perf_counter()
        draw_tracks(frame, tracks)
        t_draw = perf_counter() - t

        # --- WRITE ---
        t = perf_counter()
        out.write(frame)
        t_write = perf_counter() - t

        # Totals / stats
        f_dt = perf_counter() - f0
        frames += 1
        frame_times.append(f_dt)
        acc_read += t_read
        acc_det += t_det
        acc_emb += t_emb
        acc_track += t_track
        acc_draw += t_draw
        acc_write += t_write
        acc_frame += f_dt

        # Live FPS (EMA) and stage timings in postfix
        inst_fps = 1.0 / max(1e-6, f_dt)
        ema_fps = inst_fps if ema_fps is None else (1 - alpha) * ema_fps + alpha * inst_fps
        if frames % 5 == 0:
            pbar.set_postfix({
                "fps": f"{ema_fps:.1f}",
                "det": _fmt_ms(t_det),
                "emb": _fmt_ms(t_emb),
                "trk": _fmt_ms(t_track),
                "draw": _fmt_ms(t_draw),
            })

        pbar.update(1)

    # Cleanup
    cap.release()
    out.release()
    pbar.close()

    elapsed = perf_counter() - t_start
    mean_ms = 1000.0 * acc_frame / max(1, frames)
    eff_fps = frames / max(1e-9, elapsed)

    # Percent breakdown by stage
    def pct(x): return 100.0 * x / max(1e-9, acc_frame)
    # p50/p95 frame latency (ms)
    if frame_times:
        arr = np.array(frame_times, dtype=np.float64) * 1000.0
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
    else:
        p50 = p95 = 0.0

    print("\n=== Tracking summary ===")
    print(f"Video:          {args.source}  →  {args.output}")
    print(f"Resolution:     {W}x{H} @ {src_fps:.2f} fps (out {fps:.2f} fps)")
    print(f"Device:         {device}")
    print(f"Frames:         {frames}  |  Wall time: {elapsed:.2f}s  |  Effective FPS: {eff_fps:.1f}")
    print(f"Latency/frame:  mean {mean_ms:.1f} ms  |  p50 {p50:.1f} ms  |  p95 {p95:.1f} ms")
    print("Stage breakdown (mean per frame | share of total):")
    print(f"  read : {_fmt_ms(acc_read / max(1, frames))}  | {pct(acc_read):5.1f}%")
    print(f"  detect: {_fmt_ms(acc_det / max(1, frames))}  | {pct(acc_det):5.1f}%")
    print(f"  embed : {_fmt_ms(acc_emb / max(1, frames))}  | {pct(acc_emb):5.1f}%")
    print(f"  track : {_fmt_ms(acc_track / max(1, frames))} | {pct(acc_track):5.1f}%")
    print(f"  draw  : {_fmt_ms(acc_draw / max(1, frames))}  | {pct(acc_draw):5.1f}%")
    print(f"  write : {_fmt_ms(acc_write / max(1, frames))} | {pct(acc_write):5.1f}%")
    other = max(0.0, acc_frame - (acc_read + acc_det + acc_emb + acc_track + acc_draw + acc_write))
    print(f"  other : {_fmt_ms(other / max(1, frames))}  | {pct(other):5.1f}%")
    print(f"\nDone. Wrote {args.output}.")


if __name__ == "__main__":
    try:
        main()
    except GatedModelAccessError as e:
        # Belt & suspenders: catch here too, print clean, no traceback
        print(str(e))
        sys.exit(2)
    except Exception as e:
        # No ugly tracebacks unless explicitly requested
        if "--debug" in sys.argv or os.getenv("MOT_DEBUG") == "1":
            raise
        print(f"Error: {e}")
        sys.exit(1)
