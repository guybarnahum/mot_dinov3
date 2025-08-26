# cli.py
import os
from dotenv import load_dotenv
load_dotenv()  # pick up HF_TOKEN / HUGGINGFACE_HUB_TOKEN from .env

import argparse, sys
from time import perf_counter
import numpy as np, cv2, torch
from tqdm.auto import tqdm

from mot_dinov3 import compat
compat.apply(strict_numpy=False, quiet=True)

from mot_dinov3.detector import Detector
from mot_dinov3.embedder import GatedModelAccessError
from mot_dinov3.features.factory import create_extractor
from mot_dinov3.tracker import SimpleTracker
from mot_dinov3.scheduler import EmbeddingScheduler, SchedulerConfig
from mot_dinov3.viz import draw_tracks


def parse_args():
    ap = argparse.ArgumentParser(description="DINOv3-based MOT (tracking-by-detection)")
    ap.add_argument("--source", type=str, required=True, help="Path to input video")
    ap.add_argument("--output", type=str, default="outputs/tracked.mp4", help="Output video path")
    ap.add_argument("--det", type=str, default="yolov8n.pt", help="Ultralytics model path or name")

    # legacy / retained for convenience (you'll mostly use --embed-model now)
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

    # tracker robustness knobs
    ap.add_argument("--class-penalty", type=float, default=0.15, help="Additive cost if det class != track stable class")
    ap.add_argument("--conf-high", type=float, default=0.5, help="High-confidence threshold for Stage-1")
    ap.add_argument("--conf-low", type=float, default=0.1, help="Low-confidence threshold for Stage-1b")
    ap.add_argument("--conf-min-update", type=float, default=0.3, help="Only update emb/class when det_conf >= this")
    ap.add_argument("--conf-update-weight", type=float, default=0.5, help="EMA strength scaling with det_conf (0..1)")
    ap.add_argument("--low-conf-iou-only", action="store_true", default=True,
                    help="Use IoU-only cost for low-conf pass to avoid drift")
    ap.add_argument("--reid-sim-thr", type=float, default=0.6, help="Appearance similarity threshold to revive LOST")
    ap.add_argument("--max-age", type=int, default=30, help="Frames before ACTIVE becomes LOST")
    ap.add_argument("--reid-max-age", type=int, default=60, help="Frames to keep LOST before pruning")
    ap.add_argument("--center-gate-base", type=float, default=50.0, help="Re-ID center gate base radius (px)")
    ap.add_argument("--center-gate-slope", type=float, default=10.0, help="Gate growth per frame lost (px/frame)")

    # embedding backend + scheduling
    ap.add_argument("--embedder", type=str, default="dino", help="Embedding backend: dino (default), clip, resnet, ...")
    ap.add_argument("--embed-model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                    help="Embedding model id/name for the chosen embedder")

    ap.add_argument("--crop-pad", type=float, default=0.12, help="Padding ratio around det box for embeddings")
    ap.add_argument("--crop-square", action="store_true", default=True, help="Use square crops for embeddings (default on)")

    ap.add_argument("--embed-amp", choices=["auto","fp16","bf16","off"], default="auto",
                    help="Autocast dtype for embeddings: 'auto' chooses bf16 on Ampere+, fp16 on older GPUs.")
    ap.add_argument("--embed-refresh", type=int, default=5, help="Refresh real embedding every N frames")
    ap.add_argument("--embed-iou-gate", type=float, default=0.6, help="IoU ≥ gate → treat as stable candidate")
    ap.add_argument("--embed-overlap-thr", type=float, default=0.2, help="Det-det IoU > thr marks crowded")
    ap.add_argument("--embed-budget-ms", type=float, default=0.0,
                    help="Max ms per frame for embedding work (0 = unlimited). Budget applies to refresh work only.")

    # draw
    ap.add_argument("--draw-lost", action="store_true", help="Also draw LOST tracks")
    ap.add_argument("--line-thickness", type=int, default=2, help="BBox line thickness")
    ap.add_argument("--font-scale", type=float, default=0.5, help="Label font scale")

    ap.add_argument("--debug", action="store_true", help="Show full tracebacks for debugging")
    return ap.parse_args()


def _fmt_ms(s: float) -> str:  # seconds -> "xx.x ms"
    return f"{s * 1000:.1f} ms"


def _get_embedder_device(embedder) -> str:
    # DINOExtractor exposes the DinoV3Embedder as ._e; prefer explicit .device if present
    dev = getattr(embedder, "device", None)
    if dev is None and hasattr(embedder, "_e"):
        dev = getattr(embedder._e, "device", None)
    if isinstance(dev, torch.device):
        return dev.type
    return str(dev or "cpu")


def compute_embeddings(frame: np.ndarray,
                       boxes: np.ndarray,
                       embedder,
                       tracker,
                       scheduler: EmbeddingScheduler,
                       frame_idx: int,
                       budget_ms: float = 0.0) -> tuple[np.ndarray, float, set[int]]:
    """
    Budgeted embedding computation.
    - Always computes 'critical' embeddings (new/crowded/weak IoU).
    - 'Refresh' embeddings are computed incrementally, honoring budget_ms strictly.
    Returns:
      embs           : (N,D) float32
      t_emb_seconds  : total seconds spent in this step
      refreshed_tids : set of tids whose embedding was (re)computed this frame
    """
    on_cuda = (_get_embedder_device(embedder) == "cuda")
    if on_cuda:
        torch.cuda.synchronize()
    t0 = perf_counter()

    N = len(boxes)
    D = int(getattr(embedder, "dim", getattr(embedder, "emb_dim", 768)))
    if N == 0:
        return np.zeros((0, D), dtype=np.float32), 0.0, set()

    need_mask, reuse_tids, active_snapshot, due_mask, match_tid = scheduler.plan(tracker, boxes, frame_idx)

    embs = np.zeros((N, D), dtype=np.float32)
    refreshed_tids: set[int] = set()

    # Reuse where allowed
    if len(active_snapshot):
        tid2emb = {int(t.tid): t.emb for t in active_snapshot}
        for j, tid in enumerate(reuse_tids):
            if tid is not None and tid in tid2emb:
                embs[j] = tid2emb[tid]
                scheduler.stat_reuse += 1

    # --- 1) Critical set (must compute now; not budgeted) ---
    idx_crit = np.nonzero(need_mask & ~due_mask)[0]
    if len(idx_crit) > 0:
        t1 = perf_counter()
        embs_crit = embedder.embed_crops(frame, boxes[idx_crit])
        if on_cuda:
            torch.cuda.synchronize()
        batch_ms = (perf_counter() - t1) * 1000.0
        embs[idx_crit] = embs_crit
        scheduler.update_ema_ms_per_crop(batch_ms, len(idx_crit))
        scheduler.stat_real += len(idx_crit)
        refreshed_tids |= scheduler.mark_refreshed(active_snapshot, boxes, idx_crit, frame_idx)

    # Remaining budget for refresh (ms); 0 => unlimited
    if budget_ms <= 0:
        remaining_ms = float("inf")
    else:
        if on_cuda:
            torch.cuda.synchronize()
        used_ms = (perf_counter() - t0) * 1000.0
        remaining_ms = max(0.0, budget_ms - used_ms)

    # --- 2) Refresh (optional, strictly honor budget) ---
    if remaining_ms > 0 and len(active_snapshot) > 0:
        def _stale_score(tid: int) -> int:
            return frame_idx - int(scheduler.last_embed_frame.get(int(tid), -10**9))

        due_tids_this = [tid for j, tid in enumerate(match_tid) if due_mask[j] and tid is not None]
        due_order = sorted(set(due_tids_this), key=_stale_score, reverse=True)
        candidates = list(scheduler.pending_refresh) + [t for t in due_order if t not in scheduler.pending_refresh]

        # IoU between ACTIVE boxes and current detections (for picking best det per tid)
        A = np.stack([t.box for t in active_snapshot], axis=0).astype(np.float32) if len(active_snapshot) else np.zeros((0,4), np.float32)
        B = boxes.astype(np.float32)
        if len(A) and len(B):
            x11, y11, x12, y12 = A[:,0][:,None], A[:,1][:,None], A[:,2][:,None], A[:,3][:,None]
            x21, y21, x22, y22 = B[:,0][None,:], B[:,1][None,:], B[:,2][None,:], B[:,3][None,:]
            inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
            inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
            inter = inter_w * inter_h
            area_a = (x12 - x11) * (y12 - y11)
            area_b = (x22 - x21) * (y22 - y21)
            I = inter / (area_a + area_b - inter + 1e-6)  # (T,N)
        else:
            I = np.zeros((0, len(B)), dtype=np.float32)

        best_det_for_tid: dict[int, int] = {}
        if I.size:
            for det_j in range(N):
                trk_i = int(I[:, det_j].argmax()) if I.shape[0] else 0
                tid = int(active_snapshot[trk_i].tid)
                if (tid not in best_det_for_tid) or (I[trk_i, det_j] > I[trk_i, best_det_for_tid[tid]]):
                    best_det_for_tid[tid] = det_j

        chosen_now: list[int] = []
        for tid in candidates:
            if budget_ms <= 0:  # unlimited
                chosen_now.append(int(tid))
                continue
            if on_cuda:
                torch.cuda.synchronize()
            used_ms = (perf_counter() - t0) * 1000.0
            remaining_ms = max(0.0, budget_ms - used_ms)
            est = max(0.5, scheduler.ms_per_crop_ema)
            if remaining_ms < est:
                break
            chosen_now.append(int(tid))

        idx_refresh: list[int] = []
        for tid in chosen_now:
            j = best_det_for_tid.get(int(tid))
            if j is None or j in idx_crit or j in idx_refresh:
                continue
            idx_refresh.append(int(j))

        if len(idx_refresh) > 0:
            t2 = perf_counter()
            embs_ref = embedder.embed_crops(frame, boxes[np.array(idx_refresh, dtype=int)])
            if on_cuda:
                torch.cuda.synchronize()
            batch_ms = (perf_counter() - t2) * 1000.0
            embs[np.array(idx_refresh, dtype=int)] = embs_ref
            scheduler.update_ema_ms_per_crop(batch_ms, len(idx_refresh))
            scheduler.stat_real += len(idx_refresh)
            refreshed_tids |= scheduler.mark_refreshed(active_snapshot, boxes, np.array(idx_refresh, dtype=int), frame_idx)
            scheduler._remove_pending(chosen_now)

    if on_cuda:
        torch.cuda.synchronize()
    return embs, (perf_counter() - t0), refreshed_tids


def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    amp = args.embed_amp
    if amp == "auto":
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            amp = "bf16" if cc[0] >= 8 else "fp16"  # Ampere+ -> bf16, Turing (T4) -> fp16
        else:
            amp = "off"

    embedder = create_extractor(
        args.embedder, args.embed_model, device,
        autocast=(device == "cuda"),
        amp_dtype=("fp16" if amp == "fp16" else "bf16" if amp == "bf16" else None),
        pad=args.crop_pad, square=args.crop_square
    )

    tracker = SimpleTracker(
        iou_weight=0.3,
        app_weight=0.7,
        iou_thresh=0.3,
        iou_thresh_low=0.2,
        reid_sim_thresh=args.reid_sim_thr,
        max_age=args.max_age,
        reid_max_age=args.reid_max_age,
        ema_alpha=0.9,
        gallery_size=10,
        use_hungarian=not args.no_hungarian,
        class_consistent=True,
        class_penalty=args.class_penalty,
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        conf_min_update=args.conf_min_update,
        conf_update_weight=args.conf_update_weight,
        low_conf_iou_only=args.low_conf_iou_only,
        center_gate_base=args.center_gate_base,
        center_gate_slope=args.center_gate_slope,
    )

    sched = EmbeddingScheduler(SchedulerConfig(
        overlap_thr=args.embed_overlap_thr,
        iou_gate=args.embed_iou_gate,
        refresh_every=args.embed_refresh,
    ))

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
    frame_idx = 0  # needed by scheduler
    ema_fps = None
    alpha = 0.15  # smoothing for FPS

    # Stage accumulators (seconds)
    acc_read = acc_det = acc_emb = acc_track = acc_draw = acc_write = 0.0
    acc_frame = 0.0
    frame_times: list[float] = []  # per-frame total time (seconds) for p50/p95

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
        embs, t_emb, refreshed_tids = compute_embeddings(
            frame, boxes, embedder, tracker, sched, frame_idx, budget_ms=args.embed_budget_ms
        )

        # --- TRACK ---
        t = perf_counter()
        tracks = tracker.update(boxes, embs, confs=confs, clses=clses)
        t_track = perf_counter() - t

        # --- DRAW ---
        t = perf_counter()
        draw_tracks(
            frame, tracks,
            draw_lost=args.draw_lost,
            thickness=args.line_thickness,
            font_scale=args.font_scale,
            mark_tids=refreshed_tids   # label gets a "*" if an embed was computed this frame
        )
        t_draw = perf_counter() - t

        # --- WRITE ---
        t = perf_counter()
        out.write(frame)
        t_write = perf_counter() - t

        # Totals / stats
        f_dt = perf_counter() - f0
        frames += 1
        frame_idx += 1
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

    def pct(x): return 100.0 * x / max(1e-9, acc_frame)

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
    print(f"Done. Wrote {args.output}. {elapsed:.1f}s total")
    print(sched.summary(frames))


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
