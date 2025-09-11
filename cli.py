# cli.py (Refactored)
import os
import argparse
import sys
from dataclasses import dataclass, field, asdict
from time import perf_counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load .env file for HF_TOKEN
load_dotenv()

from mot_dinov3 import compat
compat.apply(strict_numpy=False, quiet=True)

from mot_dinov3.detector import Detector
from mot_dinov3.embedder import GatedModelAccessError
from mot_dinov3.features.factory import create_extractor
from mot_dinov3.scheduler import EmbeddingScheduler
from mot_dinov3.tracker import SimpleTracker
from mot_dinov3.viz import draw_tracks


# REFACTOR: Use dataclasses to group component parameters for clarity
@dataclass
class SchedulerParams:
    overlap_thr: float
    iou_gate: float
    refresh_every: int

@dataclass
class TrackerParams:
    iou_weight: float = 0.3
    app_weight: float = 0.7
    iou_thresh: float = 0.3
    iou_thresh_low: float = 0.2
    reid_sim_thresh: float = 0.6
    max_age: int = 30
    reid_max_age: int = 60
    ema_alpha: float = 0.9
    gallery_size: int = 10
    use_hungarian: bool = True
    class_consistent: bool = True
    class_penalty: float = 0.15
    conf_high: float = 0.5
    conf_low: float = 0.1
    conf_min_update: float = 0.3
    conf_update_weight: float = 0.5
    low_conf_iou_only: bool = True
    center_gate_base: float = 50.0
    center_gate_slope: float = 10.0
    class_vote_smoothing: float = 0.6
    class_decay_factor: float = 0.05

@dataclass
class Stats:
    frame_times: List[float] = field(default_factory=list)
    stage_timers: Dict[str, float] = field(default_factory=lambda: {
        "read": 0.0, "detect": 0.0, "embed": 0.0, 
        "track": 0.0, "draw": 0.0, "write": 0.0,
    })

def parse_args():
    ap = argparse.ArgumentParser(description="DINOv3-based MOT (tracking-by-detection)")
    
    # REFACTOR: Group arguments for better readability in --help
    g_io = ap.add_argument_group("I/O")
    g_io.add_argument("--source", type=str, required=True, help="Path to input video")
    g_io.add_argument("--output", type=str, default="outputs/tracked.mp4", help="Output video path")
    g_io.add_argument("--fps", type=float, default=0.0, help="Force output FPS (0 uses source FPS)")

    g_det = ap.add_argument_group("Detector")
    g_det.add_argument("--det", type=str, default="yolov8n.pt", help="Ultralytics model path or name")
    g_det.add_argument("--conf", type=float, default=0.3, help="Detector confidence threshold")
    g_det.add_argument("--imgsz", type=int, default=960, help="Detector input size")
    g_det.add_argument("--classes", type=str, default="", help="Comma-separated class ids to keep (optional)")

    g_emb = ap.add_argument_group("Embedding & Scheduling")
    g_emb.add_argument("--embedder", type=str, default="dino", help="Embedding backend: dino (default), clip, etc.")
    g_emb.add_argument("--embed-model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Embedding model ID")
    g_emb.add_argument("--dinov3", type=str, help="[DEPRECATED] Use --embed-model instead")
    g_emb.add_argument("--fallback-open", action="store_true", help="On auth error, fallback to open DINOv2")
    g_emb.add_argument("--embed-amp", choices=["auto", "fp16", "bf16", "off"], default="auto", help="Autocast dtype")
    g_emb.add_argument("--crop-pad", type=float, default=0.12, help="Padding ratio for embedding crops")
    g_emb.add_argument("--crop-square", action="store_true", default=True, help="Use square crops for embeddings")
    g_emb.add_argument("--embed-budget-ms", type=float, default=0.0, help="Max ms per frame for refresh work (0=unlimited)")
    g_emb.add_argument("--embed-refresh", type=int, default=5, help="Refresh embedding every N frames")
    g_emb.add_argument("--embed-iou-gate", type=float, default=0.6, help="IoU ≥ gate → stable candidate")
    g_emb.add_argument("--embed-overlap-thr", type=float, default=0.2, help="Det-det IoU > thr → crowded")

    g_trk = ap.add_argument_group("Tracker")
    g_trk.add_argument("--no-hungarian", action="store_true", help="Use greedy instead of Hungarian assignment")
    g_trk.add_argument("--class-penalty", type=float, default=0.15, help="Cost for class mismatch")
    g_trk.add_argument("--conf-high", type=float, default=0.5, help="High-confidence detection threshold")
    g_trk.add_argument("--conf-low", type=float, default=0.1, help="Low-confidence detection threshold")
    g_trk.add_argument("--conf-min-update", type=float, default=0.3, help="Min confidence to update track embedding")
    g_trk.add_argument("--conf-update-weight", type=float, default=0.5, help="EMA strength scaling with confidence")
    g_trk.add_argument("--low-conf-iou-only", action="store_true", default=True, help="Use IoU-only cost for low-conf pass")
    g_trk.add_argument("--reid-sim-thr", type=float, default=0.6, help="Similarity threshold to revive LOST tracks")
    g_trk.add_argument("--max-age", type=int, default=30, help="Frames before ACTIVE track becomes LOST")
    g_trk.add_argument("--reid-max-age", type=int, default=60, help="Frames to keep LOST track before pruning")
    g_trk.add_argument("--center-gate-base", type=float, default=50.0, help="Re-ID center gate base radius (px)")
    g_trk.add_argument("--center-gate-slope", type=float, default=10.0, help="Gate growth per frame lost (px/frame)")

    g_viz = ap.add_argument_group("Visualization")
    g_viz.add_argument("--draw-lost", action="store_true", help="Also draw LOST tracks")
    g_viz.add_argument("--line-thickness", type=int, default=2, help="BBox line thickness")
    g_viz.add_argument("--font-scale", type=float, default=0.5, help="Label font scale")

    g_sys = ap.add_argument_group("System")
    g_sys.add_argument("--cpu", action="store_true", help="Force CPU for all operations")
    g_sys.add_argument("--debug", action="store_true", help="Show full tracebacks for debugging")

    args = ap.parse_args()
    
    # REFACTOR: Handle legacy --dinov3 argument gracefully
    if args.dinov3:
        print("[WARNING] --dinov3 is deprecated. Please use --embed-model instead.")
        # Only override if the new argument was not explicitly set
        if args.embed_model == ap.get_default("embed_model"):
            args.embed_model = args.dinov3

    return args

def setup_video_io(source_path: str, output_path: str, fps_override: float) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, dict]:
    """Initializes video capture and writer objects."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_path}")

    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "src_fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["out_fps"] = fps_override if fps_override > 0 else meta["src_fps"]
    if meta["total_frames"] <= 0: meta["total_frames"] = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(output_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    
    writer = cv2.VideoWriter(output_path, fourcc, meta["out_fps"], (meta["width"], meta["height"]))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")
        
    return cap, writer, meta

def initialize_components(args: argparse.Namespace, device: str) -> Dict[str, object]:
    """Initializes all the main processing modules."""
    # REFACTOR: Detector no longer needs a device argument
    detector = Detector(args.det, imgsz=args.imgsz)

    if args.embed_amp == "auto":
        amp_dtype = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    else:
        amp_dtype = args.embed_amp if args.embed_amp != "off" else None

    embedder = create_extractor(
        args.embedder, args.embed_model, device,
        autocast=(device == "cuda"), amp_dtype=amp_dtype,
        pad=args.crop_pad, square=args.crop_square
    )
    
    # REFACTOR: Use dataclass to configure the tracker
    tracker_params = TrackerParams(
        use_hungarian=not args.no_hungarian,
        class_penalty=args.class_penalty,
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        conf_min_update=args.conf_min_update,
        conf_update_weight=args.conf_update_weight,
        low_conf_iou_only=args.low_conf_iou_only,
        reid_sim_thresh=args.reid_sim_thr,
        max_age=args.max_age,
        reid_max_age=args.reid_max_age,
        center_gate_base=args.center_gate_base,
        center_gate_slope=args.center_gate_slope,
    )

    tracker = SimpleTracker(**asdict(tracker_params))

    scheduler_params = SchedulerParams(
        overlap_thr=args.embed_overlap_thr,
        iou_gate=args.embed_iou_gate,
        refresh_every=args.embed_refresh,
    )
    scheduler = EmbeddingScheduler(scheduler_params)
    
    return {"detector": detector, "embedder": embedder, "tracker": tracker, "scheduler": scheduler}

def run_processing_loop(args: argparse.Namespace, cap: cv2.VideoCapture, writer: cv2.VideoWriter, 
                        components: Dict[str, object], total_frames: int, device: str) -> Stats:
    """The main frame-by-frame processing loop."""
    stats = Stats()
    frame_idx = 0
    keep_ids = set(int(x) for x in args.classes.split(",") if x.strip().isdigit()) if args.classes else None
    
    with tqdm(total=total_frames, unit="frame", dynamic_ncols=True, desc=f"Tracking [{device}]") as pbar:
        while True:
            f0 = perf_counter()
            t = perf_counter()
            ok, frame = cap.read()
            stats.stage_timers["read"] += perf_counter() - t
            if not ok: break

            t = perf_counter()
            boxes, confs, clses = components["detector"].detect(frame, conf_thres=args.conf)
            stats.stage_timers["detect"] += perf_counter() - t

            if keep_ids and len(boxes) > 0:
                mask = np.array([c in keep_ids for c in clses], dtype=bool)
                boxes, confs, clses = boxes[mask], confs[mask], clses[mask]

            embs, t_emb, refreshed_tids = components["scheduler"].run(
                frame=frame, boxes=boxes, embedder=components["embedder"], tracker=components["tracker"],
                frame_idx=frame_idx, budget_ms=args.embed_budget_ms
            )
            stats.stage_timers["embed"] += t_emb

            t = perf_counter()
            tracks = components["tracker"].update(boxes, embs, confs=confs, clses=clses)
            stats.stage_timers["track"] += perf_counter() - t

            t = perf_counter()
            draw_tracks(frame, tracks, draw_lost=args.draw_lost, thickness=args.line_thickness,
                        font_scale=args.font_scale, mark_tids=refreshed_tids)
            stats.stage_timers["draw"] += perf_counter() - t

            t = perf_counter()
            writer.write(frame)
            stats.stage_timers["write"] += perf_counter() - t

            stats.frame_times.append(perf_counter() - f0)
            frame_idx += 1
            pbar.update(1)
    
    return stats

def print_summary(args: argparse.Namespace, stats: Stats, meta: dict, wall_time: float, device: str, sched: EmbeddingScheduler):
    """Prints the final performance and tracking summary."""
    frames = len(stats.frame_times)
    if frames == 0:
        print("No frames were processed.")
        return

    total_time_s = sum(stats.frame_times)
    mean_ms = 1000 * total_time_s / frames
    eff_fps = frames / wall_time
    p50, p95 = 1000 * np.percentile(np.array(stats.frame_times), [50, 95])

    def pct(x): return 100 * x / total_time_s
    def fmt_ms(s): return f"{1000 * s / frames:.1f} ms"
    
    print("\n" + "="*20 + " Tracking Summary " + "="*20)
    print(f"  Video Source:   {args.source}  ->  {args.output}")
    print(f"  Resolution:     {meta['width']}x{meta['height']} @ {meta['src_fps']:.2f} fps (output {meta['out_fps']:.2f} fps)")
    print(f"  Device:         {device}")
    print(f"  Frames:         {frames}  |  Wall Time: {wall_time:.2f}s  |  Effective FPS: {eff_fps:.1f}")
    print(f"  Latency/frame:  mean {mean_ms:.1f} ms  |  p50 {p50:.1f} ms  |  p95 {p95:.1f} ms")
    print("  Stage Breakdown (mean per frame | % of total):")
    for name, seconds in stats.stage_timers.items():
        print(f"    - {name:<7}: {fmt_ms(seconds):<10} | {pct(seconds):5.1f}%")
    
    print("-" * 58)
    print(sched.summary(frames))
    print("=" * 58)

def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    interrupted = False

    try:
        cap, writer, meta = setup_video_io(args.source, args.output, args.fps)
        components = initialize_components(args, device)
        
        t_start = perf_counter()
        stats = run_processing_loop(args, cap, writer, components, meta["total_frames"], device)
        wall_time = perf_counter() - t_start

    except KeyboardInterrupt:
        interrupted = True
        wall_time = perf_counter() - t_start
    finally:
        if 'cap' in locals(): cap.release()
        if 'writer' in locals(): writer.release()
        cv2.destroyAllWindows()
    
    if interrupted:
        print("\n^C received — stopping gracefully. Partial output saved.")

    if 'stats' in locals():
        print_summary(args, stats, meta, wall_time, device, components["scheduler"])

if __name__ == "__main__":
    try:
        main()
    except GatedModelAccessError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        if "--debug" in sys.argv or os.getenv("MOT_DEBUG") == "1":
            raise
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)