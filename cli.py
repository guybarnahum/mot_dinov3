"""
DINOv3-based MOT (tracking-by-detection) Command-Line Interface.

This script processes a video to perform object detection and tracking,
saving the output with visual overlays. It now supports loading default
parameters from a TOML configuration file and processing specific
video segments defined by start and end frames.
"""
import os
import argparse
import sys
from dataclasses import dataclass, field, asdict, fields, MISSING
from time import perf_counter
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

# FIX: Import tomllib for Python 3.11+ and fall back to tomli
try:
    import tomllib  # Standard library in Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python < 3.11
    except ImportError:
        print("Error: 'tomli' is required for this script on Python < 3.11.", file=sys.stderr)
        print("Please run: pip install tomli", file=sys.stderr)
        sys.exit(1)

from mot_dinov3 import compat
compat.apply(strict_numpy=False, quiet=True)

from mot_dinov3.detector import Detector
from mot_dinov3.embedder import GatedModelAccessError
from mot_dinov3.features.factory import create_extractor
from mot_dinov3.scheduler import EmbeddingScheduler
from mot_dinov3.tracker import SimpleTracker
from mot_dinov3.viz import draw_tracks


# --- Configuration Dataclasses ---
# Centralizes all script parameters for clarity and TOML integration.

@dataclass
class IOParams:
    # FIX: Make source optional with a default of None to allow delayed initialization
    source: Optional[str] = None
    output: str = "outputs/tracked.mp4"
    fps: float = 0.0
    start_frame: int = 0
    end_frame: Optional[int] = None

@dataclass
class DetectorParams:
    det: str = "yolov8n.pt"
    conf: float = 0.3
    imgsz: int = 960
    classes: str = ""

@dataclass
class EmbedderParams:
    embedder: str = "dino"
    embed_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    fallback_open: bool = False
    embed_amp: str = "auto"
    crop_pad: float = 0.12
    crop_square: bool = True

@dataclass
class SchedulerParams:
    overlap_thr: float = 0.2
    iou_gate: float = 0.6
    refresh_every: int = 5
    ema_alpha: float = 0.35
    embed_budget_ms: float = 0.0

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
class VizParams:
    draw_lost: bool = False
    line_thickness: int = 2
    font_scale: float = 0.5

@dataclass
class SystemParams:
    cpu: bool = False
    debug: bool = False
    config: Optional[str] = None

@dataclass
class Config:
    io: IOParams = field(default_factory=IOParams)
    detector: DetectorParams = field(default_factory=DetectorParams)
    embedder: EmbedderParams = field(default_factory=EmbedderParams)
    scheduler: SchedulerParams = field(default_factory=SchedulerParams)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    viz: VizParams = field(default_factory=VizParams)
    system: SystemParams = field(default_factory=SystemParams)

# --- Performance Statistics ---
@dataclass
class Stats:
    frame_times: List[float] = field(default_factory=list)
    stage_timers: Dict[str, float] = field(default_factory=lambda: {
        "read": 0.0, "detect": 0.0, "embed": 0.0,
        "track": 0.0, "draw": 0.0, "write": 0.0,
    })

# --- Argument Parsing and Configuration ---

def _create_arg_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with all CLI options."""
    ap = argparse.ArgumentParser(description="DINOv3-based MOT with config file support.")
    
    # Special argument for config file
    ap.add_argument("-c", "--config", type=str, help="Path to a TOML configuration file")

    # Arguments to exclude from automatic generation to avoid conflicts.
    # These can still be set in the config.toml file under their respective sections.
    EXCLUDE_ARGS = {'ema_alpha'}

    # Define groups for organized --help output
    groups = {
        'io': ap.add_argument_group("I/O"),
        'detector': ap.add_argument_group("Detector"),
        'embedder': ap.add_argument_group("Embedding & Scheduling"),
        'scheduler': ap.add_argument_group("Scheduler"),
        'tracker': ap.add_argument_group("Tracker"),
        'viz': ap.add_argument_group("Visualization"),
        'system': ap.add_argument_group("System")
    }
    
    # Add arguments from dataclasses to the parser
    # FIX: Correctly iterate over Field objects
    for config_field in fields(Config()):
        section_name = config_field.name
        section_dc = config_field.type
        group = groups.get(section_name, ap)
        for field_info in fields(section_dc):
            # FIX: Skip excluded args to prevent conflicts
            if field_info.name == 'config' or field_info.name in EXCLUDE_ARGS:
                continue
            
            cli_name = f"--{field_info.name.replace('_', '-')}"
            
            arg_kwargs = {'type': field_info.type, 'default': argparse.SUPPRESS}
            if field_info.type == bool:
                arg_kwargs['action'] = 'store_true'
                del arg_kwargs['type']
            # Help text generation
            help_text = ""
            # FIX: Check against the imported MISSING constant directly
            if field_info.default is not MISSING:
                 help_text = f"(Default: {field_info.default})"
           
            group.add_argument(cli_name, help=help_text, **arg_kwargs)

    return ap


def parse_and_merge_config() -> Config:
    """Parses CLI args, loads TOML config, and merges them."""
    parser = _create_arg_parser()
    args = parser.parse_args()
    
    config_data = {}
    config_path = getattr(args, 'config', None)
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

    final_config = Config()
    cli_args_dict = vars(args)

    # FIX: Correctly iterate over Field objects
    for config_field in fields(final_config):
        section_name = config_field.name
        section_dc = config_field.type
        section_config = config_data.get(section_name, {})
        for field_info in fields(section_dc):
            value = cli_args_dict.get(field_info.name)
            if value is None:
                value = section_config.get(field_info.name)
            if value is not None:
                setattr(getattr(final_config, section_name), field_info.name, value)

    return final_config

# --- Core Application Logic ---

def setup_video_io(cfg: IOParams, meta: Dict[str, Any]) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """Initializes video capture and writer objects."""
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {cfg.source}")

    meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta["src_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta["out_fps"] = cfg.fps if cfg.fps > 0 else meta["src_fps"]

    if cfg.start_frame > 0:
        if cfg.start_frame >= meta["total_frames"]:
            raise ValueError(f"Start frame ({cfg.start_frame}) is after the end of the video ({meta['total_frames']})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(cfg.output)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    
    writer = cv2.VideoWriter(cfg.output, fourcc, meta["out_fps"], (meta["width"], meta["height"]))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {cfg.output}")
        
    return cap, writer

def initialize_components(cfg: Config, device: str) -> Dict[str, object]:
    """Initializes all the main processing modules."""
    detector = Detector(cfg.detector.det, imgsz=cfg.detector.imgsz)

    if cfg.embedder.embed_amp == "auto":
        amp_dtype = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    else:
        amp_dtype = cfg.embedder.embed_amp if cfg.embedder.embed_amp != "off" else None

    embedder = create_extractor(
        cfg.embedder.embedder, cfg.embedder.embed_model, device,
        autocast=(device == "cuda"), amp_dtype=amp_dtype,
        pad=cfg.embedder.crop_pad, square=cfg.embedder.crop_square
    )
    
    tracker = SimpleTracker(**asdict(cfg.tracker))
    scheduler = EmbeddingScheduler(cfg.scheduler)
    
    return {"detector": detector, "embedder": embedder, "tracker": tracker, "scheduler": scheduler}

def run_processing_loop(cfg: Config, cap: cv2.VideoCapture, writer: cv2.VideoWriter,
                        components: Dict[str, object], frames_to_process: int, device: str) -> Stats:
    """The main frame-by-frame processing loop."""
    stats = Stats()
    frame_idx = cfg.io.start_frame
    end_frame = cfg.io.end_frame or float('inf')
    keep_ids = set(int(x) for x in cfg.detector.classes.split(",") if x.strip().isdigit()) if cfg.detector.classes else None
    
    desc = f"Tracking [{device}] frames {cfg.io.start_frame} to {cfg.io.end_frame or 'end'}"
    with tqdm(total=frames_to_process, unit="frame", dynamic_ncols=True, desc=desc) as pbar:
        while frame_idx < end_frame:
            f0 = perf_counter()
            t = perf_counter()
            ok, frame = cap.read()
            stats.stage_timers["read"] += perf_counter() - t
            if not ok: break

            t = perf_counter()
            boxes, confs, clses = components["detector"].detect(frame, conf_thres=cfg.detector.conf)
            stats.stage_timers["detect"] += perf_counter() - t

            if keep_ids and len(boxes) > 0:
                mask = np.array([c in keep_ids for c in clses], dtype=bool)
                boxes, confs, clses = boxes[mask], confs[mask], clses[mask]

            embs, t_emb, refreshed_tids = components["scheduler"].run(
                frame=frame, boxes=boxes, embedder=components["embedder"], tracker=components["tracker"],
                frame_idx=frame_idx, budget_ms=cfg.scheduler.embed_budget_ms
            )
            stats.stage_timers["embed"] += t_emb

            t = perf_counter()
            tracks = components["tracker"].update(boxes, embs, confs=confs, clses=clses)
            stats.stage_timers["track"] += perf_counter() - t

            t = perf_counter()
            draw_tracks(frame, tracks, draw_lost=cfg.viz.draw_lost, thickness=cfg.viz.line_thickness,
                        font_scale=cfg.viz.font_scale, mark_tids=refreshed_tids)
            stats.stage_timers["draw"] += perf_counter() - t

            t = perf_counter()
            writer.write(frame)
            stats.stage_timers["write"] += perf_counter() - t

            stats.frame_times.append(perf_counter() - f0)
            frame_idx += 1
            pbar.update(1)
    
    return stats

def print_summary(cfg: Config, stats: Stats, meta: dict, wall_time: float, device: str, sched: EmbeddingScheduler):
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
    print(f"  Video Source:   {cfg.io.source}  ->  {cfg.io.output}")
    print(f"  Frame Range:    {cfg.io.start_frame} to {cfg.io.end_frame or meta['total_frames']}")
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
    cfg = parse_and_merge_config()
    device = "cpu" if cfg.system.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    interrupted = False
    
    # FIX: Add validation check for source after all configs are merged
    if cfg.io.source is None:
        raise ValueError("Input video --source is required. Please provide it via the command line or in the config file.")

    video_meta = {}
    
    try:
        cap, writer = setup_video_io(cfg.io, video_meta)

        end_frame = cfg.io.end_frame if cfg.io.end_frame is not None else video_meta["total_frames"]
        frames_to_process = end_frame - cfg.io.start_frame
        if frames_to_process <= 0:
            raise ValueError("End frame must be greater than start frame.")

        components = initialize_components(cfg, device)
        
        t_start = perf_counter()
        stats = run_processing_loop(cfg, cap, writer, components, frames_to_process, device)
        wall_time = perf_counter() - t_start

    except KeyboardInterrupt:
        interrupted = True
        if 't_start' in locals():
            wall_time = perf_counter() - t_start
    finally:
        if 'cap' in locals(): cap.release()
        if 'writer' in locals(): writer.release()
        cv2.destroyAllWindows()
    
    if interrupted:
        print("\n^C received — stopping gracefully. Partial output saved.")

    if 'stats' in locals() and stats.frame_times:
        print_summary(cfg, stats, video_meta, wall_time, device, components["scheduler"])

if __name__ == "__main__":
    try:
        main()
    except (GatedModelAccessError, FileNotFoundError, ValueError) as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        if "--debug" in sys.argv or os.getenv("MOT_DEBUG") == "1":
            raise
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)



