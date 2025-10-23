#!/usr/bin/env python3
"""
Main script to orchestrate:
- Detect connected RealSense cameras
- Start per-cam capture threads
- Start optional movement & emotion processing threads
- Start optional audio threads
- Optional unified preview window (2x2 grid)
Controls (terminal):
  • SPACE → toggle Pause/Resume
  • g     → reopen the unified preview grid (if closed)
  • q     → close ONLY the preview grid (pipeline continues; handled in PreviewGrid)
  • ESC   → Stop gracefully (terminal or window)
  • Ctrl+C → Stop gracefully (SIGINT)
"""

import os, sys, time, threading, queue, signal, argparse
from pathlib import Path
from typing import Dict, Set
import subprocess

import pyrealsense2 as rs

from tunables import *
from preview_grid import PreviewGrid
from capture_worker import capture_worker
from movement_processor import processor_worker
from emotion_processor import emotion_worker
from camera_utils import load_serial_map
from audio_worker import audio_worker

# NEW: shared control flags (pause/stop) used by all modules
from control_flags import pause_event, stop_event

def _sig_stop(signum, frame):
    print("\n[⛔] SIGINT → stopping…", flush=True)
    stop_event.set()
signal.signal(signal.SIGINT, _sig_stop)

# --- Terminal keyboard listener (SPACE / 'g' / ESC) ---
def start_keyboard_listener(preview_ref):
    """Listen on the terminal for SPACE (pause), 'g' (reopen grid), ESC (stop).
       Skips if stdin is not a TTY (e.g., launched from an IDE)."""
    if not sys.stdin.isatty():
        print("[INFO] Keyboard listener disabled (stdin is not a TTY).", flush=True)
        return None

    import termios, tty, select
    fd = sys.stdin.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except Exception:
        print("[WARN] Could not configure terminal keyboard listener.", flush=True)
        return None

    def _run():
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == ' ':
                    if pause_event.is_set():
                        pause_event.clear()
                        print("[▶] Resume (space)", flush=True)
                    else:
                        pause_event.set()
                        print("[⏸] Pause (space)", flush=True)
                elif ch in ('g', 'G'):
                    if preview_ref is not None:
                        try:
                            preview_ref.reopen_window()
                            print("[🪟] Preview reopened.", flush=True)
                        except Exception:
                            pass
                elif ch == '\x1b':  # ESC
                    print("[⛔] ESC pressed (terminal) → stopping.", flush=True)
                    stop_event.set()
                    try:
                        os.kill(os.getpid(), signal.SIGINT)  # nudge blocking waits
                    except Exception:
                        pass
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    t = threading.Thread(target=_run, name="kbd-listener", daemon=True)
    t.start()
    return t

def build_argparser():
    ap = argparse.ArgumentParser(description="Real-time unified capture+process+audio pipeline (modular)")
    ap.add_argument("--output-dir", required=True, help="Base output directory")
    ap.add_argument("--duration-sec", type=float, required=True, help="ACTIVE duration in seconds")
    ap.add_argument("--save-every", type=int, default=1, help="Save raw color/depth every Nth frame (0=off)")
    ap.add_argument("--filters", choices=["on","off"], default="off", help="Depth filters on/off")
    ap.add_argument("--viz-live", choices=["off","window"], default="off", help="Preview window (unified 2x2 grid)")
    ap.add_argument("--force-flip", choices=["flip","same"], default="flip", help="Global handedness flip baseline")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame for movement")
    ap.add_argument("--backpressure", choices=["drop-latest","block"], default="drop-latest", help="Processor queue policy")
    ap.add_argument("--csv-flush", type=int, default=30, help="CSV flush interval (frames) for movement landmarks")
    ap.add_argument("--log-flush-sec", type=int, default=5, help="Logger flush interval")
    ap.add_argument("--emo-history", type=int, default=240, help="Frames kept in on-screen VA plot (per cam)")
    ap.add_argument("--emo-stride", type=int, default=1, help="Process every Nth frame for emotion")
    ap.add_argument("--emo-csv-flush", type=int, default=30, help="CSV flush interval (frames) for emotion")

    # —— Movement vs Emotion camera selection ——
    ap.add_argument("--process-mov-cams", nargs="*", help="Labels to process for hand movement.")
    ap.add_argument("--process-emo-cams", nargs="*", help="Labels to process for emotion (valence/arousal).")

    # —— Audio ——
    ap.add_argument("--audio-out", default=None, help="Audio output directory (default: <output-dir>/audio)")
    ap.add_argument("--audio-duration-sec", type=float, default=0, help="Active duration for audio (0=off)")
    ap.add_argument("--rate", type=int, choices=[44100,48000], default=44100, help="Audio sample rate")

    # Annotated movement preview saver (already re-added earlier)
    ap.add_argument("--viz-save-every", type=int, default=3,
                    help="Save annotated movement preview every N frames (0 = OFF)")
    return ap

def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover cams
    serial_to_label = load_serial_map()
    ctx = rs.context()
    connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    active = [(s, serial_to_label.get(s, f"cam_{s[-4:]}")) for s in connected]
    if not active:
        print("[ERROR] No RealSense cameras found.")
        return
    print(f"[INFO] Connected cams: {', '.join([f'{lab}({s})' for s,lab in active])}")

    # Movement selection:
    if args.process_mov_cams is not None:
        process_set_mov: Set[str] = set([lab.strip() for lab in args.process_mov_cams if lab and lab.strip()])
    else:
        process_set_mov = set(lab for _, lab in active)  # default: all for movement

    # Emotion selection:
    if args.process_emo_cams is None:
        process_set_emo: Set[str] = set()  # default: none
    else:
        process_set_emo = set([lab.strip() for lab in args.process_emo_cams if lab and lab.strip()])

    print(f"[INFO] Movement cams: {sorted(process_set_mov) if process_set_mov else 'NONE'}")
    print(f"[INFO] Emotion cams:  {sorted(process_set_emo) if process_set_emo else 'NONE'}")

    # Start unified preview if requested
    PREVIEW = None
    if args.viz_live == "window":
        PREVIEW = PreviewGrid(title="Unified Preview", history_len=max(240, args.emo_history), target_fps=30)
        PREVIEW.start()

    # Start terminal keyboard listener (SPACE / g / ESC)
    _kbd = start_keyboard_listener(PREVIEW)

    # Spawn per-cam queues & threads
    cap_threads = []
    proc_threads = []
    queues_mov: Dict[str, queue.Queue] = {}
    queues_emo: Dict[str, queue.Queue] = {}

    for serial, label in active:
        q_mov = None
        q_emo = None

        if label in process_set_mov:
            q_mov = queue.Queue(maxsize=8)
            queues_mov[label] = q_mov
            t_proc_mov = threading.Thread(
                target=processor_worker,
                args=(label, q_mov, out_dir, args.force_flip, max(1,args.stride),
                      max(1,args.csv_flush), max(1,args.log_flush_sec), PREVIEW, max(0, args.viz_save_every)),
                daemon=True, name=f"proc-mov-{label}"
            )
            t_proc_mov.start()
            proc_threads.append(t_proc_mov)

        if label in process_set_emo:
            q_emo = queue.Queue(maxsize=8)
            queues_emo[label] = q_emo
            t_proc_emo = threading.Thread(
                target=emotion_worker,
                args=(label, q_emo, out_dir, max(1,args.emo_stride),
                      max(1,args.emo_csv_flush), max(1,args.log_flush_sec), max(10,args.emo_history), PREVIEW),
                daemon=True, name=f"proc-emo-{label}"
            )
            t_proc_emo.start()
            proc_threads.append(t_proc_emo)

        t_cap = threading.Thread(
            target=capture_worker,
            args=(serial, label, out_dir, float(args.duration_sec), max(0,args.save_every),
                  (args.filters=="on"), queues_mov.get(label, None), queues_emo.get(label, None), args.backpressure),
            daemon=True, name=f"cap-{label}"
        )
        t_cap.start()
        cap_threads.append(t_cap)
    
    # --------- Audio: only start for whitelisted devices that are PRESENT ----------
    def _list_alsa_hw_devices():
        try:
            out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT, text=True)
        except Exception:
            return []
        found = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("card "):
                try:
                    parts = [p.strip() for p in line.split(",")]
                    cidx = int(parts[0].split()[1].rstrip(":"))
                    didx = int(parts[1].split()[1].rstrip(":"))
                    found.append(f"hw:{cidx},{didx}")
                except Exception:
                    continue
        return found

    audio_threads = []
    aud_dir = Path(args.audio_out) if args.audio_out else (out_dir / "audio")
    if args.audio_duration_sec > 0:
        present = set(_list_alsa_hw_devices())
        # Only use the intersection of PRESENT devices and your WHITELIST
        enabled = [d for d in VALID_MIC_IDS if d in present]
        if not enabled:
            print("[INFO] 🎙 No whitelisted mics detected; skipping audio.")
        else:
            for dev in enabled:
                t = threading.Thread(
                    target=audio_worker,
                    args=(dev, aud_dir, float(args.audio_duration_sec), int(args.rate)),
                    daemon=True, name=f"aud-{dev}"
                )
                t.start()
                audio_threads.append(t)
            print(f"[INFO] 🎙 Audio enabled on {len(audio_threads)} mic(s): {', '.join(enabled)} → {aud_dir}")

    # Wait for capture to finish, then signal processors to exit by closing queues, then join audio
    try:
        for t in cap_threads:
            t.join()
    finally:
        for q in list(queues_mov.values()) + list(queues_emo.values()):
            setattr(q, "closed", True)
        for t in proc_threads:
            t.join()
        for t in audio_threads:
            t.join()
        if PREVIEW is not None:
            PREVIEW.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("[🏁] Done.")

if __name__ == "__main__":
    main()
