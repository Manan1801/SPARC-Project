#!/usr/bin/env python3

import os
import argparse
import subprocess
import signal
import time
from datetime import datetime
import threading
from mic_config import VALID_MIC_IDS  # Import your mic config

# Keep your alias
VALID_MIC_DEVICES = VALID_MIC_IDS  # Use the mic IDs defined in mic_config.py

# ── Cooperative control flags ────────────────────────────────────────────────
pause_event = threading.Event()   # set() => paused
stop_event  = threading.Event()   # set() => stop

def _sig_pause(signum, frame):
    if not pause_event.is_set():
        print("\n[⏸] Received SIGUSR1 → PAUSE (freezing arecord).", flush=True)
    pause_event.set()

def _sig_resume(signum, frame):
    if pause_event.is_set():
        print("\n[▶] Received SIGUSR2 → RESUME.", flush=True)
    pause_event.clear()

def _sig_stop(signum, frame):
    print("\n[⛔] Received SIGINT → Graceful stop requested.", flush=True)
    stop_event.set()

signal.signal(signal.SIGUSR1, _sig_pause)
signal.signal(signal.SIGUSR2, _sig_resume)
signal.signal(signal.SIGINT,  _sig_stop)

# ── Per-mic worker ───────────────────────────────────────────────────────────
def record_from_hw(device_str, output_path, duration_sec):
    """
    Start arecord without -d and cooperatively count ACTIVE time.
    Pause:  SIGUSR1 → SIGSTOP arecord
    Resume: SIGUSR2 → SIGCONT arecord
    Stop:   SIGINT  → SIGINT arecord (finalizes WAV header)
    """
    cmd = [
        "arecord",
        "-D", device_str,
        "-f", "cd",
        "-c", "1",
        "-r", "44100",
        "-t", "wav",
        output_path
    ]
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[INFO] 🎙 Started {device_str} → {output_path}")

        target_active = float(duration_sec)
        active_elapsed = 0.0
        last = time.monotonic()

        # Track whether we have already stopped the child to avoid spamming signals
        child_paused = False

        while not stop_event.is_set() and active_elapsed < target_active:
            now = time.monotonic()
            if pause_event.is_set():
                if proc.poll() is None and not child_paused:
                    try:
                        os.kill(proc.pid, signal.SIGSTOP)
                        child_paused = True
                    except Exception as e:
                        print(f"[WARN] Could not SIGSTOP arecord ({device_str}): {e}")
                last = now
                time.sleep(0.05)
                continue
            else:
                # Ensure child is running
                if proc.poll() is None and child_paused:
                    try:
                        os.kill(proc.pid, signal.SIGCONT)
                        child_paused = False
                    except Exception as e:
                        print(f"[WARN] Could not SIGCONT arecord ({device_str}): {e}")

                # Count ACTIVE time
                active_elapsed += (now - last)
                last = now
                time.sleep(0.02)

        # Graceful finalize WAV
        if proc and proc.poll() is None:
            try:
                # If paused, resume first so it can flush header
                if child_paused:
                    os.kill(proc.pid, signal.SIGCONT)
                    child_paused = False
                    time.sleep(0.05)
                os.kill(proc.pid, signal.SIGINT)
            except Exception as e:
                print(f"[WARN] Could not SIGINT arecord ({device_str}): {e}")

        # Wait briefly; force kill if needed
        if proc:
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        print(f"[✅] Saved to {output_path}")

    except Exception as e:
        print(f"[ERROR] ❌ Recording failed for {device_str}: {e}")
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

# ── CLI main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Record from multiple physical mics using arecord (cooperative pause/resume).")
    parser.add_argument("output_dir", help="Directory to save WAV files")
    parser.add_argument("--duration", type=float, default=60.0, help="ACTIVE duration in seconds (pauses do not count)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not VALID_MIC_DEVICES:
        print("[ERROR] No VALID_MIC_IDS found in mic_config.py")
        return

    print(f"[INFO] 🎧 Starting parallel recording from {len(VALID_MIC_DEVICES)} mic(s) for {args.duration:.1f} sec ACTIVE...")
    threads = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dev in VALID_MIC_DEVICES:
        fname = f"mic_{dev.replace(':','').replace(',','')}_{timestamp}.wav"
        path = os.path.join(args.output_dir, fname)
        print(f"[INFO] 🎙 Recording from {dev} → {path}")
        t = threading.Thread(target=record_from_hw, args=(dev, path, args.duration), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all recorders
    for t in threads:
        t.join()

    print("[✅] All mic recordings completed.")

if __name__ == "__main__":
    main()
