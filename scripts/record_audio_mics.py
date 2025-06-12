#!/usr/bin/env python3

import os
import argparse
import subprocess
from datetime import datetime
import threading
from mic_config import VALID_MIC_IDS  # Import your mic config

# Devices: USB Audio interfaces
# VALID_MIC_DEVICES = ["hw:3,0", "hw:1,0"]

VALID_MIC_DEVICES = VALID_MIC_IDS  # Use the mic IDs defined in mic_config.py

def record_from_hw(device_str, output_path, duration_sec):
    cmd = [
        "arecord",
        "-D", device_str,
        "-f", "cd",
        "-c", "1",
        "-r", "44100",
        "-t", "wav",
        "-d", str(int(duration_sec)),
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ❌ Failed recording {device_str}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Record from multiple physical mics using arecord.")
    parser.add_argument("output_dir", help="Directory to save WAV files")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (not minutes)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] 🎧 Starting parallel recording from {len(VALID_MIC_DEVICES)} mic(s) for {args.duration:.1f} sec...")

    threads = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dev in VALID_MIC_DEVICES:
        fname = f"mic_{dev.replace(':','').replace(',','')}_{timestamp}.wav"
        path = os.path.join(args.output_dir, fname)
        print(f"[INFO] 🎙 Recording from {dev} → {path}")
        t = threading.Thread(target=record_from_hw, args=(dev, path, args.duration), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
        print(f"[✅] Saved to {path}")

    print("[✅] All mic recordings completed.")

if __name__ == "__main__":
    main()
