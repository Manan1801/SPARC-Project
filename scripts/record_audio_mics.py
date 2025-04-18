#!/usr/bin/env python3

import os
import argparse
import subprocess
from datetime import datetime
import threading

# These are your 2 real USB microphones, based on `arecord -l`:
# card 2, device 0  → hw:2,0
# card 3, device 0  → hw:3,0
VALID_HW_IDS = ["hw:2,0", "hw:3,0"]

def record_with_arecord(hw_id, duration_sec, output_dir):
    label = hw_id.replace(":", "").replace(",", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"mic_{label}_{timestamp}.wav")

    cmd = [
        "arecord",
        "-D", hw_id,
        "-c", "1",              # 1 channel (mono)
        "-f", "S16_LE",         # 16-bit little endian
        "-r", "44100",          # 44100 Hz
        "-t", "wav",
        "-d", str(duration_sec),
        filename
    ]

    print(f"[INFO] 🎙 Recording from {hw_id} → {filename}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[✅] Saved to {filename}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ❌ Mic {hw_id} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Record audio using arecord from valid mic hw devices.")
    parser.add_argument("output_dir", help="Directory to save recordings")
    parser.add_argument("--duration", type=float, default=1.0, help="Recording duration in minutes (default: 1)")
    args = parser.parse_args()

    duration_sec = int(args.duration * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] 🎧 Starting parallel recording from {len(VALID_HW_IDS)} mic(s) for {args.duration:.1f} min...")

    threads = []
    for hw in VALID_HW_IDS:
        t = threading.Thread(target=record_with_arecord, args=(hw, duration_sec, args.output_dir), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("[✅] All mic recordings completed.")

if __name__ == "__main__":
    main()
