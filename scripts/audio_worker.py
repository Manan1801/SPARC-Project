#!/usr/bin/env python3
import time
from pathlib import Path
import subprocess

from tunables import VALID_MIC_IDS  

def audio_worker(device_str: str, out_dir: Path, duration_sec: float, rate: int):
    """
    Record from a single whitelisted ALSA device (e.g., 'hw:0,0') for duration_sec seconds.
    No auto-detection or fallback. If not in whitelist, skip.
    """
    if device_str not in VALID_MIC_IDS:
        print(f"[WARN] audio_worker: '{device_str}' not in list → skipping.")
        return
    if duration_sec <= 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = device_str.replace(":", "").replace(",", "")
    wav_tmp = out_dir / f"mic_{safe_name}_{ts}.part"
    wav_final = Path(str(wav_tmp).replace(".part", ".wav"))

    cmd = ["arecord", "-D", device_str, "-f", "cd", "-c", "1", "-r", str(rate), "-t", "wav", str(wav_tmp)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[ERROR] audio({device_str}): failed to start arecord: {e}")
        return

    start = time.time()
    try:
        while (time.time() - start) < duration_sec:
            time.sleep(0.05)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if wav_tmp.exists():
            wav_tmp.replace(wav_final)
            print(f"[✅] Audio saved: {wav_final}")
        else:
            print(f"[WARN] audio({device_str}): no output file produced.")
    except Exception as e:
        print(f"[ERROR] audio({device_str}): {e}")
        try:
            proc.kill()
        except Exception:
            pass
