#!/usr/bin/env python3
# audio_worker.py — optional centralized logging
import time
from pathlib import Path
import subprocess

from tunables import VALID_MIC_IDS
from logger_utils import DebouncedLogger

def audio_worker(device_str: str, out_dir: Path, duration_sec: float, rate: int, logger: DebouncedLogger | None = None):
    """
    Record from a single whitelisted ALSA device (e.g., 'hw:0,0') for duration_sec seconds.
    No auto-detection or fallback. If not in whitelist, skip.
    """
    # Logging is optional; if not provided, be quiet except for final success line.
    def _log_info(msg: str):
        if logger: logger.info(msg)
    def _log_warn(msg: str):
        if logger: logger.warn(msg)
    def _flush(force: bool = False):
        if logger: logger.periodic_flush(force=force)

    if device_str not in VALID_MIC_IDS:
        _log_warn(f"audio_worker: '{device_str}' not in whitelist → skipping.")
        _flush()
        return
    if duration_sec <= 0:
        _log_warn("audio_worker: duration_sec <= 0 → skipping.")
        _flush()
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = device_str.replace(":", "").replace(",", "")
    wav_tmp = out_dir / f"mic_{safe_name}_{ts}.part"
    wav_final = Path(str(wav_tmp).replace(".part", ".wav"))

    cmd = ["arecord", "-D", device_str, "-f", "cd", "-c", "1", "-r", str(rate), "-t", "wav", str(wav_tmp)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log_info(f"Audio started on {device_str} → {wav_tmp.name}")
    except Exception as e:
        _log_warn(f"audio({device_str}): failed to start arecord: {e}")
        _flush()
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
            print(f"[✅] Audio saved: {wav_final}")  # keep one visible success line
            _log_info(f"Audio saved: {wav_final.name}")
        else:
            _log_warn(f"audio({device_str}): no output file produced.")
    except Exception as e:
        _log_warn(f"audio({device_str}): {e}")
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _flush(force=True)
