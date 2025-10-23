#!/usr/bin/env python3
import time
from pathlib import Path
from typing import Optional, List
import threading

class DebouncedLogger:
    def __init__(self, log_path: Path, flush_interval_sec: int = 5):
        self.log_path = Path(log_path)
        self._lines: List[str] = []
        self._last_msg: Optional[str] = None
        self._repeat: int = 0
        self._lock = threading.Lock()
        self._flush_interval = max(1, int(flush_interval_sec))
        self._last_flush = time.monotonic()

    def _flush_repeat(self):
        if self._last_msg is not None:
            if self._repeat > 1:
                self._lines.append(f"[WARN x{self._repeat}] {self._last_msg}")
            else:
                self._lines.append(f"[WARN] {self._last_msg}")
        self._last_msg = None
        self._repeat = 0

    def warn(self, msg: str):
        with self._lock:
            if msg == self._last_msg:
                self._repeat += 1
            else:
                self._flush_repeat()
                self._last_msg = msg
                self._repeat = 1

    def info(self, msg: str):
        with self._lock:
            self._flush_repeat()
            self._lines.append(f"[INFO] {msg}")

    def periodic_flush(self, force: bool = False):
        now = time.monotonic()
        with self._lock:
            if force or (now - self._last_flush) >= self._flush_interval:
                self._flush_repeat()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                if self._lines:
                    with open(self.log_path, "a") as f:
                        f.write("\n".join(self._lines) + "\n")
                    self._lines.clear()
                self._last_flush = now
