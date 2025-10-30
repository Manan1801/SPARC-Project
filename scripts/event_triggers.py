#!/usr/bin/env python3
"""
event_triggers.py

Central place for trigger checks, flagging, and logging while capture/processing runs.

CURRENTLY IMPLEMENTED TRIGGER
----------------------------
1) Right-hand wrist movement speed trigger:
   - current_avg_speed = cumulative_movement_upto_now / elapsed_time_s
   - Compare against [lower_bound, upper_bound] from a reference CSV with headers:
         time_s, lower_bound, upper_bound
     Bounds are picked using the latest row with time_s <= elapsed_time_s
     (falls back to first/last row when out-of-range).

   - If value < lower_bound  -> "LOW SPEED" (bad)
     If value > upper_bound  -> "HIGH SPEED" (bad)
     Else                    -> "GOOD" (no flag)

CADENCE / THROTTLING
--------------------
- To avoid log spam, we only raise/log at most once per "cadence window" slot:
      slot_index = floor(elapsed_time_s / SPEED_TRIGGER_CADENCE_WINDOW_S)
- Within a slot, we log the *first* bad status and show a PreviewGrid overlay.
- Good status is still returned but not logged.

LOGGING
-------
- Logs are centralized via logger_utils.DebouncedLogger.
- By default, a per-cam logger (cam_dir/logs/speed_trigger.txt) is created via
  logger_utils.get_speed_trigger_logger(cam_dir), unless a custom logger is injected.
- We DO NOT write files directly here; no duplicate writes.

PREVIEW GRID OVERLAY
--------------------
- If a PreviewGrid instance is provided, we attempt to call:
      preview_grid.annotate_event(cam_label, kind, text, ttl_s)
  If that method doesn't exist, we safely no-op.
"""

from __future__ import annotations
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from tunables import (
    SPEED_TRIGGER_CADENCE_WINDOW_S,
    SPEED_TRIGGER_REFCSV_PATH,
    SPEED_TRIGGER_OVERLAY_TTL_S,
)

# centralized logger
from logger_utils import DebouncedLogger, get_speed_trigger_logger

# --------------------------- Utilities ---------------------------

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _notify_preview_grid(preview_grid, cam_label: str, text: str, kind: str = "speed",
                         ttl_s: float = SPEED_TRIGGER_OVERLAY_TTL_S) -> None:
    if preview_grid is None:
        return
    try:
        if hasattr(preview_grid, "annotate_event"):
            preview_grid.annotate_event(cam_label, kind, text, ttl_s)
    except Exception:
        pass


# --------------------------- Reference CSV ---------------------------

@dataclass
class RefRow:
    time_s: float
    lo: float
    hi: float


class ReferenceBounds:
    def __init__(self, csv_path: Path):
        self.rows: List[RefRow] = []
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(f"[event_triggers] Reference CSV not found: {csv_path}")
        with open(csv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                t = float(r["time_s"])
                lo = float(r["lower_bound"])
                hi = float(r["upper_bound"])
                self.rows.append(RefRow(t, lo, hi))
        self.rows.sort(key=lambda r: r.time_s)
        if not self.rows:
            raise ValueError("[event_triggers] Reference CSV is empty.")

    def get_bounds(self, t: float) -> Tuple[float, float]:
        if t <= self.rows[0].time_s:
            return self.rows[0].lo, self.rows[0].hi
        if t >= self.rows[-1].time_s:
            return self.rows[-1].lo, self.rows[-1].hi
        last = self.rows[0]
        for row in self.rows[1:]:
            if row.time_s > t:
                return last.lo, last.hi
            last = row
        return self.rows[-1].lo, self.rows[-1].hi


# --------------------------- Trigger Base ---------------------------

class BaseTrigger:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass


# ---------------------- Right Wrist Speed Trigger -------------------

class RightWristSpeedTrigger(BaseTrigger):
    """
    Checks current average speed against time-indexed bounds.
    Throttles logging to one BAD event per cadence slot.

    update(...) returns a dict:
        {
          "good": bool,
          "reason": "good"|"low"|"high",
          "value": float,
          "lower": float,
          "upper": float,
          "slot": int,
          "elapsed_s": float,
        }
    """

    def __init__(self,
                 movement_cam_dir: Path,
                 cam_label: str,
                 reference_csv: Optional[Path] = None,
                 cadence_window_s: float = SPEED_TRIGGER_CADENCE_WINDOW_S,
                 overlay_ttl_s: float = SPEED_TRIGGER_OVERLAY_TTL_S,
                 logger: Optional[DebouncedLogger] = None):
        super().__init__("right_wrist_speed")
        self.cam_label = cam_label
        self.cadence_window_s = max(0.1, float(cadence_window_s))
        self.overlay_ttl_s = float(overlay_ttl_s)
        self.movement_cam_dir = Path(movement_cam_dir)

        ref_path = Path(reference_csv) if reference_csv else Path(SPEED_TRIGGER_REFCSV_PATH)
        self.ref = ReferenceBounds(ref_path)

        # Logging: use injected centralized logger if provided,
        # else create a per-cam speed-trigger logger via logger_utils.
        self.logger: DebouncedLogger = logger if logger is not None else get_speed_trigger_logger(self.movement_cam_dir)

        # internal cadence state
        self._last_bad_slot_logged: Optional[int] = None

    def reset(self):
        self._last_bad_slot_logged = None

    def _slot_index(self, elapsed_s: float) -> int:
        return int(math.floor(elapsed_s / self.cadence_window_s))

    def _judge(self, value: float, lo: float, hi: float) -> Tuple[bool, str]:
        if value < lo:
            return False, "low"
        if value > hi:
            return False, "high"
        return True, "good"

    def _format_line(self, *, ts_s: float, frame_idx: int, slot: int,
                     status: str, value: float, lo: float, hi: float) -> str:
        # E.g. "2025-10-30T23:59:12 ts=12.33 frame=370 slot=4 status=HIGH value=1.234 lo=0.800 hi=1.100 cam=cam2"
        return (f"{_iso_now()} ts={ts_s:.3f} frame={frame_idx} slot={slot} "
                f"status={status.upper()} value={value:.6f} lo={lo:.6f} hi={hi:.6f} cam={self.cam_label}")

    def update(self,
               *,
               elapsed_time_s: float,
               frame_idx: int,
               cumulative_movement: float,
               preview_grid=None) -> dict:
        # Guard for early frames
        if elapsed_time_s <= 0:
            return {
                "good": True,
                "reason": "good",
                "value": 0.0,
                "lower": float("nan"),
                "upper": float("nan"),
                "slot": self._slot_index(0.0),
                "elapsed_s": 0.0,
            }

        # 1) current average speed
        cur_avg_speed = float(cumulative_movement) / float(elapsed_time_s)

        # 2) bounds for this time
        lo, hi = self.ref.get_bounds(elapsed_time_s)

        # 3) judge
        good, reason = self._judge(cur_avg_speed, lo, hi)

        # 4) cadence slot
        slot = self._slot_index(elapsed_time_s)

        # 5) decide logging/overlay
        if not good:
            # Only one BAD log per slot
            if self._last_bad_slot_logged != slot:
                line = self._format_line(
                    ts_s=elapsed_time_s,
                    frame_idx=frame_idx,
                    slot=slot,
                    status=("LOW" if reason == "low" else "HIGH"),
                    value=cur_avg_speed,
                    lo=lo,
                    hi=hi,
                )
                try:
                    self.logger.info(line)
                    self.logger.periodic_flush()
                except Exception:
                    pass
                self._last_bad_slot_logged = slot

                # UI overlay: "Low Speed" or "High Speed" for few seconds
                overlay_text = "Low Speed" if reason == "low" else "High Speed"
                _notify_preview_grid(preview_grid, self.cam_label, overlay_text,
                                     kind="speed", ttl_s=self.overlay_ttl_s)

        return {
            "good": good,
            "reason": reason,           # "good" | "low" | "high"
            "value": cur_avg_speed,     # current average
            "lower": lo,
            "upper": hi,
            "slot": slot,
            "elapsed_s": elapsed_time_s,
        }


# --------------------------- Manager (future) ---------------------------

class EventManager:
    def __init__(self):
        self.triggers = []

    def add(self, trig: BaseTrigger):
        self.triggers.append(trig)

    def reset_all(self):
        for t in self.triggers:
            t.reset()
