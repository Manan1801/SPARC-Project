#!/usr/bin/env python3
"""
event_triggers.py

Central place for trigger checks, flagging, and logging while capture/processing runs.
(Existing RightWristSpeedTrigger kept; ObjectUntouchedTrigger updated per spec.)
"""

from __future__ import annotations
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from tunables import (
    SPEED_TRIGGER_CADENCE_WINDOW_S,
    SPEED_TRIGGER_REFCSV_PATH,
    SPEED_TRIGGER_OVERLAY_TTL_S,  # still used by speed trigger
    OBJECT_TRIGGER_WINDOW_SEC,     # ← window length used for object trigger
)

from logger_utils import (
    DebouncedLogger,
    get_speed_trigger_logger,
    get_object_trigger_logger,  # logs to <cam>/logs/object_trigger
)

# --------------------------- Utilities ---------------------------

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

# (speed trigger still uses preview overlays via annotate_event)
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


# ---------------------- Right Wrist Speed Trigger (unchanged) -------------------

class RightWristSpeedTrigger(BaseTrigger):
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

        self.logger: DebouncedLogger = logger if logger is not None else get_speed_trigger_logger(self.movement_cam_dir)
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
        return (f"{_iso_now()} ts={ts_s:.3f} frame={frame_idx} slot={slot} "
                f"status={status.upper()} value={value:.6f} lo={lo:.6f} hi={hi:.6f} cam={self.cam_label}")

    def update(self,
               *,
               elapsed_time_s: float,
               frame_idx: int,
               cumulative_movement: float,
               preview_grid=None) -> dict:
        if elapsed_time_s <= 0:
            return {"good": True, "reason": "good", "value": 0.0,
                    "lower": float("nan"), "upper": float("nan"),
                    "slot": self._slot_index(0.0), "elapsed_s": 0.0}

        cur_avg_speed = float(cumulative_movement) / float(elapsed_time_s)
        lo, hi = self.ref.get_bounds(elapsed_time_s)
        good, reason = self._judge(cur_avg_speed, lo, hi)
        slot = self._slot_index(elapsed_time_s)

        if not good and self._last_bad_slot_logged != slot:
            line = self._format_line(ts_s=elapsed_time_s, frame_idx=frame_idx, slot=slot,
                                     status=("LOW" if reason == "low" else "HIGH"),
                                     value=cur_avg_speed, lo=lo, hi=hi)
            try:
                self.logger.info(line); self.logger.periodic_flush()
            except Exception:
                pass
            self._last_bad_slot_logged = slot
            _notify_preview_grid(preview_grid, self.cam_label,
                                 "Low Speed" if reason == "low" else "High Speed",
                                 kind="speed")

        return {"good": good, "reason": reason, "value": cur_avg_speed,
                "lower": lo, "upper": hi, "slot": slot, "elapsed_s": elapsed_time_s}


# ---------------------- Object Untouched Trigger (UPDATED) -------------------

class ObjectUntouchedTrigger(BaseTrigger):
    """
    GOOD iff within the current cadence window (length W = OBJECT_TRIGGER_WINDOW_SEC),
    the number of objects whose untouched coverage ≥ 90% of W is in {2, 3}.
    Otherwise BAD.

    Expects confirmed untouched intervals as:
        untouched_out = { obj_name: [[start_f, end_f], ...], ... }

    update(...) returns:
        {
          "good": bool,
          "qualified_count": int,
          "per_object_sec": {obj: seconds_in_window},
          "per_object_pct": {obj: percentage_of_window},
          "per_object_state": {obj: 'untouched'|'other'},  # at window end
          "slot": int,
          "window_start_s": float,
          "window_end_s": float,
        }
    """
    def __init__(self,
                 cam_dir: Path,
                 cam_label: Optional[str] = None,                 # ← now optional for compatibility
                 cadence_window_s: float = OBJECT_TRIGGER_WINDOW_SEC,
                 logger: Optional[DebouncedLogger] = None):
        super().__init__("objects_untouched")
        self.cam_label = cam_label or "(unknown)"
        self.cam_dir = Path(cam_dir)
        self.W = float(max(0.5, cadence_window_s))
        # Threshold is always 90% of the window (per spec)
        self.tol_thresh = 0.9 * self.W
        self.logger: DebouncedLogger = logger if logger is not None else get_object_trigger_logger(self.cam_dir)
        self._last_slot_logged: Optional[int] = None

    def reset(self):
        self._last_slot_logged = None

    def _slot_index(self, elapsed_s: float) -> int:
        return int(math.floor(elapsed_time_s / self.W))

    def _slot_bounds(self, slot_idx: int) -> Tuple[float, float]:
        start_s = slot_idx * self.W
        end_s = (slot_idx + 1) * self.W
        return start_s, end_s

    @staticmethod
    def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
        # inclusive frame intervals [a0,a1], [b0,b1]
        lo = max(a0, b0)
        hi = min(a1, b1)
        return max(0, hi - lo + 1)

    @staticmethod
    def _contains_frame(spans: List[List[int]], f: int) -> bool:
        for s, e in spans:
            if s <= f <= e:
                return True
        return False

    def update(self,
               *,
               elapsed_time_s: float,
               frame_idx: int,
               fps: float,
               untouched_out: Dict[str, List[List[int]]],
               preview_grid=None) -> dict:
        # NOTE: object trigger does NOT pop any preview overlay (per spec).
        slot = self._slot_index(elapsed_time_s)
        win_s, win_e = self._slot_bounds(slot)
        win_f0 = int(math.floor(win_s * fps))
        win_f1 = int(math.floor(win_e * fps)) - 1  # inclusive end frame
        window_len_s = self.W

        per_obj_sec: Dict[str, float] = {}
        per_obj_pct: Dict[str, float] = {}
        per_obj_state: Dict[str, str] = {}

        qualified = 0

        for obj, spans in untouched_out.items():
            # 1) Coverage (frames→seconds) inside this window
            total_frames = 0
            for s, e in spans:
                total_frames += self._overlap_len(s, e, win_f0, win_f1)
            sec = total_frames / float(max(1.0, fps))
            per_obj_sec[obj] = sec
            pct = (sec / window_len_s) * 100.0 if window_len_s > 0 else 0.0
            per_obj_pct[obj] = pct

            # 2) State at the window end
            end_state = "untouched" if self._contains_frame(spans, win_f1) else "other"
            per_obj_state[obj] = end_state

            # 3) Qualification (≥ 90% of window)
            if sec >= self.tol_thresh:
                qualified += 1

        good = (2 <= qualified <= 3)

        # Log once per slot with full per-object breakdown
        if self._last_slot_logged != slot:
            status = "GOOD" if good else "BAD"
            obj_chunks = []
            for obj in sorted(per_obj_sec.keys()):
                obj_chunks.append(
                    f"{obj}: sec={per_obj_sec[obj]:.2f} pct={per_obj_pct[obj]:.1f}% state={per_obj_state[obj]}"
                )
            obj_str = " | ".join(obj_chunks)
            line = (f"{_iso_now()} t={elapsed_time_s:.3f} slot={slot} "
                    f"win=[{win_s:.1f},{win_e:.1f}) signal={status} qualified={qualified} "
                    f"tol90={self.tol_thresh:.2f}s cam={self.cam_label} || {obj_str}")
            try:
                self.logger.info(line); self.logger.periodic_flush()
            except Exception:
                pass
            self._last_slot_logged = slot

        return {
            "good": good,
            "qualified_count": qualified,
            "per_object_sec": per_obj_sec,
            "per_object_pct": per_obj_pct,
            "per_object_state": per_obj_state,
            "slot": slot,
            "window_start_s": win_s,
            "window_end_s": win_e,
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
