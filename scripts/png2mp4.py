#!/usr/bin/env python3
"""
Convert PNG color frames to MP4 with lab-specific folder logic.

USAGE EXAMPLES:
  # Single sequence (color)
  python png_to_mp4.py --input /path/to/01/cam2/color --fps 30

  # Single sequence (color_mp)
  python png_to_mp4.py --input /path/to/01/cam2/color_mp --fps 30

  # Single cam*/ folder (ONLY process its color/ and/or color_mp/)
  python png_to_mp4.py --input /path/to/01/cam2 --fps 30

  # Any folder that directly contains multiple cam*/ folders (includes numbered sessions)
  python png_to_mp4.py --input /path/to/01 --fps 30

Notes:
- Output path is determined automatically:
    cam*/color     -> cam*/video/cam*.mp4
    cam*/color_mp  -> cam*/video/cam*_mp.mp4
- Parent folders for outputs are created if needed.
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Convert PNG frames to MP4 (color/color_mp) with auto output paths")
    p.add_argument("--input", required=True,
                   help="Path to cam*/color/, cam*/color_mp/, a single cam*/ folder, or a parent containing cam*/ folders")
    p.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    return p.parse_args()


def find_png_frames(folder: Path):
    return sorted(folder.glob("*.png"))


def write_video_from_frames(frames, out_path: Path, fps: int):
    if not frames:
        print(f"[WARN] No .png frames found in {out_path.parent.parent}/(color|color_mp)")
        return False

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"[WARN] First frame unreadable: {frames[0]}")
        return False

    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    wrote_any = False
    for fp in tqdm(frames, desc=f"Writing {out_path.name}", unit="frame"):
        img = cv2.imread(str(fp))
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {fp}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        wrote_any = True

    writer.release()
    if wrote_any:
        print(f"✅ Saved: {out_path}")
    else:
        print(f"[WARN] No frames written for {out_path}")
    return wrote_any


def process_color_folder(color_dir: Path, fps: int):
    cam_dir = color_dir.parent                # cam*
    cam_name = cam_dir.name                   # e.g., cam2
    out_path = cam_dir / "video" / f"{cam_name}.mp4"
    frames = find_png_frames(color_dir)
    print(f"[INFO] Found {len(frames)} frames in {color_dir}")
    write_video_from_frames(frames, out_path, fps)


def process_color_mp_folder(color_mp_dir: Path, fps: int):
    cam_dir = color_mp_dir.parent
    cam_name = cam_dir.name
    out_path = cam_dir / "video" / f"{cam_name}_mp.mp4"
    frames = find_png_frames(color_mp_dir)
    print(f"[INFO] Found {len(frames)} frames in {color_mp_dir}")
    write_video_from_frames(frames, out_path, fps)


def list_cam_dirs(parent: Path):
    return [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("cam")]


def process_all_under_camroot(camroot: Path, fps: int):
    """Process every cam*/color and cam*/color_mp directly under camroot."""
    cam_dirs = list_cam_dirs(camroot)
    if not cam_dirs:
        print(f"[WARN] No 'cam*' folders found in {camroot}")
        return
    for cam in sorted(cam_dirs):
        color_dir = cam / "color"
        color_mp_dir = cam / "color_mp"
        if color_dir.is_dir():
            process_color_folder(color_dir, fps)
        else:
            print(f"[INFO] Skipping (no color/): {cam}")
        if color_mp_dir.is_dir():
            process_color_mp_folder(color_mp_dir, fps)
        else:
            print(f"[INFO] Skipping (no color_mp/): {cam}")


def main():
    args = parse_args()
    in_path = Path(args.input).resolve()

    if not in_path.exists():
        print(f"[ERROR] Input path does not exist: {in_path}")
        return

    # 1) Direct color / color_mp inside a single cam*/
    if in_path.is_dir() and in_path.name in ("color", "color_mp") and in_path.parent.name.startswith("cam"):
        if in_path.name == "color":
            process_color_folder(in_path, args.fps)
        else:
            process_color_mp_folder(in_path, args.fps)
        return

    # 2) Single cam*/ folder — process ONLY its own color/ and/or color_mp/
    if in_path.is_dir() and in_path.name.startswith("cam"):
        processed = False
        if (in_path / "color").is_dir():
            process_color_folder(in_path / "color", args.fps)
            processed = True
        if (in_path / "color_mp").is_dir():
            process_color_mp_folder(in_path / "color_mp", args.fps)
            processed = True
        if not processed:
            print(f"[WARN] {in_path} has no color/ or color_mp/ subfolders.")
        return

    # 3) Any parent folder that directly contains cam*/ folders (includes numbered session folders)
    if in_path.is_dir() and list_cam_dirs(in_path):
        process_all_under_camroot(in_path, args.fps)
        return

    print(f"[ERROR] Unsupported input path form. Provide one of:\n"
          f"  - .../cam*/color/\n"
          f"  - .../cam*/color_mp/\n"
          f"  - .../cam*/\n"
          f"  - .../<parent_with_cam_subfolders>/")

if __name__ == "__main__":
    main()
