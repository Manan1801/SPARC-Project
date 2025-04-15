#!/usr/bin/env python3

import subprocess
import os
import sys
from datetime import datetime

def record_per_camera(cams, topics_dict, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []

    for cam in cams:
        topics = topics_dict.get(cam, [])
        topic_str = " ".join([f"/{cam}{t}" for t in topics])
        output_file = os.path.join(output_dir, f"{cam}_{timestamp}.bag")
        cmd = f"rosbag record -O {output_file} {topic_str}"
        print(f"[INFO] Starting rosbag for {cam} → {output_file}")
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    return processes

if __name__ == "__main__":
    import signal

    if len(sys.argv) < 2:
        print("[ERROR] Please provide output directory path as an argument.")
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    cams = ["cam1", "cam2", "cam3", "cam4"]
    base_topics = [
        "/camera/color/image_raw",
        "/camera/depth/image_rect_raw",
        "/camera/color/camera_info",
        "/camera/depth/camera_info",
        "/diagnostics"
    ]
    topics_dict = {cam: base_topics for cam in cams}

    processes = record_per_camera(cams, topics_dict, output_dir)

    try:
        input("[INFO] Press ENTER to stop rosbag recordings...\n")
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.send_signal(signal.SIGINT)
        print("[INFO] Recording stopped.")
