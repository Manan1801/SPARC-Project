#!/usr/bin/env python3

import rospy
from diagnostic_msgs.msg import DiagnosticArray
from datetime import datetime
import os
import sys

log_files = {}

def callback(msg, camera_ns):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for status in msg.status:
        if "Dropped" in status.name or "frames" in status.name.lower():
            log_entry = f"{now} | {status.name} | {status.message}"
            print(f"[{camera_ns}] {log_entry}")
            log_files[camera_ns].write(log_entry + "\n")
            log_files[camera_ns].flush()

def diagnostics_logger(log_dir):
    rospy.init_node("realsense_diagnostics_logger")
    cams = ["cam1", "cam2", "cam3", "cam4"]
    os.makedirs(log_dir, exist_ok=True)

    for cam in cams:
        topic = f"/{cam}/diagnostics"
        log_path = os.path.join(log_dir, f"{cam}_diagnostics.log")
        log_files[cam] = open(log_path, "w")
        rospy.Subscriber(topic, DiagnosticArray, callback, callback_args=cam)
        print(f"[INFO] Subscribed to {topic} → Logging to {log_path}")

    rospy.spin()

    for f in log_files.values():
        f.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide log directory path as an argument.")
        sys.exit(1)

    diagnostics_logger(sys.argv[1])
