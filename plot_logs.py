#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import re

def plot_system_logs(log_path):
    df = pd.read_csv(log_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    ax[0].plot(df['Timestamp'], df['CPU (%)'], label='CPU (%)')
    ax[1].plot(df['Timestamp'], df['RAM (%)'], label='RAM (%)', color='orange')
    ax[2].plot(df['Timestamp'], df['Disk Write (MB/s)'], label='Disk Write MB/s', color='green')

    for a in ax:
        a.legend()
        a.grid(True)

    plt.xlabel("Time")
    plt.suptitle("System Resource Usage")
    plt.tight_layout()
    plt.show()

def plot_dropped_frames(diagnostics_dir):
    dropped_counts = {}
    for file in glob.glob(os.path.join(diagnostics_dir, "*_diagnostics.log")):
        cam = os.path.basename(file).split("_")[0]  # cam1, cam2, etc.
        with open(file, "r") as f:
            times = []
            drops = []
            for line in f:
                match = re.match(r"(.+?) \\| .*? \\| .*?(\\d+) frames", line)
                if match:
                    times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
                    drops.append(int(match.group(2)))
            if times:
                dropped_counts[cam] = (times, drops)

    if not dropped_counts:
        print("[WARN] No diagnostics data found.")
        return

    plt.figure(figsize=(12, 6))
    for cam, (times, drops) in dropped_counts.items():
        plt.plot(times, drops, label=cam)

    plt.xlabel("Time")
    plt.ylabel("Dropped Frames")
    plt.title("Dropped Frames per Camera Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    system_log = input("Enter path to system_monitor_<timestamp>.log: ").strip()
    if os.path.exists(system_log):
        plot_system_logs(system_log)
    else:
        print(f"[ERROR] File not found: {system_log}")

    diag_dir = input("Enter path to directory containing *_diagnostics.log files: ").strip()
    if os.path.exists(diag_dir):
        plot_dropped_frames(diag_dir)
    else:
        print(f"[ERROR] Directory not found: {diag_dir}")
