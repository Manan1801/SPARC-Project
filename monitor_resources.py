#!/usr/bin/env python3

import psutil
import time
from datetime import datetime

def log_system_status(log_path, interval=5):
    with open(log_path, "w") as f:
        f.write("Timestamp,CPU (%),RAM (%),Disk Read (MB/s),Disk Write (MB/s)\n")
        prev_disk = psutil.disk_io_counters()

        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().percent
                curr_disk = psutil.disk_io_counters()
                read_speed = (curr_disk.read_bytes - prev_disk.read_bytes) / 1024 / 1024 / interval
                write_speed = (curr_disk.write_bytes - prev_disk.write_bytes) / 1024 / 1024 / interval
                prev_disk = curr_disk

                log_line = f"{timestamp},{cpu},{ram},{read_speed:.2f},{write_speed:.2f}"
                print(log_line)
                f.write(log_line + "\n")
                f.flush()
                time.sleep(interval - 1)
            except KeyboardInterrupt:
                print("\n[INFO] Stopping system monitoring.")
                break

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"system_monitor_{timestamp}.log"
    log_system_status(log_file)
