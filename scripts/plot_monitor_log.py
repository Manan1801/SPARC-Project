#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("[USAGE] plot_monitor_log.py <system_monitor_*.txt>")
    sys.exit(1)

log_file = sys.argv[1]

# Read the CSV and clean headers
try:
    df = pd.read_csv(log_file)
    df.rename(columns={
        'Timestamp': 'timestamp',
        'CPU (%)': 'cpu',
        'RAM (%)': 'ram',
        'Disk Read (MB/s)': 'disk_read',
        'Disk Write (MB/s)': 'disk_write'
    }, inplace=True)
except Exception as e:
    print(f"[ERROR] Failed to read log file: {e}")
    sys.exit(1)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df['timestamp'], df['cpu'], label='CPU %', color='blue')
plt.ylabel("CPU (%)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(df['timestamp'], df['ram'], label='RAM %', color='orange')
plt.ylabel("RAM (%)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(df['timestamp'], df['disk_read'], label='Disk Read (MB/s)', color='green')
plt.plot(df['timestamp'], df['disk_write'], label='Disk Write (MB/s)', color='red')
plt.xlabel("Time")
plt.ylabel("Disk (MB/s)")
plt.legend()
plt.grid()

plt.suptitle("System Monitor Log")
plt.tight_layout()
plt.show()
