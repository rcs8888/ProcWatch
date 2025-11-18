"""
process_collector.py
--------------------
This script performs live process monitoring on the host system during an experiment.
It continuously samples process information (PID, parent PID, CPU%, memory%, executable path)
and saves it to a session-specific log file (process_stream.csv).

Each run generates or appends to a session directory (e.g., logs/test1/) to keep data organized.
The collected data is later used by prepare_dataset.py for labeling and feature extraction.

Usage:
    python3 process_collector.py --session test1

Press Ctrl+C to stop data collection when finished.

Author: Rachel Catherine Soubier
University of North Carolina Wilmington
Date: November 2025
"""

import psutil
import csv
import time
import argparse
import os
from datetime import datetime

def collect_process_data(session_id, interval=1):
    """
    Collects live process metrics periodically and appends them to a CSV file.
    Saves under logs/<session_id>/process_stream.csv
    """
    # Create logs/<session_id> directory if it doesn't exist
    session_dir = os.path.join("logs", session_id)
    os.makedirs(session_dir, exist_ok=True)

    outfile = os.path.join(session_dir, "process_stream.csv")
    fieldnames = ["timestamp", "session_id", "pid", "ppid", "cpu", "mem", "exe"]

    # If file doesn't exist, write header
    file_exists = os.path.exists(outfile)
    with open(outfile, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print(f"[+] Starting process monitoring session '{session_id}'...")
        print(f"[+] Writing data to: {outfile}")
        print(f"[i] Press Ctrl+C to stop collection.\n")

        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for proc in psutil.process_iter(['pid', 'ppid', 'cpu_percent', 'memory_percent', 'exe']):
                    try:
                        info = proc.info
                        writer.writerow({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "pid": info.get("pid"),
                            "ppid": info.get("ppid"),
                            "cpu": info.get("cpu_percent", 0.0),
                            "mem": info.get("memory_percent", 0.0),
                            "exe": info.get("exe", "")
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                csvfile.flush()
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[!] Stopping process collection.")
            print(f"[+] Session '{session_id}' complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Collect live process metrics for ML dataset")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--session", type=str, default=None, help="Optional session ID for labeling")
    args = parser.parse_args()

    # Default session name if none provided
    if not args.session:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.session = f"session_{timestamp}"

    collect_process_data(args.session, interval=args.interval)

if __name__ == "__main__":
    main()
