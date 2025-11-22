#!/usr/bin/env python3
"""
process_collector.py (rewritten)
--------------------------------
Collects live process metrics and writes them to logs/<session>/process_stream.csv

CSV columns:
  timestamp (UTC ISO with offset), session_id, pid, ppid, cpu, mem, exe

Usage:
    python3 process_collector.py --session test1 --interval 1.0 --duration 300

Notes:
 - cpu is psutil's per-process cpu percent (the value is relative to the last call).
 - mem is psutil's memory_percent (percentage of RAM).
 - Timestamps are written in UTC ISO format (e.g. 2025-11-21T12:34:56.123456+00:00).
"""

import os
import csv
import time
import argparse
from datetime import datetime, timezone
import psutil

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

def collect_process_data(session_id: str, interval: float = 1.0, duration: float | None = None):
    """
    Collects live process metrics periodically and appends them to:
        logs/<session_id>/process_stream.csv

    interval: seconds between samples (float)
    duration: total seconds to run (None => run until Ctrl+C)
    """
    session_dir = os.path.join("logs", session_id)
    ensure_dir(session_dir)
    outfile = os.path.join(session_dir, "process_stream.csv")

    fieldnames = ["timestamp", "session_id", "pid", "ppid", "cpu", "mem", "exe"]

    file_exists = os.path.exists(outfile)
    # open in append mode and keep the file handle open for performance
    with open(outfile, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print(f"[+] Starting process monitoring session '{session_id}'")
        print(f"[+] Writing to: {outfile}")
        print(f"[i] Sampling every {interval}s. Press Ctrl+C to stop.\n")

        # To get meaningful cpu_percent() values we call cpu_percent once for each process
        # before entering the timed loop; subsequent calls return percent since last call.
        # We tolerate that the very first sample may be low/zero for many processes.
        try:
            # Make an initial pass to prime cpu_percent() readings
            procs = list(psutil.process_iter(['pid']))
            for proc in procs:
                try:
                    proc.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            start_ts = time.time()
            while True:
                loop_ts = time.time()
                # Stop if duration reached
                if duration is not None and (loop_ts - start_ts) >= duration:
                    print("[i] Duration reached — stopping collection.")
                    break

                timestamp = iso_utc_now()

                # Iterate processes snapshot
                for proc in psutil.process_iter(['pid', 'ppid', 'exe']):
                    try:
                        info = proc.info
                        pid = info.get('pid')
                        ppid = info.get('ppid')
                        exe = info.get('exe') or ""

                        # cpu_percent(None) returns percent since last call to this method for the process
                        cpu = proc.cpu_percent(None)
                        mem = proc.memory_percent()

                        writer.writerow({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "pid": pid,
                            "ppid": ppid,
                            "cpu": f"{cpu:.4f}",
                            "mem": f"{mem:.6f}",
                            "exe": exe
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # process disappeared or denied — skip
                        continue
                    except Exception as e:
                        # unexpected error for a row should not crash collector
                        # print a short message and continue
                        print(f"[!] Row write error for pid={getattr(proc, 'pid', 'N/A')}: {e}")
                        continue

                # flush so that other tools reading the file see progress
                csvfile.flush()
                # sleep until next sample (account for time spent gathering)
                elapsed = time.time() - loop_ts
                to_sleep = max(0.0, interval - elapsed)
                time.sleep(to_sleep)

        except KeyboardInterrupt:
            print("\n[!] Stopping process collection (user requested).")
        except Exception as e:
            # Log unexpected fatal error then re-raise
            print(f"\n[!] Fatal error in collector: {e}")
            raise
        finally:
            print(f"[+] Session '{session_id}' finished. Data written to {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Collect live process metrics for ML dataset")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--session", type=str, default=None, help="Session ID for logs (default: autogenerated)")
    parser.add_argument("--duration", type=float, default=None, help="Total duration to run in seconds (optional)")
    args = parser.parse_args()

    # default session name if none provided
    if not args.session:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.session = f"session_{ts}"

    collect_process_data(args.session, interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()


