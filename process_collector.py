#!/usr/bin/env python3
"""
process_collector.py
Enhanced process collector with comprehensive metrics for anomaly detection.

Improvements:
- I/O statistics (read/write bytes and counts)
- Network connections count
- Thread count
- File descriptors
- Context switches
- Process status
- Microsecond timestamps for better temporal alignment
- Configurable sampling rate (default 0.5s for better temporal resolution)

Author: Enhanced by Assistant
Date: 2025-11
"""

import os
import csv
import time
import argparse
from datetime import datetime, timezone
import psutil
import math
import platform


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def iso_utc_now_precise():
    """Return ISO timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def shannon_entropy(items):
    """Compute entropy of a list of strings."""
    if not items:
        return 0.0
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    total = len(items)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def get_process_io(proc):
    """Safely get I/O statistics."""
    try:
        io = proc.io_counters()
        return {
            'io_read_bytes': io.read_bytes,
            'io_write_bytes': io.write_bytes,
            'io_read_count': io.read_count,
            'io_write_count': io.write_count
        }
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return {
            'io_read_bytes': 0,
            'io_write_bytes': 0,
            'io_read_count': 0,
            'io_write_count': 0
        }


def get_network_connections(proc):
    """Safely get network connection count."""
    try:
        conns = proc.connections()
        return len(conns)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


def get_file_descriptors(proc):
    """Safely get file descriptor count (Linux/Mac only)."""
    try:
        if hasattr(proc, 'num_fds'):
            return proc.num_fds()
        else:
            # Fallback: count open files
            return len(proc.open_files())
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return 0


def get_context_switches(proc):
    """Safely get context switch counts."""
    try:
        ctx = proc.num_ctx_switches()
        return {
            'ctx_switches_voluntary': ctx.voluntary,
            'ctx_switches_involuntary': ctx.involuntary
        }
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return {
            'ctx_switches_voluntary': 0,
            'ctx_switches_involuntary': 0
        }


def get_thread_count(proc):
    """Safely get thread count."""
    try:
        return proc.num_threads()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


def get_process_status(proc):
    """Safely get process status."""
    try:
        return proc.status()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 'unknown'


def collect_process_data(session_id, interval=0.5, duration=None):
    """
    Collect comprehensive process metrics.
    
    Args:
        session_id: Session identifier
        interval: Sampling interval in seconds (default 0.5 for better temporal resolution)
        duration: Total collection duration in seconds (None = infinite)
    """
    session_dir = os.path.join("logs", session_id)
    ensure_dir(session_dir)

    outfile = os.path.join(session_dir, "process_stream.csv")

    # COMPREHENSIVE field schema
    fieldnames = [
        # Basic identifiers
        "timestamp",
        "session_id",
        "pid",
        "ppid",
        
        # CPU and Memory
        "cpu",
        "mem",
        "exe",
        
        # Process relationships
        "num_children",
        "child_entropy",
        
        # I/O Statistics (NEW)
        "io_read_bytes",
        "io_write_bytes",
        "io_read_count",
        "io_write_count",
        
        # Network (NEW)
        "num_connections",
        
        # Threading (NEW)
        "num_threads",
        
        # File descriptors (NEW)
        "num_fds",
        
        # Context switches (NEW)
        "ctx_switches_voluntary",
        "ctx_switches_involuntary",
        
        # Process status (NEW)
        "status"
    ]

    file_exists = os.path.exists(outfile)

    with open(outfile, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print(f"[+] Starting ENHANCED process monitoring session '{session_id}'")
        print(f"[+] Writing to {outfile}")
        print(f"[i] Sampling interval = {interval}s (faster = better temporal resolution)")
        print(f"[i] Platform: {platform.system()}")

        # Prime CPU usage to avoid initial 0.0 values
        print("[i] Priming CPU measurements...")
        for proc in psutil.process_iter(['pid']):
            try:
                proc.cpu_percent(None)
            except:
                pass
        time.sleep(0.1)  # Short wait after priming

        start_ts = time.time()
        sample_count = 0

        try:
            while True:
                loop_start = time.time()
                
                # Check duration
                if duration and (loop_start - start_ts) >= duration:
                    print("[i] Duration reached — stopping.")
                    break

                timestamp = iso_utc_now_precise()

                # Snapshot all processes
                all_procs = list(psutil.process_iter(['pid', 'ppid', 'exe']))

                # Build parent→children map
                child_map = {}
                for p in all_procs:
                    try:
                        pid = p.info['pid']
                        ppid = p.info.get('ppid')
                        child_map.setdefault(ppid, []).append(p)
                    except:
                        continue

                # Collect metrics for each process
                for proc in all_procs:
                    try:
                        info = proc.info
                        pid = info["pid"]
                        ppid = info.get("ppid")
                        exe = info.get("exe") or ""

                        # CPU and Memory
                        cpu = proc.cpu_percent(None)
                        mem = proc.memory_percent()

                        # Children
                        children = child_map.get(pid, [])
                        num_children = len(children)
                        child_exe_names = [(c.info.get("exe") or "") for c in children]
                        child_entropy = shannon_entropy(child_exe_names)

                        # I/O Statistics
                        io_stats = get_process_io(proc)
                        
                        # Network
                        num_connections = get_network_connections(proc)
                        
                        # Threads
                        num_threads = get_thread_count(proc)
                        
                        # File descriptors
                        num_fds = get_file_descriptors(proc)
                        
                        # Context switches
                        ctx_stats = get_context_switches(proc)
                        
                        # Status
                        status = get_process_status(proc)

                        # Write row
                        writer.writerow({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "pid": pid,
                            "ppid": ppid,
                            "cpu": f"{cpu:.4f}",
                            "mem": f"{mem:.6f}",
                            "exe": exe,
                            "num_children": num_children,
                            "child_entropy": f"{child_entropy:.6f}",
                            "io_read_bytes": io_stats['io_read_bytes'],
                            "io_write_bytes": io_stats['io_write_bytes'],
                            "io_read_count": io_stats['io_read_count'],
                            "io_write_count": io_stats['io_write_count'],
                            "num_connections": num_connections,
                            "num_threads": num_threads,
                            "num_fds": num_fds,
                            "ctx_switches_voluntary": ctx_stats['ctx_switches_voluntary'],
                            "ctx_switches_involuntary": ctx_stats['ctx_switches_involuntary'],
                            "status": status
                        })

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    except Exception as e:
                        # Silently skip problematic processes
                        continue

                csvfile.flush()
                sample_count += 1
                
                # Progress indicator every 100 samples
                if sample_count % 100 == 0:
                    elapsed = time.time() - start_ts
                    rate = sample_count / elapsed if elapsed > 0 else 0
                    print(f"[i] Collected {sample_count} samples ({rate:.1f} samples/sec)")

                # Sleep to maintain interval
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[!] Collection stopped by user.")

        print(f"[+] Completed. Total samples: {sample_count}")
        print(f"[+] Output → {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced process collector with comprehensive metrics")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--interval", type=float, default=0.5, 
                       help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Total duration in seconds (default: infinite)")
    args = parser.parse_args()

    collect_process_data(args.session, args.interval, args.duration)


if __name__ == "__main__":
    main()
