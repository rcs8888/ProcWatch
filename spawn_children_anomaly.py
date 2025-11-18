#!/usr/bin/env python3
"""
spawn_children_anomaly.py (cleaned & fixed)
-------------------------------------------
Interactive anomaly spawner with increased realism and randomness to make
heuristic detection harder while still producing observable load for your
process_collector/prepare_dataset pipeline.

Usage:
    python3 spawn_children_anomaly.py --session testsession1

Author: Rachel Catherine Soubier (upgraded, cleaned)
Date: November 2025
"""

import os
import time
import random
import subprocess
import argparse
from datetime import datetime, timezone
from multiprocessing import Process, Event

# ------------------------
# Worker implementations (improved)
# ------------------------

def cpu_worker(stop_evt: Event):
    """Tight CPU loop until stop_evt is set (keeps CPU fully busy)."""
    while not stop_evt.is_set():
        x = 0
        for i in range(1000):
            x += i * i

def cpu_worker_with_intensity(stop_evt: Event, intensity: float):
    """
    Busy loop that occasionally sleeps to reduce average CPU.
    intensity: 0.0-1.0 (1.0 -> max busy)
    """
    busy_iters = int(20000 * intensity) + 1
    sleep_chance = max(0.0, 1.0 - intensity)
    while not stop_evt.is_set():
        s = 0
        for _ in range(busy_iters):
            s += 1234 * 5678
        # occasional tiny random pause to create jitter
        if random.random() < sleep_chance:
            time.sleep(0.005 + random.random() * 0.02)

def mem_worker(stop_evt: Event, size_mb: int):
    """
    Allocate approximately size_mb MB but randomly add/remove chunks to
    create a fluctuating memory footprint instead of a perfect plateau.
    """
    chunks = []
    per_chunk = 1 * 1024 * 1024
    try:
        # gradually allocate some baseline
        target = max(1, size_mb // 3)
        for _ in range(target):
            if stop_evt.is_set():
                return
            chunks.append(bytearray(per_chunk))
            time.sleep(0.01)

        # random fluctuate for the remainder
        while not stop_evt.is_set():
            # sometimes add
            if random.random() < 0.4 and len(chunks) < size_mb + 50:
                add_n = random.randint(1, min(5, max(1, size_mb)))
                for _ in range(add_n):
                    chunks.append(bytearray(per_chunk))
                time.sleep(random.uniform(0.01, 0.1))
            # sometimes free
            if random.random() < 0.4 and chunks:
                rem_n = random.randint(1, min(5, len(chunks)))
                for _ in range(rem_n):
                    chunks.pop()
                time.sleep(random.uniform(0.01, 0.05))
            # small idle to avoid hammering memory ops
            time.sleep(random.uniform(0.05, 0.25))
    finally:
        try:
            del chunks
        except Exception:
            pass

def io_worker(stop_evt: Event, tmpfile: str):
    """
    Do bursty writes to tmpfile until stopped.
    Bursts + quiet periods create non-uniform I/O patterns.
    """
    try:
        with open(tmpfile, "w") as f:
            while not stop_evt.is_set():
                burst = random.randint(50, 600)
                for _ in range(burst):
                    # write a moderate line
                    f.write("".join(str(random.randint(0, 9)) for _ in range(200)) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                time.sleep(random.uniform(0.05, 0.35))
    except Exception:
        # ignore IO errors
        pass

# ------------------------
# Helpers: stress-ng wrapper
# ------------------------

def has_stress_ng():
    """Return True if stress-ng is available on PATH."""
    try:
        subprocess.run(["stress-ng", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def run_stress_ng_cpu(workers: int, timeout: int):
    cmd = ["stress-ng", "--cpu", str(workers), "--timeout", f"{timeout}s", "--metrics-brief"]
    return subprocess.Popen(cmd)

def run_stress_ng_vm(workers: int, size_mb: int, timeout: int):
    cmd = ["stress-ng", "--vm", str(workers), "--vm-bytes", f"{size_mb}M", "--timeout", f"{timeout}s", "--metrics-brief"]
    return subprocess.Popen(cmd)

def run_stress_ng_io(workers: int, timeout: int):
    cmd = ["stress-ng", "--hdd", str(workers), "--timeout", f"{timeout}s", "--metrics-brief"]
    return subprocess.Popen(cmd)

# ------------------------
# Logging
# ------------------------

def log_event(session: str, event_type: str, details: str):
    """
    Append event: UTC ISO timestamp, event_type, details
    Kept identical to previous format for compatibility with prepare_dataset.py
    """
    log_dir = os.path.join("logs", session)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "anomaly_events.csv")
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{ts},{event_type},{details}\n")
    print(f"[log] {ts} {event_type} {details}")

# ------------------------
# Utility: spawn benign noise
# ------------------------

def spawn_benign_noise():
    """Spawn a small, short-lived benign process to blur patterns."""
    try:
        if random.random() < 0.6:
            p = subprocess.Popen(["sleep", str(random.randint(1, 5))])
        else:
            p = subprocess.Popen(["/bin/echo", "ok"])
        # p.pid exists for both Popen and Process-like objects; guard just in case
        pid = getattr(p, "pid", None)
        print(f"  [noise] benign pid={pid}")
    except Exception:
        pass

# ------------------------
# High-level spawn functions (upgraded)
# ------------------------

def spawn_cpu(session: str, duration: int = 30, workers: int = 4, intensity: float = 0.9):
    """
    CPU anomaly with mid-run churn, quiet dips, and benign noise interleaving.
    Uses stress-ng if available, otherwise Python workers.
    """
    log_event(session, "anomaly_start", f"cpu,duration={duration},workers={workers},intensity={intensity}")
    print(f"[+] CPU anomaly: duration={duration}s, workers={workers}, intensity={intensity}")

    procs = []      # list of (Process, meta) or ("stress-ng", Popen)
    stop_evts = []  # list of Event objects controlling python Process workers

    if has_stress_ng():
        p = run_stress_ng_cpu(workers=workers, timeout=duration)
        procs.append(("stress-ng", p))
        end_time = time.time() + duration
        while time.time() < end_time:
            if random.random() < 0.25:
                spawn_benign_noise()
            time.sleep(random.uniform(0.5, 2.0))
        # ensure stress-ng ends
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            pass
    else:
        # start Python workers
        for i in range(workers):
            stop_evt = Event()
            p = Process(target=cpu_worker_with_intensity, args=(stop_evt, intensity), daemon=True)
            p.start()
            procs.append((p, None))
            stop_evts.append(stop_evt)
            print(f"  spawned cpu worker pid={getattr(p, 'pid', 'N/A')}")

        start_time = time.time()
        end_time = start_time + duration

        # Periodically perform random churn / quiet dips / benign noise
        while time.time() < end_time:
            # small random sleep to avoid deterministic pattern
            time.sleep(random.uniform(0.2, 1.0))

            # occasional quiet dip (lower CPU briefly)
            if random.random() < 0.5:
                dip = random.uniform(0.5, 2.5)
                print(f"  [cpu] quiet dip for {dip:.2f}s")
                time.sleep(dip)

            # random worker churn: stop one, then start a new one
            if random.random() < 0.45 and stop_evts:
                # pick a live event index
                live_indices = [i for i, e in enumerate(stop_evts) if not e.is_set()]
                if live_indices:
                    idx = random.choice(live_indices)
                    try:
                        stop_evts[idx].set()
                        print(f"  [cpu] terminated worker idx={idx}")
                    except Exception:
                        pass
                    # remove events that are set
                    stop_evts = [e for e in stop_evts if not e.is_set()]
                    time.sleep(random.uniform(0.1, 0.8))
                    # start a new worker
                    new_evt = Event()
                    newproc = Process(target=cpu_worker_with_intensity, args=(new_evt, intensity), daemon=True)
                    newproc.start()
                    procs.append((newproc, None))
                    stop_evts.append(new_evt)
                    print(f"  [cpu] spawned new worker pid={getattr(newproc, 'pid', 'N/A')}")

            # occasionally spawn benign noise processes while CPU anomaly runs
            if random.random() < 0.35:
                spawn_benign_noise()

        # teardown python workers
        for evt in stop_evts:
            try:
                evt.set()
            except Exception:
                pass
        for p, _ in procs:
            if isinstance(p, Process):
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass

    log_event(session, "anomaly_end", f"cpu,duration={duration},workers={workers},intensity={intensity}")
    print("[+] CPU anomaly finished.")

def spawn_mem(session: str, duration: int = 30, workers: int = 3, size_mb: int = 150):
    """
    Memory anomaly with fluctuating allocation behavior and occasional benign noise.
    """
    log_event(session, "anomaly_start", f"mem,duration={duration},workers={workers},size_mb={size_mb}")
    print(f"[+] MEM anomaly: duration={duration}s, workers={workers}, size_mb={size_mb}")

    procs = []
    stop_evts = []

    if has_stress_ng():
        p = run_stress_ng_vm(workers=workers, size_mb=size_mb, timeout=duration)
        procs.append(("stress-ng", p))
        end_time = time.time() + duration
        while time.time() < end_time:
            if random.random() < 0.3:
                spawn_benign_noise()
            time.sleep(random.uniform(0.5, 1.5))
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            pass
    else:
        for i in range(workers):
            stop_evt = Event()
            p = Process(target=mem_worker, args=(stop_evt, size_mb), daemon=True)
            p.start()
            procs.append((p, None))
            stop_evts.append(stop_evt)
            print(f"  spawned mem worker pid={getattr(p, 'pid', 'N/A')}")

        start_time = time.time()
        end_time = start_time + duration
        while time.time() < end_time:
            time.sleep(random.uniform(0.2, 0.8))
            if random.random() < 0.35:
                spawn_benign_noise()

        for evt in stop_evts:
            try:
                evt.set()
            except Exception:
                pass
        for p, _ in procs:
            if isinstance(p, Process):
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass

    log_event(session, "anomaly_end", f"mem,duration={duration},workers={workers},size_mb={size_mb}")
    print("[+] MEM anomaly finished.")

def spawn_io(session: str, duration: int = 30, workers: int = 2):
    """
    I/O anomaly with bursty writes and random quiet periods. Writes to /tmp files.
    """
    log_event(session, "anomaly_start", f"io,duration={duration},workers={workers}")
    print(f"[+] IO anomaly: duration={duration}s, workers={workers}")

    procs = []
    stop_evts = []

    if has_stress_ng():
        p = run_stress_ng_io(workers=workers, timeout=duration)
        procs.append(("stress-ng", p))
        end_time = time.time() + duration
        while time.time() < end_time:
            if random.random() < 0.35:
                spawn_benign_noise()
            time.sleep(random.uniform(0.5, 1.5))
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            pass
    else:
        for i in range(workers):
            stop_evt = Event()
            tmpf = f"/tmp/io_stress_{session}_{i}.log"
            p = Process(target=io_worker, args=(stop_evt, tmpf), daemon=True)
            p.start()
            procs.append((p, tmpf))
            stop_evts.append(stop_evt)
            print(f"  spawned io worker pid={getattr(p, 'pid', 'N/A')} -> {tmpf}")

        start_time = time.time()
        end_time = start_time + duration
        while time.time() < end_time:
            time.sleep(random.uniform(0.1, 0.8))
            if random.random() < 0.45:
                spawn_benign_noise()

        for evt in stop_evts:
            try:
                evt.set()
            except Exception:
                pass
        for p, tmpf in procs:
            if isinstance(p, Process):
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass

    log_event(session, "anomaly_end", f"io,duration={duration},workers={workers}")
    print("[+] IO anomaly finished.")

def spawn_mixed(session: str):
    """
    A mixed anomaly that runs short CPU, then mem, then IO segments, with small overlaps.
    """
    spawn_cpu(session, duration=10, workers=3, intensity=0.85)
    time.sleep(random.uniform(0.5, 1.5))
    spawn_mem(session, duration=10, workers=2, size_mb=100)
    time.sleep(random.uniform(0.5, 1.5))
    spawn_io(session, duration=10, workers=1)
    print("[+] Mixed anomaly finished.")

def spawn_normal(session: str, duration: int = 8):
    """
    Normal background behavior: spawn a few short-lived benign processes.
    """
    log_event(session, "normal_start", f"normal,duration={duration}")
    print(f"[+] Normal background: spawning sleep processes for {duration}s")
    procs = []
    for _ in range(random.randint(2, 6)):
        p = subprocess.Popen(["sleep", str(random.randint(2, duration))])
        procs.append(p)
        print(f"  spawned background pid={getattr(p, 'pid', 'N/A')}")
        time.sleep(random.uniform(0.1, 0.4))

    time.sleep(duration)
    log_event(session, "normal_end", f"normal,duration={duration}")
    print("[+] Normal background finished.")

# ------------------------
# Interactive menu
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive anomaly spawner (use with process_collector.py)")
    parser.add_argument("--session", required=True, help="session name to match collector (e.g., testsession1)")
    args = parser.parse_args()
    session = args.session

    print(f"[*] Anomaly spawner (session={session})")
    if has_stress_ng():
        print("[i] stress-ng detected on PATH; will use it for heavier, more reliable stress if available.")
    else:
        print("[i] stress-ng not found; using Python-based workers (these also show up in top/ps).")

    menu = """
Choose an option:
1) Spawn normal background tasks
2) CPU stress anomaly (moderate)
3) Memory stress anomaly (moderate)
4) I/O stress anomaly (moderate)
5) Mixed (CPU + MEM + IO)
6) Quit
"""

    try:
        while True:
            print(menu)
            choice = input("Enter your choice [1-6]: ").strip()
            if choice == "1":
                spawn_normal(session, duration=8)
            elif choice == "2":
                spawn_cpu(session, duration=25, workers=4, intensity=0.9)
            elif choice == "3":
                spawn_mem(session, duration=25, workers=3, size_mb=200)
            elif choice == "4":
                spawn_io(session, duration=25, workers=2)
            elif choice == "5":
                spawn_mixed(session)
            elif choice == "6":
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please choose 1-6.")
            print("\n--- done ---\n")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user. Cleaning up...")

if __name__ == "__main__":
    main()
