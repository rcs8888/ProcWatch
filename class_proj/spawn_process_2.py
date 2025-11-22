#!/usr/bin/env python3
"""
spawn_process_2.py
---------------------------
ML-friendly anomaly spawner that produces realistic CPU / MEM / IO events.

Log format (compatible with prepare_dataset.py):
  logs/<session>/anomaly_events.csv
  each line: <UTC-ISO-timestamp>,<event_type>,<details>

Event types used:
  - normal_start / normal_end
  - anomaly_start / anomaly_end

Features:
 - Stable but variable-intensity CPU workers (with jitter)
 - Fluctuating memory workers (allocate/free chunks)
 - Bursty I/O writers (bursts + quiet periods)
 - Benign background noise (short sleeps / echo) interleaved during anomalies
 - Mid-run worker churn and quiet dips to avoid perfectly deterministic patterns

Usage:
    python3 spawn_process_2.py --session testsession1
"""

from multiprocessing import Process, Event
from datetime import datetime, timezone
import argparse
import os
import time
import random
import subprocess
import signal
import sys

# ------------------------
# Logging
# ------------------------
def log_event(session: str, event_type: str, details: str):
    log_dir = os.path.join("logs", session)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "anomaly_events.csv")
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{ts},{event_type},{details}\n")
    print(f"[log] {ts} {event_type} {details}")

# ------------------------
# Worker implementations
# ------------------------
def cpu_worker(stop_evt: Event, intensity: float):
    """Busy loop with adjustable intensity + short sleeps to create jitter."""
    # intensity: 0.0 - 1.0
    base_iters = max(1000, int(20000 * intensity))
    # slightly vary it per loop to avoid exact periodicity
    while not stop_evt.is_set():
        iters = base_iters + random.randint(-base_iters//10, base_iters//10)
        s = 0
        for _ in range(iters):
            s += 12345 * 6789  # cheap integer math
        # small probabilistic sleep to reduce average CPU and create jitter
        if random.random() < (1.0 - intensity) * 0.8:
            time.sleep(0.002 + random.random() * 0.01)

def mem_worker(stop_evt: Event, target_mb: int):
    """
    Allocate memory in 1MB chunks up to an initial baseline, then randomly add/remove
    small numbers of chunks to produce a fluctuating footprint.
    """
    chunks = []
    per_chunk = 1 * 1024 * 1024
    try:
        baseline = max(1, target_mb // 3)
        # gradually allocate baseline
        for _ in range(baseline):
            if stop_evt.is_set():
                return
            chunks.append(bytearray(per_chunk))
            time.sleep(0.01)
        # fluctuate for remainder
        while not stop_evt.is_set():
            r = random.random()
            if r < 0.35 and len(chunks) < target_mb + 50:
                # add 1-4 chunks
                add = random.randint(1, min(4, target_mb))
                for _ in range(add):
                    chunks.append(bytearray(per_chunk))
                time.sleep(random.uniform(0.01, 0.08))
            elif r < 0.7 and chunks:
                rem = random.randint(1, min(4, len(chunks)))
                for _ in range(rem):
                    chunks.pop()
                time.sleep(random.uniform(0.005, 0.04))
            else:
                time.sleep(random.uniform(0.05, 0.2))
    finally:
        # free memory
        try:
            del chunks
        except Exception:
            pass

def io_worker(stop_evt: Event, tmpfile: str):
    """
    Burst writes to tmpfile followed by quiet periods to create non-uniform IO.
    """
    try:
        with open(tmpfile, "w") as f:
            while not stop_evt.is_set():
                burst_len = random.randint(40, 400)
                for _ in range(burst_len):
                    f.write("".join(str(random.randint(0,9)) for _ in range(160)) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                # quiet / jitter
                time.sleep(random.uniform(0.03, 0.4))
    except Exception:
        # ignore file errors
        pass

# ------------------------
# Helper subprocess noise
# ------------------------
def spawn_benign_subprocess():
    """Spawn a short-lived benign subprocess (sleep or echo) to create noise."""
    try:
        if random.random() < 0.7:
            # short sleep so it shows in process lists briefly
            p = subprocess.Popen(["sleep", str(random.randint(1, 3))])
        else:
            p = subprocess.Popen(["/bin/echo", "ok"])
        return getattr(p, "pid", None)
    except Exception:
        return None

# ------------------------
# stress-ng availability wrapper (keeps optional)
# ------------------------
def has_stress_ng():
    try:
        subprocess.run(["stress-ng", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

# ------------------------
# High-level spawners
# ------------------------
def _teardown_processes(stop_evts, procs):
    # signal python workers to stop
    for evt in stop_evts:
        try:
            evt.set()
        except Exception:
            pass
    # join processes gracefully
    for p in procs:
        try:
            if isinstance(p, Process):
                p.join(timeout=1.0)
            else:
                # p may be a subprocess.Popen for stress-ng
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            pass

def spawn_normal(session: str, duration: int = 10):
    log_event(session, "normal_start", f"normal,duration={duration}")
    print("[+] Starting normal background activity")
    end = time.time() + duration
    while time.time() < end:
        spawn_benign_subprocess()
        time.sleep(random.uniform(0.3, 0.9))
    log_event(session, "normal_end", f"normal,duration={duration}")
    print("[+] Normal background finished")

def spawn_cpu(session: str, duration: int = 20, workers: int = 3, intensity: float = 0.9):
    log_event(session, "anomaly_start", f"cpu,duration={duration},workers={workers},intensity={intensity}")
    print(f"[+] CPU anomaly: duration={duration}s workers={workers} intensity={intensity}")

    procs = []
    stop_evts = []
    start_time = time.time()
    end_time = start_time + duration

    # If stress-ng available prefer it for reliable stress
    if has_stress_ng():
        try:
            p = subprocess.Popen(["stress-ng", "--cpu", str(workers), "--timeout", f"{duration}s"])
            procs.append(p)
            # while running, spawn benign noise occasionally
            while time.time() < end_time:
                if random.random() < 0.25:
                    spawn_benign_subprocess()
                time.sleep(random.uniform(0.4, 1.5))
        finally:
            # ensure termination
            for p in procs:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
    else:
        # start python CPU workers
        for _ in range(workers):
            evt = Event()
            p = Process(target=cpu_worker, args=(evt, intensity), daemon=True)
            p.start()
            stop_evts.append(evt)
            procs.append(p)

        # mid-run behavior: churn workers, create dips and noise
        while time.time() < end_time:
            time.sleep(random.uniform(0.15, 0.9))

            # quiet dip: short sleep for lower overall intensity
            if random.random() < 0.45:
                dip = random.uniform(0.3, 2.0)
                # temporarily pause main loop to create dip (workers still running but jitter reduces)
                time.sleep(dip)

            # worker churn: randomly replace one worker
            if random.random() < 0.4 and stop_evts:
                # signal one worker to stop, then start a new one
                idx = random.randrange(len(stop_evts))
                try:
                    stop_evts[idx].set()
                except Exception:
                    pass
                # cleanup stop_evts to remove set ones
                stop_evts = [e for e in stop_evts if not getattr(e, "is_set", lambda: False)()]
                # small pause then start a new worker
                time.sleep(random.uniform(0.05, 0.4))
                new_evt = Event()
                newproc = Process(target=cpu_worker, args=(new_evt, intensity), daemon=True)
                newproc.start()
                procs.append(newproc)
                stop_evts.append(new_evt)

            # occasionally spawn benign subprocesses
            if random.random() < 0.35:
                spawn_benign_subprocess()

        # teardown
        _teardown_processes(stop_evts, procs)

    log_event(session, "anomaly_end", f"cpu,duration={duration},workers={workers},intensity={intensity}")
    print("[+] CPU anomaly finished")

def spawn_mem(session: str, duration: int = 20, workers: int = 2, size_mb: int = 150):
    log_event(session, "anomaly_start", f"mem,duration={duration},workers={workers},size_mb={size_mb}")
    print(f"[+] MEM anomaly: duration={duration}s workers={workers} size_mb={size_mb}")

    procs = []
    stop_evts = []
    start_time = time.time()
    end_time = start_time + duration

    if has_stress_ng():
        try:
            p = subprocess.Popen(["stress-ng", "--vm", str(workers), "--vm-bytes", f"{size_mb}M", "--timeout", f"{duration}s"])
            procs.append(p)
            while time.time() < end_time:
                if random.random() < 0.25:
                    spawn_benign_subprocess()
                time.sleep(random.uniform(0.4, 1.2))
        finally:
            for p in procs:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
    else:
        for _ in range(workers):
            evt = Event()
            p = Process(target=mem_worker, args=(evt, size_mb), daemon=True)
            p.start()
            stop_evts.append(evt)
            procs.append(p)

        while time.time() < end_time:
            # occasionally spawn noise
            if random.random() < 0.35:
                spawn_benign_subprocess()
            time.sleep(random.uniform(0.2, 0.8))

        _teardown_processes(stop_evts, procs)

    log_event(session, "anomaly_end", f"mem,duration={duration},workers={workers},size_mb={size_mb}")
    print("[+] MEM anomaly finished")

def spawn_io(session: str, duration: int = 20, workers: int = 2):
    log_event(session, "anomaly_start", f"io,duration={duration},workers={workers}")
    print(f"[+] IO anomaly: duration={duration}s workers={workers}")

    procs = []
    stop_evts = []
    tmpfiles = []
    start_time = time.time()
    end_time = start_time + duration

    for i in range(workers):
        tmp = f"/tmp/io_{session}_{i}.log"
        evt = Event()
        p = Process(target=io_worker, args=(evt, tmp), daemon=True)
        p.start()
        stop_evts.append(evt)
        procs.append(p)
        tmpfiles.append(tmp)

    while time.time() < end_time:
        # occasional benign noise
        if random.random() < 0.35:
            spawn_benign_subprocess()
        time.sleep(random.uniform(0.1, 0.6))

    _teardown_processes(stop_evts, procs)

    # leave small files but do not remove them automatically
    log_event(session, "anomaly_end", f"io,duration={duration},workers={workers}")
    print("[+] IO anomaly finished")

def spawn_mixed(session: str):
    # short overlapping phases with small random offsets to avoid rigid timing
    spawn_cpu(session, duration=10, workers=3, intensity=0.85)
    time.sleep(random.uniform(0.3, 1.0))
    spawn_mem(session, duration=10, workers=2, size_mb=100)
    time.sleep(random.uniform(0.3, 1.0))
    spawn_io(session, duration=8, workers=1)
    print("[+] Mixed anomaly finished")

# ------------------------
# CLI / Menu
# ------------------------
def install_sigint_handler():
    def handler(sig, frame):
        print("\n[!] Interrupted — exiting.")
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

def main():
    install_sigint_handler()
    parser = argparse.ArgumentParser(description="Spawn process anomalies for ML pipeline")
    parser.add_argument("--session", required=True, help="session name (matches logs/<session>/...)")
    args = parser.parse_args()
    session = args.session

    print(f"[*] Spawner (session={session})  — logs will be written to logs/{session}/anomaly_events.csv")
    if has_stress_ng():
        print("[i] stress-ng found on PATH — will use when available for stronger stress.")

    menu = """
Choose:
 1) Normal background
 2) CPU anomaly
 3) Memory anomaly
 4) I/O anomaly
 5) Mixed anomaly
 6) Quit
"""

    while True:
        print(menu)
        choice = input("Select [1-6]: ").strip()
        if choice == "1":
            spawn_normal(session, duration=10)
        elif choice == "2":
            spawn_cpu(session, duration=20, workers=3, intensity=0.9)
        elif choice == "3":
            spawn_mem(session, duration=20, workers=2, size_mb=150)
        elif choice == "4":
            spawn_io(session, duration=20, workers=2)
        elif choice == "5":
            spawn_mixed(session)
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Invalid choice — enter 1-6.")
        print("\n--- done ---\n")
        time.sleep(0.2)

if __name__ == "__main__":
    main()


