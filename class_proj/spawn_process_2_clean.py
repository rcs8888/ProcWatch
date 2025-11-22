#!/usr/bin/env python3
"""
spawn_process_2_clean.py
------------------------
ML-friendly anomaly spawner.

- Benign runs are explicitly logged as normal_start / normal_end.
- Real anomalies are logged as anomaly_start / anomaly_end.
- Log format compatible with prepare_dataset.py:
    logs/<session>/anomaly_events.csv
    each line: <UTC-ISO-timestamp>,<event_type>,<details>

Author: Assistant (adapted for Rachel Soubier)
Date: 2025-11
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
    line = f"{ts},{event_type},{details}\n"
    # append atomically
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[log] {ts} {event_type} {details}")

# ------------------------
# Worker implementations
# ------------------------
def cpu_worker(stop_evt: Event, intensity: float):
    """Busy loop with adjustable intensity + jitter."""
    base_iters = max(1000, int(20000 * max(0.05, intensity)))
    try:
        while not stop_evt.is_set():
            iters = base_iters + random.randint(-base_iters // 10, base_iters // 10)
            s = 0
            for _ in range(iters):
                s += 12345 * 6789
            # small probabilistic sleep to add jitter
            if random.random() < max(0.01, (1.0 - intensity)) * 0.8:
                time.sleep(0.002 + random.random() * 0.01)
    except KeyboardInterrupt:
        pass

def mem_worker(stop_evt: Event, target_mb: int):
    """Fluctuating memory allocation in ~1MB chunks."""
    chunks = []
    per_chunk = 1 * 1024 * 1024
    try:
        baseline = max(1, target_mb // 3)
        for _ in range(baseline):
            if stop_evt.is_set():
                return
            chunks.append(bytearray(per_chunk))
            time.sleep(0.01)
        while not stop_evt.is_set():
            r = random.random()
            if r < 0.35 and len(chunks) < target_mb + 50:
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
        try:
            del chunks
        except Exception:
            pass

def io_worker(stop_evt: Event, tmpfile: str):
    """Bursty writes followed by quiet periods."""
    try:
        with open(tmpfile, "w") as f:
            while not stop_evt.is_set():
                burst_len = random.randint(40, 300)
                for _ in range(burst_len):
                    f.write("".join(str(random.randint(0,9)) for _ in range(160)) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                time.sleep(random.uniform(0.03, 0.35))
    except Exception:
        pass

# ------------------------
# Helper subprocess noise (benign)
# ------------------------
def spawn_benign_subprocess():
    """Short-lived benign subprocess that will NOT be logged as an anomaly."""
    try:
        if random.random() < 0.75:
            p = subprocess.Popen(["sleep", str(random.randint(1, 3))])
        else:
            p = subprocess.Popen(["/bin/echo", "ok"])
        pid = getattr(p, "pid", None)
        # do not log these as anomalies; they are intentionally benign
        print(f"[noise] benign subprocess pid={pid}")
        return pid
    except Exception:
        return None

# ------------------------
# Helpers
# ------------------------
def has_stress_ng():
    try:
        subprocess.run(["stress-ng", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def _teardown(stop_evts, procs):
    for evt in stop_evts:
        try:
            evt.set()
        except Exception:
            pass
    for p in procs:
        try:
            if isinstance(p, Process):
                p.join(timeout=1.0)
            else:
                # subprocess.Popen (stress-ng)
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass

# ------------------------
# Spawners
# ------------------------
def spawn_normal(session: str, duration: int = 10):
    """Spawn benign background tasks and log only normal_start/normal_end."""
    details = f"normal,duration={duration}"
    log_event(session, "normal_start", details)
    print("[+] Starting normal background (benign)")

    end = time.time() + duration
    while time.time() < end:
        spawn_benign_subprocess()
        time.sleep(random.uniform(0.25, 0.9))

    log_event(session, "normal_end", details)
    print("[+] Normal background finished (no anomaly events logged)")

def spawn_cpu(session: str, duration: int = 20, workers: int = 3, intensity: float = 0.9):
    details = f"cpu,duration={duration},workers={workers},intensity={intensity}"
    log_event(session, "anomaly_start", details)
    print("[+] CPU anomaly (logged)")

    procs = []
    stop_evts = []
    start_time = time.time()
    end_time = start_time + duration

    if has_stress_ng():
        try:
            p = subprocess.Popen(["stress-ng", "--cpu", str(workers), "--timeout", f"{duration}s"])
            procs.append(p)
            while time.time() < end_time:
                if random.random() < 0.3:
                    spawn_benign_subprocess()
                time.sleep(random.uniform(0.4, 1.5))
        finally:
            _teardown(stop_evts, procs)
    else:
        for _ in range(workers):
            evt = Event()
            p = Process(target=cpu_worker, args=(evt, intensity), daemon=True)
            p.start()
            stop_evts.append(evt)
            procs.append(p)

        # mid-run churn and jitter
        while time.time() < end_time:
            time.sleep(random.uniform(0.15, 0.9))
            if random.random() < 0.45:
                dip = random.uniform(0.2, 1.8)
                time.sleep(dip)
            if random.random() < 0.35 and stop_evts:
                # stop a worker and start a new one
                idx = random.randrange(len(stop_evts))
                try:
                    stop_evts[idx].set()
                except Exception:
                    pass
                # cleanup removed events
                stop_evts = [e for e in stop_evts if not getattr(e, "is_set", lambda: False)()]
                time.sleep(random.uniform(0.05, 0.4))
                new_evt = Event()
                newproc = Process(target=cpu_worker, args=(new_evt, intensity), daemon=True)
                newproc.start()
                procs.append(newproc)
                stop_evts.append(new_evt)
            if random.random() < 0.35:
                spawn_benign_subprocess()

        _teardown(stop_evts, procs)

    log_event(session, "anomaly_end", details)
    print("[+] CPU anomaly finished")

def spawn_mem(session: str, duration: int = 20, workers: int = 2, size_mb: int = 150):
    details = f"mem,duration={duration},workers={workers},size_mb={size_mb}"
    log_event(session, "anomaly_start", details)
    print("[+] Memory anomaly (logged)")

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
            _teardown(stop_evts, procs)
    else:
        for _ in range(workers):
            evt = Event()
            p = Process(target=mem_worker, args=(evt, size_mb), daemon=True)
            p.start()
            stop_evts.append(evt)
            procs.append(p)

        while time.time() < end_time:
            if random.random() < 0.35:
                spawn_benign_subprocess()
            time.sleep(random.uniform(0.2, 0.8))

        _teardown(stop_evts, procs)

    log_event(session, "anomaly_end", details)
    print("[+] Memory anomaly finished")

def spawn_io(session: str, duration: int = 20, workers: int = 2):
    details = f"io,duration={duration},workers={workers}"
    log_event(session, "anomaly_start", details)
    print("[+] I/O anomaly (logged)")

    procs = []
    stop_evts = []
    start_time = time.time()
    end_time = start_time + duration

    for i in range(workers):
        tmp = f"/tmp/io_{session}_{i}.log"
        evt = Event()
        p = Process(target=io_worker, args=(evt, tmp), daemon=True)
        p.start()
        stop_evts.append(evt)
        procs.append(p)

    while time.time() < end_time:
        if random.random() < 0.35:
            spawn_benign_subprocess()
        time.sleep(random.uniform(0.1, 0.6))

    _teardown(stop_evts, procs)
    log_event(session, "anomaly_end", details)
    print("[+] I/O anomaly finished")

def spawn_mixed(session: str):
    # Mixed anomaly is still an anomaly window (log start/end once)
    details = "mixed,short_sequence"
    log_event(session, "anomaly_start", details)
    print("[+] Mixed anomaly (logged)")

    # overlap short CPU / MEM / IO segments
    spawn_cpu(session, duration=8, workers=2, intensity=0.85)
    time.sleep(random.uniform(0.3, 0.8))
    spawn_mem(session, duration=8, workers=1, size_mb=80)
    time.sleep(random.uniform(0.3, 0.8))
    spawn_io(session, duration=6, workers=1)
    log_event(session, "anomaly_end", details)
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
1) Normal background (benign)
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


