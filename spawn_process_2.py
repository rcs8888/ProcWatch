#!/usr/bin/env python3
"""
spawn_process_2.py
Anomaly spawner with realistic anomaly types.

- Logs PIDs of anomalous processes
- CPU anomalies, incl. gradual escalation
- Memory anomalies
- I/O anomalies
- Network anomalies (port scanning simulation)
- File descriptor leaks
- Fork bomb (light, controlled)
- Benign baseline generation

Author: Rachel Soubier
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
import socket
import tempfile

# ------------------------
# Logging
# ------------------------
def log_event(session: str, event_type: str, details: str):
    log_dir = os.path.join("logs", session)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "anomaly_events.csv")
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts},{event_type},{details}\n"
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

def network_worker(stop_evt: Event):
    """Simulate network anomaly - rapid connection attempts."""
    hosts = ['8.8.8.8', '1.1.1.1', '208.67.222.222']  # Public DNS servers
    ports = [53, 80, 443]
    
    try:
        while not stop_evt.is_set():
            # Burst of connection attempts
            for _ in range(random.randint(5, 15)):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    sock.connect((random.choice(hosts), random.choice(ports)))
                    sock.close()
                except:
                    pass
            time.sleep(random.uniform(0.1, 0.5))
    except Exception:
        pass

def fd_leak_worker(stop_evt: Event):
    """File descriptor leak - opens files without closing."""
    files = []
    temp_dir = tempfile.gettempdir()
    try:
        counter = 0
        while not stop_evt.is_set():
            # Open files without closing (leak)
            for _ in range(random.randint(3, 10)):
                try:
                    filepath = os.path.join(temp_dir, f"leak_{os.getpid()}_{counter}.tmp")
                    f = open(filepath, "w")
                    f.write("leak" * 100)
                    files.append(f)  # Keep reference so it doesn't close
                    counter += 1
                except:
                    break
            time.sleep(random.uniform(0.2, 0.8))
            
            # Occasionally close some to vary the pattern
            if random.random() < 0.3 and len(files) > 10:
                for _ in range(random.randint(1, 5)):
                    if files:
                        try:
                            files.pop().close()
                        except:
                            pass
    finally:
        # Cleanup
        for f in files:
            try:
                f.close()
            except:
                pass

# ------------------------
# Helper subprocess noise (benign)
# ------------------------
def spawn_benign_subprocess():
    """Short-lived benign subprocess that will NOT be logged as an anomaly."""
    try:
        if random.random() < 0.75:
            p = subprocess.Popen(["sleep", str(random.randint(1, 3))], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            p = subprocess.Popen(["/bin/echo", "ok"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.pid
    except Exception:
        return None

# ------------------------
# Helpers
# ------------------------
def has_stress_ng():
    try:
        subprocess.run(["stress-ng", "--version"], stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, check=True)
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
def spawn_normal(session: str, duration: int = 30):
    """
    Spawn benign background tasks - LONGER duration for better baseline.
    """
    details = f"normal,duration={duration}"
    log_event(session, "normal_start", details)
    print(f"[+] Starting normal background (benign) for {duration}s")

    end = time.time() + duration
    while time.time() < end:
        spawn_benign_subprocess()
        time.sleep(random.uniform(0.5, 2.0))

    log_event(session, "normal_end", details)
    print("[+] Normal background finished")

def spawn_cpu(session: str, duration: int = 20, workers: int = 3, intensity: float = 0.9):
    """CPU anomaly with logged PIDs."""
    my_pid = os.getpid()
    details = f"cpu,pid={my_pid},duration={duration},workers={workers},intensity={intensity}"
    log_event(session, "anomaly_start", details)
    print(f"[+] CPU anomaly (PID: {my_pid})")

    procs = []
    stop_evts = []
    end_time = time.time() + duration

    if has_stress_ng():
        try:
            p = subprocess.Popen(["stress-ng", "--cpu", str(workers), "--timeout", f"{duration}s"])
            procs.append(p)
            print(f"[+] Spawned stress-ng PID: {p.pid}")
            while time.time() < end_time:
                if random.random() < 0.3:
                    spawn_benign_subprocess()
                time.sleep(random.uniform(0.4, 1.5))
        finally:
            _teardown(stop_evts, procs)
    else:
        for i in range(workers):
            evt = Event()
            p = Process(target=cpu_worker, args=(evt, intensity), daemon=True)
            p.start()
            print(f"[+] Spawned CPU worker #{i+1} PID: {p.pid}")
            stop_evts.append(evt)
            procs.append(p)

        while time.time() < end_time:
            time.sleep(random.uniform(0.2, 1.0))
            if random.random() < 0.3:
                spawn_benign_subprocess()

        _teardown(stop_evts, procs)

    log_event(session, "anomaly_end", details)
    print("[+] CPU anomaly finished")

def spawn_cpu_gradual(session: str, total_duration: int = 30):
    """
    CPU anomaly with GRADUAL escalation (more realistic).
    """
    my_pid = os.getpid()
    details = f"cpu_gradual,pid={my_pid},duration={total_duration}"
    log_event(session, "anomaly_start", details)
    print(f"[+] Gradual CPU anomaly (PID: {my_pid})")
    
    # Ramp up intensity gradually
    intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
    segment_duration = total_duration // len(intensities)
    
    for intensity in intensities:
        print(f"[>] Ramping to intensity {intensity}")
        spawn_cpu(session, duration=segment_duration, workers=2, intensity=intensity)
    
    log_event(session, "anomaly_end", details)
    print("[+] Gradual CPU anomaly finished")

def spawn_mem(session: str, duration: int = 20, workers: int = 2, size_mb: int = 150):
    """Memory anomaly with logged PIDs."""
    my_pid = os.getpid()
    details = f"mem,pid={my_pid},duration={duration},workers={workers},size_mb={size_mb}"
    log_event(session, "anomaly_start", details)
    print(f"[+] Memory anomaly (PID: {my_pid})")

    procs = []
    stop_evts = []
    end_time = time.time() + duration

    if has_stress_ng():
        try:
            p = subprocess.Popen(["stress-ng", "--vm", str(workers), "--vm-bytes", 
                                f"{size_mb}M", "--timeout", f"{duration}s"])
            procs.append(p)
            print(f"[+] Spawned stress-ng PID: {p.pid}")
            while time.time() < end_time:
                if random.random() < 0.25:
                    spawn_benign_subprocess()
                time.sleep(random.uniform(0.4, 1.2))
        finally:
            _teardown(stop_evts, procs)
    else:
        for i in range(workers):
            evt = Event()
            p = Process(target=mem_worker, args=(evt, size_mb), daemon=True)
            p.start()
            print(f"[+] Spawned memory worker #{i+1} PID: {p.pid}")
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
    """I/O anomaly with logged PIDs."""
    my_pid = os.getpid()
    details = f"io,pid={my_pid},duration={duration},workers={workers}"
    log_event(session, "anomaly_start", details)
    print(f"[+] I/O anomaly (PID: {my_pid})")

    procs = []
    stop_evts = []
    end_time = time.time() + duration

    for i in range(workers):
        tmp = f"/tmp/io_{session}_{i}_{random.randint(1000,9999)}.log"
        evt = Event()
        p = Process(target=io_worker, args=(evt, tmp), daemon=True)
        p.start()
        print(f"[+] Spawned I/O worker #{i+1} PID: {p.pid}")
        stop_evts.append(evt)
        procs.append(p)

    while time.time() < end_time:
        if random.random() < 0.35:
            spawn_benign_subprocess()
        time.sleep(random.uniform(0.1, 0.6))

    _teardown(stop_evts, procs)
    log_event(session, "anomaly_end", details)
    print("[+] I/O anomaly finished")

def spawn_network(session: str, duration: int = 20, workers: int = 3):
    """Network anomaly - rapid connection attempts."""
    my_pid = os.getpid()
    details = f"network,pid={my_pid},duration={duration},workers={workers}"
    log_event(session, "anomaly_start", details)
    print(f"[+] Network anomaly (PID: {my_pid})")

    procs = []
    stop_evts = []
    end_time = time.time() + duration

    for i in range(workers):
        evt = Event()
        p = Process(target=network_worker, args=(evt,), daemon=True)
        p.start()
        print(f"[+] Spawned network worker #{i+1} PID: {p.pid}")
        stop_evts.append(evt)
        procs.append(p)

    while time.time() < end_time:
        time.sleep(0.5)

    _teardown(stop_evts, procs)
    log_event(session, "anomaly_end", details)
    print("[+] Network anomaly finished")

def spawn_fd_leak(session: str, duration: int = 20, workers: int = 2):
    """File descriptor leak anomaly."""
    my_pid = os.getpid()
    details = f"fd_leak,pid={my_pid},duration={duration},workers={workers}"
    log_event(session, "anomaly_start", details)
    print(f"[+] File descriptor leak anomaly (PID: {my_pid})")

    procs = []
    stop_evts = []
    end_time = time.time() + duration

    for i in range(workers):
        evt = Event()
        p = Process(target=fd_leak_worker, args=(evt,), daemon=True)
        p.start()
        print(f"[+] Spawned FD leak worker #{i+1} PID: {p.pid}")
        stop_evts.append(evt)
        procs.append(p)

    while time.time() < end_time:
        time.sleep(0.5)

    _teardown(stop_evts, procs)
    log_event(session, "anomaly_end", details)
    print("[+] FD leak anomaly finished")

def spawn_fork_bomb_light(session: str, duration: int = 15):
    """
    Light fork bomb - controlled rapid process spawning.
    BE CAREFUL - this spawns many processes quickly!
    """
    my_pid = os.getpid()
    details = f"fork_bomb,pid={my_pid},duration={duration}"
    log_event(session, "anomaly_start", details)
    print(f"[+] Fork bomb (light/controlled) anomaly (PID: {my_pid})")
    
    children = []
    end_time = time.time() + duration
    max_children = 50  # Safety limit
    
    try:
        while time.time() < end_time and len(children) < max_children:
            # Spawn short-lived processes rapidly
            p = subprocess.Popen(["sleep", "1"], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children.append(p)
            time.sleep(0.05)  # Very short interval
            
            # Clean up finished processes
            children = [c for c in children if c.poll() is None]
    finally:
        # Cleanup
        for p in children:
            try:
                p.terminate()
                p.wait(timeout=1)
            except:
                pass
    
    log_event(session, "anomaly_end", details)
    print("[+] Fork bomb anomaly finished")

def spawn_mixed(session: str):
    """Mixed anomaly - combination of types."""
    my_pid = os.getpid()
    details = f"mixed,pid={my_pid}"
    log_event(session, "anomaly_start", details)
    print(f"[+] Mixed anomaly (PID: {my_pid})")

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
    parser = argparse.ArgumentParser(description="Enhanced process anomaly spawner")
    parser.add_argument("--session", required=True, help="session name")
    args = parser.parse_args()
    session = args.session

    print(f"[*] Enhanced Spawner (session={session})")
    print(f"[*] Logs → logs/{session}/anomaly_events.csv")
    if has_stress_ng():
        print("[i] stress-ng found - will use for stronger stress")

    menu = """
Choose:
1) Normal background (benign, 30s)
2) CPU anomaly
3) CPU anomaly (gradual escalation)
4) Memory anomaly
5) I/O anomaly
6) Network anomaly
7) File descriptor leak
8) Fork bomb light (USE WITH CAUTION!)
9) Mixed anomaly
10) Quit
"""

    while True:
        print(menu)
        choice = input("Select [1-10]: ").strip()
        if choice == "1":
            spawn_normal(session, duration=30)
        elif choice == "2":
            spawn_cpu(session, duration=20, workers=3, intensity=0.9)
        elif choice == "3":
            spawn_cpu_gradual(session, total_duration=30)
        elif choice == "4":
            spawn_mem(session, duration=20, workers=2, size_mb=150)
        elif choice == "5":
            spawn_io(session, duration=20, workers=2)
        elif choice == "6":
            spawn_network(session, duration=20, workers=3)
        elif choice == "7":
            spawn_fd_leak(session, duration=20, workers=2)
        elif choice == "8":
            confirm = input("Fork bomb spawns many processes. Continue? [y/N]: ").strip().lower()
            if confirm == 'y':
                spawn_fork_bomb_light(session, duration=15)
            else:
                print("Skipped.")
        elif choice == "9":
            spawn_mixed(session)
        elif choice == "10":
            print("Exiting.")
            break
        else:
            print("Invalid choice — enter 1-10.")
        print("\n--- done ---\n")
        time.sleep(0.2)

if __name__ == "__main__":
    main()
