"""
prepare_dataset.py (clean version)
----------------------------------
Automatically labels process data as normal/anomalous based on anomaly event logs.

Usage:
    python3 prepare_dataset.py
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
def load_data(session):
    base_dir = f"logs/{session}"
    proc_file = os.path.join(base_dir, "process_stream.csv")
    events_file = os.path.join(base_dir, "anomaly_events.csv")

    if not os.path.exists(proc_file):
        raise FileNotFoundError(f"Missing: {proc_file}")
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Missing: {events_file}")

    # Load process monitoring data
    df = pd.read_csv(proc_file)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    ).dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    df = df.dropna(subset=["timestamp"])

    # Load anomaly events
    df_events = pd.read_csv(
        events_file,
        names=["timestamp", "event_type", "category", "param1", "param2", "param3"],
        header=None,
        on_bad_lines="skip"
    )
    df_events["timestamp"] = pd.to_datetime(df_events["timestamp"], errors="coerce", utc=True)
    df_events = df_events.dropna(subset=["timestamp"])

    return df, df_events


# -----------------------------------------------------------------------------
# Label anomalies
# -----------------------------------------------------------------------------
def label_anomalies(df, df_events, window=15):
    df["label"] = 0
    active_periods = []
    current_start = None

    for _, event in df_events.iterrows():
        etype = str(event["event_type"]).lower()
        ts = event["timestamp"]

        if "start" in etype:
            current_start = ts
        elif "end" in etype and current_start is not None:
            active_periods.append((current_start, ts))
            current_start = None

    # Handle leftover "start" without "end"
    if current_start is not None:
        active_periods.append((current_start, current_start + timedelta(seconds=window)))

    # Fallback if nothing detected
    if not active_periods and len(df_events) > 0:
        for _, event in df_events.iterrows():
            if "anomaly" in str(event["event_type"]).lower():
                start = event["timestamp"] - timedelta(seconds=window)
                end = event["timestamp"] + timedelta(seconds=window)
                active_periods.append((start, end))

    # Apply labeling
    total_labeled = 0
    for start, end in active_periods:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        count = mask.sum()
        total_labeled += count
        df.loc[mask, "label"] = 1

    return df, active_periods, total_labeled


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def feature_engineering(df):
    if "cpu" not in df.columns or "mem" not in df.columns:
        raise ValueError("Expected columns 'cpu' and 'mem' in dataset.")

    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce").fillna(0)
    df["mem"] = pd.to_numeric(df["mem"], errors="coerce").fillna(0)

    df = df.sort_values(["pid", "timestamp"])
    df["delta_cpu"] = df.groupby("pid")["cpu"].diff().fillna(0)
    df["delta_mem"] = df.groupby("pid")["mem"].diff().fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    session = input("Enter session name to process: ").strip()
    df, df_events = load_data(session)
    df, anomaly_windows, labeled_samples = label_anomalies(df, df_events, window=15)
    df = feature_engineering(df)

    # -------------------------------------------------------------------------
    # Clean, human-readable output section
    # -------------------------------------------------------------------------
    print(f"\n[✓] Session: {session}")
    print(f"    ├── Process samples: {len(df)}")
    print(f"    ├── Anomaly events: {len(df_events)}")
    print(f"    ├── Process log range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"    └── Event log range:   {df_events['timestamp'].min()} → {df_events['timestamp'].max()}")

    if labeled_samples > 0:
        print(f"\n[+] {len(anomaly_windows)} anomaly window(s) detected:")
        for i, (start, end) in enumerate(anomaly_windows, start=1):
            duration = (end - start).total_seconds()
            print(f"     {i}) {start} → {end}  ({duration:.1f}s)")
        print(f"\n[✓] Total samples labeled anomalous: {labeled_samples}")
    else:
        print("\n[!] No samples labeled anomalous — check timestamps or timezone alignment.")

    out_file = f"logs/{session}/labeled_dataset.csv"
    df.to_csv(out_file, index=False)

    print(f"\n[✓] Saved labeled dataset → {out_file}")
    print(df["label"].value_counts().to_string(index=True))
    print()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
