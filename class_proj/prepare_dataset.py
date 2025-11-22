#!/usr/bin/env python3
"""
prepare_dataset.py (Clean, silent version)

Automatically labels process data as normal/anomalous based on anomaly logs.

Usage:
    python3 prepare_dataset.py --session mysession
"""

import pandas as pd
import numpy as np
import os
import argparse
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

    # Process stream
    df = pd.read_csv(proc_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    # Event log
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

    # Filter ONLY anomaly events
    df_events = df_events[df_events["event_type"].str.contains("anomaly", case=False, na=False)]

    # If there are no anomaly events at all, skip labeling entirely
    if df_events.empty:
        return df, [], 0

    # Detect start/end pairs
    for _, event in df_events.iterrows():
        etype = str(event["event_type"]).lower()
        ts = event["timestamp"]

        if "start" in etype:
            current_start = ts
        elif "end" in etype and current_start is not None:
            active_periods.append((current_start, ts))
            current_start = None

    # Handle unmatched starts
    if current_start is not None:
        active_periods.append((current_start, current_start + timedelta(seconds=window)))

    # Apply labels
    total_labeled = 0
    for start, end in active_periods:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        total_labeled += mask.sum()
        df.loc[mask, "label"] = 1

    return df, active_periods, total_labeled





# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def feature_engineering(df):
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce").fillna(0)
    df["mem"] = pd.to_numeric(df["mem"], errors="coerce").fillna(0)

    df = df.sort_values(["pid", "timestamp"])
    df["delta_cpu"] = df.groupby("pid")["cpu"].diff().fillna(0)
    df["delta_mem"] = df.groupby("pid")["mem"].diff().fillna(0)

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    session = args.session

    df, df_events = load_data(session)
    df, anomaly_windows, labeled_samples = label_anomalies(df, df_events, window=15)
    df = feature_engineering(df)

    # Save output
    out_file = f"logs/{session}/labeled_dataset.csv"
    df.to_csv(out_file, index=False)

    # Minimal clean output
    print(f"[✓] Session: {session}")
    print(f"[✓] Samples: {len(df)}")
    print(f"[✓] Events: {len(df_events)}")
    print(f"[✓] Labeled anomalous samples: {labeled_samples}")
    print(f"[✓] Saved → {out_file}")
    print(df["label"].value_counts())


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()


