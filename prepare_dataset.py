#!/usr/bin/env python3
"""
prepare_dataset.py
Dataset preparation with labeling and feature engineering.

- Time window labeling
- I/O feature engineering
- Network and threading features
- Process baseline features (deviation from normal)
- Robust temporal features

Author: Rachel Soubier
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Load raw session logs
# =============================================================================
def load_data(session):
    """Load process stream and anomaly event logs."""
    base = f"logs/{session}"
    proc_file = os.path.join(base, "process_stream.csv")
    event_file = os.path.join(base, "anomaly_events.csv")

    if not os.path.exists(proc_file):
        raise FileNotFoundError(f"Process stream not found: {proc_file}")
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"Event log not found: {event_file}")

    print(f"[+] Loading process stream: {proc_file}")
    df = pd.read_csv(proc_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    print(f"    Loaded {len(df)} process records")

    print(f"[+] Loading anomaly events: {event_file}")
    # Read raw lines to handle variable number of columns
    # Format: timestamp,event_type,detail1,detail2,detail3,...
    events = []
    with open(event_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                # First column is timestamp, second is event_type, rest is details
                timestamp = parts[0]
                event_type = parts[1]
                details = ','.join(parts[2:]) if len(parts) > 2 else ''
                events.append({
                    'timestamp': timestamp,
                    'event_type': event_type,
                    'details': details
                })
    
    df_events = pd.DataFrame(events)
    df_events["timestamp"] = pd.to_datetime(df_events["timestamp"], utc=True, errors="coerce")
    df_events = df_events.dropna(subset=["timestamp"])
    print(f"    Loaded {len(df_events)} events")

    return df, df_events


# =============================================================================
# Label anomalies with exact time windows
# =============================================================================
def label_anomalies(df, df_events, buffer_seconds=2):
    """
    Label using exact anomaly windows from events.
    
    Args:
        df: Process dataframe
        df_events: Event dataframe
        buffer_seconds: Small buffer before/after anomaly window (default 2s)
    
    Returns:
        Labeled dataframe, list of anomaly periods, total labeled count
    """
    print(f"\n[+] Labeling anomalies with exact time windows (+/- {buffer_seconds}s buffer)")
    
    df["label"] = 0
    
    # Filter for anomaly events only
    anomaly_events = df_events[df_events["event_type"].str.contains("anomaly", case=False, na=False)]
    
    if anomaly_events.empty:
        print("    [!] WARNING: No anomaly events found!")
        return df, [], 0
    
    # Parse start/end pairs
    active_periods = []
    current_start = None
    
    for _, e in anomaly_events.iterrows():
        event = str(e["event_type"]).lower()
        ts = e["timestamp"]
        
        if "start" in event:
            if current_start is not None:
                # Previous anomaly didn't have an end - close it
                print(f"    [!] WARNING: Unclosed anomaly at {current_start}, closing at {ts}")
                active_periods.append((current_start, ts))
            current_start = ts
        elif "end" in event:
            if current_start is not None:
                active_periods.append((current_start, ts))
                current_start = None
            else:
                print(f"    [!] WARNING: Anomaly end without start at {ts}")
    
    # Handle incomplete event (anomaly started but never ended)
    if current_start is not None:
        end_time = df["timestamp"].max()
        print(f"    [!] WARNING: Unclosed anomaly at {current_start}, extending to end of data")
        active_periods.append((current_start, end_time))
    
    print(f"    Found {len(active_periods)} anomaly periods:")
    
    # Label each period
    total_labeled = 0
    for i, (start, end) in enumerate(active_periods, 1):
        # Add buffer
        start_buffered = start - timedelta(seconds=buffer_seconds)
        end_buffered = end + timedelta(seconds=buffer_seconds)
        
        duration = (end - start).total_seconds()
        print(f"      Period {i}: {start} to {end} (duration: {duration:.1f}s)")
        
        mask = (df["timestamp"] >= start_buffered) & (df["timestamp"] <= end_buffered)
        labeled_count = mask.sum()
        df.loc[mask, "label"] = 1
        total_labeled += labeled_count
        
        print(f"        → Labeled {labeled_count} rows")
    
    print(f"    Total labeled as anomalies: {total_labeled} / {len(df)} ({total_labeled/len(df)*100:.2f}%)")
    
    return df, active_periods, total_labeled


# =============================================================================
# Feature Engineering
# =============================================================================

def feature_engineering(df):
    """Comprehensive feature engineering for anomaly detection."""
    print("\n[+] Feature engineering...")
    
    # ===== BASIC CLEANUP =====
    print("    [1/8] Basic cleanup and normalization")
    numeric_cols = ['cpu', 'mem', 'num_children', 'child_entropy', 
                    'io_read_bytes', 'io_write_bytes', 'io_read_count', 'io_write_count',
                    'num_connections', 'num_threads', 'num_fds',
                    'ctx_switches_voluntary', 'ctx_switches_involuntary']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0
    
    # Sort for time-series operations
    df = df.sort_values(["pid", "timestamp"]).copy()
    
    # ===== FORWARD-FILL MISSING VALUES (better for time series) =====
    print("    [2/8] Forward-filling missing values")
    df['cpu'] = df.groupby('pid')['cpu'].ffill().bfill().fillna(0)
    df['mem'] = df.groupby('pid')['mem'].ffill().bfill().fillna(0)
    
    # ===== SHORT-TERM ROLLING STATISTICS =====
    print("    [3/8] Rolling statistics")
    for window in [5, 10]:
        df[f'cpu_mean_{window}'] = df.groupby("pid")["cpu"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'cpu_std_{window}'] = df.groupby("pid")["cpu"].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        df[f'mem_mean_{window}'] = df.groupby("pid")["mem"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'mem_std_{window}'] = df.groupby("pid")["mem"].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    
    # Rates relative to mean/std
    df["cpu_rate"] = (df["cpu"] - df["cpu_mean_5"]) / (df["cpu_std_5"] + 1e-5)
    df["mem_rate"] = (df["mem"] - df["mem_mean_5"]) / (df["mem_std_5"] + 1e-5)
    
    # ===== EXPONENTIAL MOVING AVERAGES =====
    print("    [4/8] Exponential moving averages")
    df["cpu_ema"] = df.groupby("pid")["cpu"].ewm(span=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df["mem_ema"] = df.groupby("pid")["mem"].ewm(span=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    df["cpu_spike"] = df["cpu"] / (df["cpu_ema"] + 1e-5)
    df["mem_spike"] = df["mem"] / (df["mem_ema"] + 1e-5)
    
    df["has_children"] = (df["num_children"] > 0).astype(int)
    
    # ===== DELTAS (CHANGES) =====
    print("    [5/8] Delta features")
    df["delta_cpu"] = df.groupby("pid")["cpu"].diff().fillna(0)
    df["delta_mem"] = df.groupby("pid")["mem"].diff().fillna(0)
    
    # Child process growth
    df["children_rate"] = df.groupby("pid")["num_children"].diff().fillna(0)
    
    # ===== I/O FEATURES =====
    print("    [6/8] I/O features")
    # I/O rates (bytes per observation period)
    df["io_read_rate"] = df.groupby("pid")["io_read_bytes"].diff().fillna(0)
    df["io_write_rate"] = df.groupby("pid")["io_write_bytes"].diff().fillna(0)
    df["io_total_rate"] = df["io_read_rate"] + df["io_write_rate"]
    
    # I/O operation counts
    df["io_read_ops_rate"] = df.groupby("pid")["io_read_count"].diff().fillna(0)
    df["io_write_ops_rate"] = df.groupby("pid")["io_write_count"].diff().fillna(0)
    
    # I/O burst detection
    df["io_write_burst"] = (
        df["io_write_rate"] > df.groupby("pid")["io_write_rate"].rolling(20, min_periods=1).quantile(0.95).reset_index(level=0, drop=True)
    ).astype(int)
    
    # Average bytes per I/O operation (efficiency metric)
    df["io_bytes_per_op"] = (df["io_total_rate"] + 1) / (df["io_read_ops_rate"] + df["io_write_ops_rate"] + 1)
    
    # ===== NETWORK & THREADING FEATURES =====
    print("    [7/8] Network and threading features")
    df["connections_rate"] = df.groupby("pid")["num_connections"].diff().fillna(0)
    df["threads_rate"] = df.groupby("pid")["num_threads"].diff().fillna(0)
    df["fds_rate"] = df.groupby("pid")["num_fds"].diff().fillna(0)
    
    # Context switch rates
    df["ctx_voluntary_rate"] = df.groupby("pid")["ctx_switches_voluntary"].diff().fillna(0)
    df["ctx_involuntary_rate"] = df.groupby("pid")["ctx_switches_involuntary"].diff().fillna(0)
    df["ctx_total_rate"] = df["ctx_voluntary_rate"] + df["ctx_involuntary_rate"]
    
    # High involuntary switches = CPU contention
    df["ctx_involuntary_ratio"] = df["ctx_involuntary_rate"] / (df["ctx_total_rate"] + 1)
    
    # ===== PROCESS BASELINE FEATURES =====
    print("    [8/8] Process baseline features")
    # Use first 5 observations per PID as "normal baseline"
    baseline = df.groupby('pid').head(5).groupby('pid').agg({
        'cpu': 'mean',
        'mem': 'mean',
        'io_total_rate': 'mean'
    }).rename(columns={
        'cpu': 'cpu_baseline',
        'mem': 'mem_baseline',
        'io_total_rate': 'io_baseline'
    })
    
    df = df.merge(baseline, on='pid', how='left')
    
    # Fill missing baselines with global median
    df['cpu_baseline'] = df['cpu_baseline'].fillna(df['cpu'].median())
    df['mem_baseline'] = df['mem_baseline'].fillna(df['mem'].median())
    df['io_baseline'] = df['io_baseline'].fillna(df['io_total_rate'].median())
    
    # Deviation from baseline
    df['cpu_dev_from_baseline'] = df['cpu'] - df['cpu_baseline']
    df['mem_dev_from_baseline'] = df['mem'] - df['mem_baseline']
    df['io_dev_from_baseline'] = df['io_total_rate'] - df['io_baseline']
    
    # Relative deviation (percentage)
    df['cpu_pct_dev_baseline'] = df['cpu_dev_from_baseline'] / (df['cpu_baseline'] + 1e-5)
    df['mem_pct_dev_baseline'] = df['mem_dev_from_baseline'] / (df['mem_baseline'] + 1e-5)
    
    # ===== BURST DETECTION =====
    df["cpu_burst"] = (
        df["delta_cpu"] > df.groupby("pid")["delta_cpu"].rolling(20, min_periods=1).quantile(0.95).reset_index(level=0, drop=True)
    ).astype(int)
    df["mem_burst"] = (
        df["delta_mem"] > df.groupby("pid")["delta_mem"].rolling(20, min_periods=1).quantile(0.95).reset_index(level=0, drop=True)
    ).astype(int)
    
    # ===== PROCESS AGE =====
    first_seen = df.groupby("pid")["timestamp"].transform("min")
    df["proc_age"] = (df["timestamp"] - first_seen).dt.total_seconds()
    
    # Resource usage per unit age (younger processes using lots = suspicious)
    df["cpu_per_age"] = df["cpu"] / (df["proc_age"] + 1)
    df["mem_per_age"] = df["mem"] / (df["proc_age"] + 1)
    df["io_per_age"] = df["io_total_rate"] / (df["proc_age"] + 1)
    
    # ===== PARENT-NORMALIZED FEATURES =====
    parent_stats = df.groupby("pid").agg({
        'cpu': 'last',
        'mem': 'last'
    })
    
    df["cpu_norm_parent"] = df.apply(
        lambda r: r.cpu / parent_stats.loc[r.ppid, 'cpu'] if r.ppid in parent_stats.index else 0,
        axis=1
    )
    df["mem_norm_parent"] = df.apply(
        lambda r: r.mem / parent_stats.loc[r.ppid, 'mem'] if r.ppid in parent_stats.index else 0,
        axis=1
    )
    
    # ===== FINAL CLEANUP =====
    print("    Cleaning inf/nan values")
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"    [✓] Feature engineering complete. Total features: {len(df.columns)}")
    
    return df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--buffer", type=int, default=2, 
                       help="Buffer seconds around anomaly windows (default: 2)")
    args = parser.parse_args()

    print("="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    # Load data
    df, df_events = load_data(args.session)
    
    # Label anomalies with exact windows
    df, periods, total = label_anomalies(df, df_events, buffer_seconds=args.buffer)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Save
    out = f"logs/{args.session}/labeled_dataset.csv"
    df.to_csv(out, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Session:                  {args.session}")
    print(f"Total samples:            {len(df)}")
    print(f"Total features:           {len(df.columns)}")
    print(f"Anomaly events:           {len(df_events)}")
    print(f"Anomaly periods:          {len(periods)}")
    print(f"Labeled anomaly rows:     {total}")
    print(f"\nClass distribution:")
    class_dist = df["label"].value_counts()
    for label, count in class_dist.items():
        label_name = "Anomaly" if label == 1 else "Normal"
        pct = count / len(df) * 100
        print(f"  {label_name} ({label}):  {count:8d} ({pct:5.2f}%)")
    
    print(f"\nOutput: {out}")
    print("="*60)


if __name__ == "__main__":
    main()

