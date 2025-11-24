#!/usr/bin/env python3
"""
combine_sessions.py
Combine multiple session datasets into a single training dataset.

This is useful for:
- Creating larger training datasets from multiple collection runs
- Combining sessions with different anomaly types
- Building a comprehensive dataset for better model generalization

Usage:
    python combine_sessions.py --sessions session1 session2 session3 --output combined
    
Author: Assistant
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def load_session(session_name):
    """Load a single session's labeled dataset."""
    dataset_path = f"logs/{session_name}/labeled_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"[!] WARNING: Dataset not found: {dataset_path}")
        return None
    
    print(f"[+] Loading session: {session_name}")
    # Use dtype specification and low_memory=False for large files
    df = pd.read_csv(dataset_path, low_memory=False)
    
    # Add session identifier column if not present
    if 'session' not in df.columns:
        df['session'] = session_name
    
    print(f"    Samples: {len(df)}")
    print(f"    Normal:  {(df['label']==0).sum()}")
    print(f"    Anomaly: {(df['label']==1).sum()}")
    
    return df


def combine_sessions(session_names, output_name, balance=False, sample_ratio=None):
    """
    Combine multiple session datasets into one.
    
    Args:
        session_names: List of session names to combine
        output_name: Name for the combined output session
        balance: If True, balance classes across sessions
        sample_ratio: If set, sample this ratio of data (e.g., 0.5 = 50%)
    """
    print("="*60)
    print("COMBINING MULTIPLE SESSIONS")
    print("="*60)
    print(f"Sessions to combine: {', '.join(session_names)}")
    print(f"Output name: {output_name}")
    print()
    
    # Load all sessions
    dfs = []
    for session in session_names:
        df = load_session(session)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("[!] ERROR: No valid datasets found!")
        return
    
    print(f"\n[+] Loaded {len(dfs)} sessions successfully")
    
    # Combine
    print("[+] Combining datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"    Combined total samples: {len(combined_df)}")
    print(f"    Combined normal:        {(combined_df['label']==0).sum()}")
    print(f"    Combined anomaly:       {(combined_df['label']==1).sum()}")
    
    # Optional: Balance classes
    if balance:
        print("\n[+] Balancing classes...")
        normal_count = (combined_df['label']==0).sum()
        anomaly_count = (combined_df['label']==1).sum()
        
        if normal_count > anomaly_count:
            # Downsample normal to match anomaly
            normal_df = combined_df[combined_df['label']==0].sample(n=anomaly_count, random_state=42)
            anomaly_df = combined_df[combined_df['label']==1]
        else:
            # Downsample anomaly to match normal
            normal_df = combined_df[combined_df['label']==0]
            anomaly_df = combined_df[combined_df['label']==1].sample(n=normal_count, random_state=42)
        
        combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"    After balancing: {len(combined_df)} samples")
        print(f"    Normal:  {(combined_df['label']==0).sum()}")
        print(f"    Anomaly: {(combined_df['label']==1).sum()}")
    
    # Optional: Sample subset
    if sample_ratio and 0 < sample_ratio < 1:
        print(f"\n[+] Sampling {sample_ratio*100}% of data...")
        combined_df = combined_df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"    After sampling: {len(combined_df)} samples")
    
    # Clean and validate
    print("\n[+] Cleaning data...")
    # Remove any inf/nan that might have crept in
    combined_df = combined_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Sort by timestamp if present
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save combined dataset
    output_dir = f"logs/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "labeled_dataset.csv")
    
    print(f"\n[+] Saving combined dataset to {output_path}")
    print(f"    This may take a while for large datasets...")
    
    # Save in chunks for large datasets to avoid memory issues
    chunk_size = 50000
    if len(combined_df) > chunk_size:
        print(f"    Using chunked write (chunks of {chunk_size} rows)")
        for i in range(0, len(combined_df), chunk_size):
            chunk = combined_df.iloc[i:i+chunk_size]
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            print(f"    Progress: {min(i+chunk_size, len(combined_df))}/{len(combined_df)} rows written")
    else:
        combined_df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("COMBINATION SUMMARY")
    print("="*60)
    print(f"Output location: {output_path}")
    print(f"Total samples:   {len(combined_df)}")
    print(f"Total features:  {len(combined_df.columns)}")
    print(f"\nClass distribution:")
    print(combined_df['label'].value_counts())
    pct_anomaly = (combined_df['label']==1).sum() / len(combined_df) * 100
    print(f"\nAnomaly rate: {pct_anomaly:.2f}%")
    
    # Per-session breakdown if session column exists
    if 'session' in combined_df.columns:
        print(f"\nPer-session breakdown:")
        session_stats = combined_df.groupby('session').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        session_stats.columns = ['Total', 'Anomalies', 'Anomaly_Rate']
        print(session_stats.to_string())
    
    print("\n" + "="*60)
    print(f"[✓] Combined dataset ready for training!")
    print(f"[✓] Use: python train_anomaly_model_advanced.py --session {output_name}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple session datasets into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine 3 sessions
  python combine_sessions.py --sessions sess1 sess2 sess3 --output combined
  
  # Combine with class balancing
  python combine_sessions.py --sessions sess1 sess2 --output balanced --balance
  
  # Combine and sample 50% of data (for faster training)
  python combine_sessions.py --sessions sess1 sess2 sess3 --output sampled --sample 0.5
        """
    )
    
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="List of session names to combine (space-separated)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Name for the combined output session"
    )
    
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes (downsample majority class to match minority)"
    )
    
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample this fraction of data (0.0-1.0, e.g., 0.5 for 50%%)"
    )
    
    args = parser.parse_args()
    
    # Validate sample ratio
    if args.sample is not None:
        if not 0 < args.sample <= 1:
            parser.error("--sample must be between 0 and 1")
    
    # Check for duplicate sessions
    if len(args.sessions) != len(set(args.sessions)):
        print("[!] WARNING: Duplicate session names detected, removing duplicates...")
        args.sessions = list(set(args.sessions))
    
    # Combine
    combine_sessions(
        session_names=args.sessions,
        output_name=args.output,
        balance=args.balance,
        sample_ratio=args.sample
    )


if __name__ == "__main__":
    main()

