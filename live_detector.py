#!/usr/bin/env python3
"""
live_detector.py
Real-time process anomaly detection system using trained ML model.

When an anomaly is detected, provides:
- Process ID and executable path
- Anomaly score and confidence
- Suspicious metrics that triggered detection
- Likely attack type indicators (process hollowing, crypto mining, etc.)
- Recommended actions

Usage:
    python live_detector.py --model models/tuned_model_all2.pkl --interval 2
    
Author: Rachel Soubier
Date: 2025-11
"""

import os
import sys
import argparse
import time
import joblib
import pandas as pd
import numpy as np
import psutil
import math
from datetime import datetime, timezone
from collections import deque, defaultdict
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Feature Collection
# =============================================================================

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
        return {'io_read_bytes': 0, 'io_write_bytes': 0, 'io_read_count': 0, 'io_write_count': 0}


def get_network_connections(proc):
    """Safely get network connection count."""
    try:
        return len(proc.connections())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


def get_file_descriptors(proc):
    """Safely get file descriptor count."""
    try:
        if hasattr(proc, 'num_fds'):
            return proc.num_fds()
        else:
            return len(proc.open_files())
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return 0


def get_context_switches(proc):
    """Safely get context switch counts."""
    try:
        ctx = proc.num_ctx_switches()
        return {'ctx_switches_voluntary': ctx.voluntary, 'ctx_switches_involuntary': ctx.involuntary}
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return {'ctx_switches_voluntary': 0, 'ctx_switches_involuntary': 0}


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


# =============================================================================
# Attack Type Classifier
# =============================================================================

class AttackTypeClassifier:
    """
    Identifies likely attack types based on suspicious metric patterns.
    """
    
    @staticmethod
    def classify_attack(suspicious_metrics, process_info):
        """
        Determine likely attack type based on suspicious metrics.
        
        Returns: (attack_type, confidence, description, recommendations)
        """
        attacks = []
        
        # Process Hollowing / Code Injection
        if any('child' in m.lower() for m in suspicious_metrics):
            if process_info.get('num_children', 0) > 5:
                attacks.append({
                    'type': 'Process Injection / Hollowing',
                    'confidence': 0.85,
                    'description': 'Excessive child process creation detected. May indicate process hollowing, DLL injection, or process spawning attacks.',
                    'indicators': [
                        f"Spawned {process_info.get('num_children', 0)} child processes",
                        'Abnormal process tree structure'
                    ],
                    'recommendations': [
                        'Inspect child processes with process explorer',
                        'Check for unsigned or suspicious DLLs',
                        'Examine process memory for injected code'
                    ]
                })
        
        # Crypto Mining
        cpu_high = any('cpu' in m.lower() and ('spike' in m.lower() or 'burst' in m.lower()) for m in suspicious_metrics)
        if cpu_high and process_info.get('cpu', 0) > 70:
            attacks.append({
                'type': 'Cryptocurrency Mining',
                'confidence': 0.80,
                'description': 'Sustained high CPU usage detected. May indicate cryptocurrency mining malware.',
                'indicators': [
                    f"CPU usage: {process_info.get('cpu', 0):.1f}%",
                    'Prolonged compute-intensive activity',
                    f"Process age: {process_info.get('proc_age', 0):.0f}s"
                ],
                'recommendations': [
                    'Check process name against known miners',
                    'Monitor network connections to mining pools',
                    'Examine process for packed/obfuscated code'
                ]
            })
        
        # Data Exfiltration
        io_high = any('io' in m.lower() for m in suspicious_metrics)
        net_high = any('connection' in m.lower() or 'network' in m.lower() for m in suspicious_metrics)
        if io_high and net_high:
            attacks.append({
                'type': 'Data Exfiltration',
                'confidence': 0.75,
                'description': 'High I/O combined with network activity. May indicate data theft or exfiltration.',
                'indicators': [
                    f"I/O write rate: {process_info.get('io_write_rate', 0):.0f} bytes/sec",
                    f"Network connections: {process_info.get('num_connections', 0)}",
                    'Simultaneous disk and network activity'
                ],
                'recommendations': [
                    'Capture network traffic for analysis',
                    'Check destination IPs/domains',
                    'Identify files being accessed',
                    'Review process for data staging behavior'
                ]
            })
        
        # Ransomware
        io_write_very_high = process_info.get('io_write_rate', 0) > 5000000  # >5MB/sec
        fd_high = process_info.get('num_fds', 0) > 100
        if io_write_very_high or (io_high and fd_high):
            attacks.append({
                'type': 'Ransomware',
                'confidence': 0.70,
                'description': 'Rapid file modification activity. May indicate ransomware encryption.',
                'indicators': [
                    f"Extremely high I/O write rate: {process_info.get('io_write_rate', 0):.0f} bytes/sec",
                    f"Open file descriptors: {process_info.get('num_fds', 0)}",
                    'Mass file access pattern'
                ],
                'recommendations': [
                    '⚠️ URGENT: Isolate system immediately',
                    'Kill process if confirmed malicious',
                    'Check for ransom notes or file extensions',
                    'Restore from backup if encrypted'
                ]
            })
        
        # Memory Scraping / Credential Theft
        mem_high = any('mem' in m.lower() for m in suspicious_metrics)
        if mem_high and process_info.get('mem', 0) > 60:
            attacks.append({
                'type': 'Memory Scraping',
                'confidence': 0.65,
                'description': 'Abnormal memory usage. May indicate credential dumping or memory scraping.',
                'indicators': [
                    f"Memory usage: {process_info.get('mem', 0):.1f}%",
                    'Unusual memory access patterns'
                ],
                'recommendations': [
                    'Check for LSASS access attempts',
                    'Review process privileges',
                    'Look for memory dump tools (Mimikatz, etc.)'
                ]
            })
        
        # Fork Bomb / Resource Exhaustion
        if process_info.get('num_threads', 0) > 100 or process_info.get('num_children', 0) > 20:
            attacks.append({
                'type': 'Fork Bomb / DoS',
                'confidence': 0.80,
                'description': 'Excessive resource creation. May indicate fork bomb or denial-of-service attack.',
                'indicators': [
                    f"Thread count: {process_info.get('num_threads', 0)}",
                    f"Child processes: {process_info.get('num_children', 0)}",
                    'Rapid resource exhaustion'
                ],
                'recommendations': [
                    'Kill process immediately',
                    'Set process/user resource limits',
                    'Check for recursive process spawning'
                ]
            })
        
        # Network Scanning / C2 Communication
        if process_info.get('num_connections', 0) > 10:
            attacks.append({
                'type': 'Network Scanning / C2',
                'confidence': 0.70,
                'description': 'Excessive network connections. May indicate port scanning or C2 communication.',
                'indicators': [
                    f"Active connections: {process_info.get('num_connections', 0)}",
                    'Unusual network behavior'
                ],
                'recommendations': [
                    'Capture connection details (IPs, ports)',
                    'Check against threat intelligence feeds',
                    'Monitor for beaconing behavior',
                    'Inspect for tunneling protocols'
                ]
            })
        
        # File Descriptor Leak / Resource Exhaustion
        if process_info.get('num_fds', 0) > 200:
            attacks.append({
                'type': 'File Descriptor Leak',
                'confidence': 0.60,
                'description': 'Excessive open file handles. May indicate FD leak or file system attack.',
                'indicators': [
                    f"Open file descriptors: {process_info.get('num_fds', 0)}",
                    'Unclosed file handles'
                ],
                'recommendations': [
                    'List open files with lsof',
                    'Check for file system filling',
                    'Monitor for inode exhaustion'
                ]
            })
        
        # If no specific attack identified, return generic anomaly
        if not attacks:
            attacks.append({
                'type': 'Unknown Anomaly',
                'confidence': 0.50,
                'description': 'Anomalous behavior detected but specific attack type unclear.',
                'indicators': ['Multiple suspicious metrics triggered'],
                'recommendations': [
                    'Investigate process manually',
                    'Check process reputation/signature',
                    'Review full process behavior timeline'
                ]
            })
        
        # Return highest confidence attack
        attacks.sort(key=lambda x: x['confidence'], reverse=True)
        return attacks[0]


# =============================================================================
# Live Detector
# =============================================================================

class LiveProcessDetector:
    """
    Real-time process anomaly detection system.
    """
    
    def __init__(self, model_path, scaler_path=None, features_path=None, threshold=0.7):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained ML model
            scaler_path: Path to feature scaler (optional)
            features_path: Path to feature names file (optional)
            threshold: Anomaly probability threshold (default 0.7)
        """
        print(f"[+] Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            print(f"[+] Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        
        self.feature_names = None
        if features_path and os.path.exists(features_path):
            print(f"[+] Loading feature names from: {features_path}")
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]
        
        self.threshold = threshold
        
        # Process history for feature calculation
        self.process_history = defaultdict(lambda: deque(maxlen=20))
        self.first_seen = {}
        
        # Alert tracking (avoid duplicate alerts)
        self.alerted_pids = set()
        
        # Initialize CPU measurements
        for proc in psutil.process_iter(['pid']):
            try:
                proc.cpu_percent(None)
            except:
                pass
        
        print(f"[+] Detector initialized with threshold: {threshold}")
        print(f"[+] Ready to monitor processes...\n")
    
    def collect_process_metrics(self):
        """Collect current metrics for all processes."""
        all_procs = list(psutil.process_iter(['pid', 'ppid', 'exe', 'name']))
        
        # Build parent→children map
        child_map = {}
        for p in all_procs:
            try:
                pid = p.info['pid']
                ppid = p.info.get('ppid')
                child_map.setdefault(ppid, []).append(p)
            except:
                continue
        
        metrics_list = []
        
        for proc in all_procs:
            try:
                pid = proc.pid
                ppid = proc.ppid()
                exe = proc.exe()
                name = proc.name()
                
                # Basic metrics
                cpu = proc.cpu_percent(None)
                mem = proc.memory_percent()
                
                # Children
                children = child_map.get(pid, [])
                num_children = len(children)
                child_exe_names = [c.info.get('exe', '') for c in children]
                child_entropy = shannon_entropy(child_exe_names)
                
                # I/O
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
                
                # Process age
                if pid not in self.first_seen:
                    self.first_seen[pid] = time.time()
                proc_age = time.time() - self.first_seen[pid]
                
                metrics = {
                    'pid': pid,
                    'ppid': ppid,
                    'exe': exe,
                    'name': name,
                    'cpu': cpu,
                    'mem': mem,
                    'num_children': num_children,
                    'child_entropy': child_entropy,
                    'io_read_bytes': io_stats['io_read_bytes'],
                    'io_write_bytes': io_stats['io_write_bytes'],
                    'io_read_count': io_stats['io_read_count'],
                    'io_write_count': io_stats['io_write_count'],
                    'num_connections': num_connections,
                    'num_threads': num_threads,
                    'num_fds': num_fds,
                    'ctx_switches_voluntary': ctx_stats['ctx_switches_voluntary'],
                    'ctx_switches_involuntary': ctx_stats['ctx_switches_involuntary'],
                    'status': status,
                    'proc_age': proc_age
                }
                
                # Add to history
                self.process_history[pid].append(metrics)
                
                metrics_list.append(metrics)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                continue
        
        return metrics_list
    
    def compute_features(self, metrics_list):
        """Compute features from raw metrics (must match training features)."""
        df = pd.DataFrame(metrics_list)
        
        if df.empty:
            return df
        
        # Sort by PID
        df = df.sort_values('pid').reset_index(drop=True)
        
        # Initialize all feature columns with zeros
        feature_cols = [
            'io_read_rate', 'io_write_rate', 'io_total_rate', 'io_read_ops_rate', 'io_write_ops_rate',
            'delta_cpu', 'delta_mem', 'children_rate', 'connections_rate', 'threads_rate', 'fds_rate',
            'ctx_voluntary_rate', 'ctx_involuntary_rate', 'ctx_total_rate', 'ctx_involuntary_ratio',
            'cpu_mean_5', 'cpu_std_5', 'mem_mean_5', 'mem_std_5', 'cpu_mean_10', 'cpu_std_10',
            'mem_mean_10', 'mem_std_10', 'cpu_rate', 'mem_rate', 'cpu_ema', 'mem_ema',
            'cpu_spike', 'mem_spike', 'has_children', 'cpu_roll_std', 'mem_roll_std',
            'cpu_burst', 'mem_burst', 'io_write_burst', 'io_bytes_per_op',
            'cpu_baseline', 'mem_baseline', 'io_baseline', 'cpu_dev_from_baseline',
            'mem_dev_from_baseline', 'io_dev_from_baseline', 'cpu_pct_dev_baseline',
            'mem_pct_dev_baseline', 'cpu_per_age', 'mem_per_age', 'io_per_age',
            'cpu_norm_parent', 'mem_norm_parent', 'cpu_max_10', 'cpu_min_10',
            'cpu_max_20', 'mem_max_10', 'cpu_trend', 'mem_trend', 'cpu_acceleration',
            'cpu_volatility', 'mem_volatility', 'cpu_pct_rank', 'mem_pct_rank',
            'cpu_mem_product', 'cpu_mem_ratio', 'resource_pressure', 'cpu_jump', 'mem_jump'
        ]
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Compute temporal features from history
        for pid in df['pid'].unique():
            history = list(self.process_history[pid])
            if len(history) < 2:
                continue
            
            idx = df[df['pid'] == pid].index
            
            # Calculate rates from history
            if len(history) >= 2:
                prev = history[-2]
                curr = history[-1]
                
                df.loc[idx, 'io_read_rate'] = max(0, curr['io_read_bytes'] - prev['io_read_bytes'])
                df.loc[idx, 'io_write_rate'] = max(0, curr['io_write_bytes'] - prev['io_write_bytes'])
                df.loc[idx, 'io_total_rate'] = df.loc[idx, 'io_read_rate'] + df.loc[idx, 'io_write_rate']
                df.loc[idx, 'io_read_ops_rate'] = max(0, curr['io_read_count'] - prev['io_read_count'])
                df.loc[idx, 'io_write_ops_rate'] = max(0, curr['io_write_count'] - prev['io_write_count'])
                
                df.loc[idx, 'delta_cpu'] = curr['cpu'] - prev['cpu']
                df.loc[idx, 'delta_mem'] = curr['mem'] - prev['mem']
                df.loc[idx, 'children_rate'] = curr['num_children'] - prev['num_children']
                df.loc[idx, 'connections_rate'] = curr['num_connections'] - prev['num_connections']
                df.loc[idx, 'threads_rate'] = curr['num_threads'] - prev['num_threads']
                df.loc[idx, 'fds_rate'] = curr['num_fds'] - prev['num_fds']
                
                df.loc[idx, 'ctx_voluntary_rate'] = max(0, curr['ctx_switches_voluntary'] - prev['ctx_switches_voluntary'])
                df.loc[idx, 'ctx_involuntary_rate'] = max(0, curr['ctx_switches_involuntary'] - prev['ctx_switches_involuntary'])
                df.loc[idx, 'ctx_total_rate'] = df.loc[idx, 'ctx_voluntary_rate'] + df.loc[idx, 'ctx_involuntary_rate']
                
                total = df.loc[idx, 'ctx_total_rate'].values[0]
                if total > 0:
                    df.loc[idx, 'ctx_involuntary_ratio'] = df.loc[idx, 'ctx_involuntary_rate'].values[0] / total
            
            # Rolling statistics (5-sample window)
            if len(history) >= 5:
                cpu_vals = [h['cpu'] for h in history[-5:]]
                mem_vals = [h['mem'] for h in history[-5:]]
                
                df.loc[idx, 'cpu_mean_5'] = np.mean(cpu_vals)
                df.loc[idx, 'cpu_std_5'] = np.std(cpu_vals)
                df.loc[idx, 'mem_mean_5'] = np.mean(mem_vals)
                df.loc[idx, 'mem_std_5'] = np.std(mem_vals)
                df.loc[idx, 'cpu_roll_std'] = np.std(cpu_vals)
                df.loc[idx, 'mem_roll_std'] = np.std(mem_vals)
                
                # Rates relative to mean/std
                cpu_mean = df.loc[idx, 'cpu_mean_5'].values[0]
                cpu_std = df.loc[idx, 'cpu_std_5'].values[0]
                mem_mean = df.loc[idx, 'mem_mean_5'].values[0]
                mem_std = df.loc[idx, 'mem_std_5'].values[0]
                
                curr_cpu = history[-1]['cpu']
                curr_mem = history[-1]['mem']
                
                df.loc[idx, 'cpu_rate'] = (curr_cpu - cpu_mean) / (cpu_std + 1e-5)
                df.loc[idx, 'mem_rate'] = (curr_mem - mem_mean) / (mem_std + 1e-5)
                
                # Spikes
                if cpu_mean > 0:
                    df.loc[idx, 'cpu_spike'] = curr_cpu / cpu_mean
                if mem_mean > 0:
                    df.loc[idx, 'mem_spike'] = curr_mem / mem_mean
                
                # Max/min
                df.loc[idx, 'cpu_max_10'] = max(cpu_vals)
                df.loc[idx, 'cpu_min_10'] = min(cpu_vals)
                df.loc[idx, 'mem_max_10'] = max(mem_vals)
            
            # 10-sample window
            if len(history) >= 10:
                cpu_vals_10 = [h['cpu'] for h in history[-10:]]
                mem_vals_10 = [h['mem'] for h in history[-10:]]
                
                df.loc[idx, 'cpu_mean_10'] = np.mean(cpu_vals_10)
                df.loc[idx, 'cpu_std_10'] = np.std(cpu_vals_10)
                df.loc[idx, 'mem_mean_10'] = np.mean(mem_vals_10)
                df.loc[idx, 'mem_std_10'] = np.std(mem_vals_10)
                df.loc[idx, 'cpu_volatility'] = np.std(cpu_vals_10)
                df.loc[idx, 'mem_volatility'] = np.std(mem_vals_10)
                
                # EMA approximation
                df.loc[idx, 'cpu_ema'] = np.mean(cpu_vals_10)
                df.loc[idx, 'mem_ema'] = np.mean(mem_vals_10)
            
            # Trends (5-sample lookback)
            if len(history) >= 5:
                df.loc[idx, 'cpu_trend'] = history[-1]['cpu'] - history[-5]['cpu']
                df.loc[idx, 'mem_trend'] = history[-1]['mem'] - history[-5]['mem']
            
            # Baseline (first 5 observations)
            if len(history) >= 5:
                baseline_cpu = np.mean([h['cpu'] for h in history[:5]])
                baseline_mem = np.mean([h['mem'] for h in history[:5]])
                baseline_io = np.mean([h['io_write_bytes'] for h in history[:5]])
                
                df.loc[idx, 'cpu_baseline'] = baseline_cpu
                df.loc[idx, 'mem_baseline'] = baseline_mem
                df.loc[idx, 'io_baseline'] = baseline_io
                
                curr_cpu = history[-1]['cpu']
                curr_mem = history[-1]['mem']
                curr_io = history[-1]['io_write_bytes']
                
                df.loc[idx, 'cpu_dev_from_baseline'] = curr_cpu - baseline_cpu
                df.loc[idx, 'mem_dev_from_baseline'] = curr_mem - baseline_mem
                df.loc[idx, 'io_dev_from_baseline'] = curr_io - baseline_io
                
                if baseline_cpu > 0:
                    df.loc[idx, 'cpu_pct_dev_baseline'] = (curr_cpu - baseline_cpu) / baseline_cpu
                if baseline_mem > 0:
                    df.loc[idx, 'mem_pct_dev_baseline'] = (curr_mem - baseline_mem) / baseline_mem
        
        # Additional features
        df['has_children'] = (df['num_children'] > 0).astype(int)
        df['cpu_mem_product'] = df['cpu'] * df['mem']
        df['cpu_mem_ratio'] = df['cpu'] / (df['mem'] + 1e-6)
        df['resource_pressure'] = df['cpu'] + df['mem']
        
        # Per-age features
        df['cpu_per_age'] = df['cpu'] / (df['proc_age'] + 1)
        df['mem_per_age'] = df['mem'] / (df['proc_age'] + 1)
        df['io_per_age'] = df['io_write_rate'] / (df['proc_age'] + 1)
        
        # I/O bytes per operation
        total_io_rate = df['io_read_rate'] + df['io_write_rate']
        total_ops_rate = df['io_read_ops_rate'] + df['io_write_ops_rate']
        df['io_bytes_per_op'] = (total_io_rate + 1) / (total_ops_rate + 1)
        
        # Burst detection (simplified - use current vs history percentile)
        for pid in df['pid'].unique():
            history = list(self.process_history[pid])
            if len(history) >= 10:
                idx = df[df['pid'] == pid].index
                
                cpu_vals = [h['cpu'] for h in history]
                mem_vals = [h['mem'] for h in history]
                io_vals = [h['io_write_bytes'] for h in history]
                
                cpu_95 = np.percentile(cpu_vals, 95) if cpu_vals else 0
                mem_95 = np.percentile(mem_vals, 95) if mem_vals else 0
                io_95 = np.percentile(io_vals, 95) if io_vals else 0
                
                curr_cpu = history[-1]['cpu']
                curr_mem = history[-1]['mem']
                curr_io_rate = df.loc[idx, 'io_write_rate'].values[0] if len(df.loc[idx]) > 0 else 0
                
                df.loc[idx, 'cpu_burst'] = int(curr_cpu > cpu_95)
                df.loc[idx, 'mem_burst'] = int(curr_mem > mem_95)
                df.loc[idx, 'io_write_burst'] = int(curr_io_rate > io_95)
                df.loc[idx, 'cpu_jump'] = int(curr_cpu > cpu_95)
                df.loc[idx, 'mem_jump'] = int(curr_mem > mem_95)
        
        # Fill NaN values
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        return df
    
    def detect_anomalies(self, df):
        """Run ML model on features and detect anomalies."""
        if df.empty:
            return []
        
        # Save metadata columns
        metadata_cols = ['pid', 'ppid', 'exe', 'name', 'status']
        metadata = df[metadata_cols].copy()
        
        # Extract features (exclude metadata)
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        
        # Ensure all features are numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Filter to only features the model expects (in the correct order)
        if self.feature_names:
            # Use only features that exist in both the current data and training
            available_features = [f for f in self.feature_names if f in feature_cols]
            missing_features = [f for f in self.feature_names if f not in feature_cols]
            
            if missing_features:
                # Add missing features with zeros
                for f in missing_features:
                    df[f] = 0.0
                available_features = self.feature_names
            
            X = df[available_features]
        else:
            X = df[feature_cols]
        
        # Scale if scaler available
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predict
        try:
            y_proba = self.model.predict_proba(X)[:, 1]
        except:
            # Fallback if predict_proba not available
            y_pred = self.model.predict(X)
            y_proba = y_pred.astype(float)
        
        # Find anomalies above threshold
        anomalies = []
        for idx, prob in enumerate(y_proba):
            if prob >= self.threshold:
                pid = metadata.iloc[idx]['pid']
                
                # Skip if already alerted
                if pid in self.alerted_pids:
                    continue
                
                self.alerted_pids.add(pid)
                
                # Get suspicious metrics (top contributing features)
                suspicious_metrics = self.identify_suspicious_metrics(X.iloc[idx], X.columns.tolist())
                
                # Combine metadata with full process info
                process_info = df.iloc[idx].to_dict()
                process_info.update(metadata.iloc[idx].to_dict())
                
                anomalies.append({
                    'pid': pid,
                    'ppid': metadata.iloc[idx]['ppid'],
                    'exe': metadata.iloc[idx]['exe'],
                    'name': metadata.iloc[idx]['name'],
                    'anomaly_score': prob,
                    'suspicious_metrics': suspicious_metrics,
                    'process_info': process_info
                })
        
        return anomalies
    
    def identify_suspicious_metrics(self, feature_values, feature_cols):
        """Identify which metrics are most suspicious."""
        suspicious = []
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Weight features by importance and value
            weighted = []
            for i, col in enumerate(feature_cols):
                val = feature_values.iloc[i] if hasattr(feature_values, 'iloc') else feature_values[i]
                importance = importances[i] if i < len(importances) else 0
                
                # Normalize value (higher absolute values more suspicious)
                normalized_val = abs(val) / (abs(val) + 1)
                
                weighted.append((col, val, importance * normalized_val))
            
            # Sort by weighted score
            weighted.sort(key=lambda x: x[2], reverse=True)
            
            # Take top 5
            for col, val, score in weighted[:5]:
                if score > 0.01:  # Only include meaningful contributions
                    suspicious.append(f"{col}={val:.2f}")
        else:
            # Fallback: just show high values
            for col in feature_cols:
                val = feature_values[col]
                if abs(val) > 10:  # Arbitrary threshold
                    suspicious.append(f"{col}={val:.2f}")
        
        return suspicious
    
    def print_alert(self, anomaly):
        """Print formatted alert for detected anomaly."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*80)
        print(f"!!! ANOMALY DETECTED - {timestamp}")
        print("="*80)
        
        print(f"\n[+] Process Information:")
        print(f"   PID:        {anomaly['pid']}")
        print(f"   Parent PID: {anomaly['ppid']}")
        print(f"   Name:       {anomaly['name']}")
        print(f"   Executable: {anomaly['exe']}")
        
        print(f"\n[+] Detection Metrics:")
        print(f"   Anomaly Score:  {anomaly['anomaly_score']:.2%} (threshold: {self.threshold:.0%})")
        print(f"   Confidence:     {'HIGH' if anomaly['anomaly_score'] > 0.9 else 'MEDIUM' if anomaly['anomaly_score'] > 0.8 else 'LOW'}")
        
        print(f"\n[+]  Suspicious Metrics:")
        for metric in anomaly['suspicious_metrics']:
            print(f"   • {metric}")
        
        # Classify attack type
        attack_info = AttackTypeClassifier.classify_attack(
            anomaly['suspicious_metrics'],
            anomaly['process_info']
        )
        
        print(f"\n[+] Likely Attack Type: {attack_info['type']}")
        print(f"   Confidence: {attack_info['confidence']:.0%}")
        print(f"   {attack_info['description']}")
        
        print(f"\n[+] Indicators:")
        for indicator in attack_info['indicators']:
            print(f"   • {indicator}")
        
        print(f"\n[+] Recommended Actions:")
        for i, rec in enumerate(attack_info['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80 + "\n")
    
    def run(self, interval=2.0):
        """Run live detection loop."""
        print(f"[*] Starting live detection (interval: {interval}s)")
        print(f"[*] Monitoring all system processes...")
        print(f"[*] Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Collect metrics
                metrics = self.collect_process_metrics()
                
                # Compute features
                df = self.compute_features(metrics)
                
                # Detect anomalies
                anomalies = self.detect_anomalies(df)
                
                # Alert on anomalies
                for anomaly in anomalies:
                    self.print_alert(anomaly)
                
                # Sleep
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[*] Detection stopped by user")
            print(f"[*] Total anomalies detected: {len(self.alerted_pids)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live process anomaly detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with tuned model
  python live_detector.py --model models/tuned_model_all2.pkl
  
  # With custom threshold (more sensitive)
  python live_detector.py --model models/tuned_model_all2.pkl --threshold 0.6
  
  # With all components
  python live_detector.py --model models/tuned_model_all2.pkl \\
                          --scaler models/tuned_scaler_all2.pkl \\
                          --features models/tuned_features_all2.txt \\
                          --interval 1
        """
    )
    
    parser.add_argument("--model", required=True, help="Path to trained ML model")
    parser.add_argument("--scaler", default=None, help="Path to feature scaler")
    parser.add_argument("--features", default=None, help="Path to feature names file")
    parser.add_argument("--threshold", type=float, default=0.7, 
                       help="Anomaly probability threshold (0.0-1.0, default: 0.7)")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Sampling interval in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"[!] ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    # Auto-detect scaler and features if not provided
    model_dir = os.path.dirname(args.model)
    model_base = os.path.basename(args.model).replace('.pkl', '')
    
    if args.scaler is None:
        scaler_path = os.path.join(model_dir, model_base.replace('model', 'scaler') + '.pkl')
        if os.path.exists(scaler_path):
            args.scaler = scaler_path
    
    if args.features is None:
        features_path = os.path.join(model_dir, model_base.replace('model', 'features') + '.txt')
        if os.path.exists(features_path):
            args.features = features_path
    
    # Create and run detector
    detector = LiveProcessDetector(
        model_path=args.model,
        scaler_path=args.scaler,
        features_path=args.features,
        threshold=args.threshold
    )
    
    detector.run(interval=args.interval)


if __name__ == "__main__":
    main()

