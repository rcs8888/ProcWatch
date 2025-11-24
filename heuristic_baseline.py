#!/usr/bin/env python3
"""
heuristic_baseline.py
Rule-based heuristic baseline for process anomaly detection.

This represents traditional threshold-based detection methods used before ML.
Useful for demonstrating ML superiority in your paper.

Rules implemented:
1. High CPU usage (>80%)
2. High memory usage (>70%)
3. Excessive I/O activity (top 5% of write rates)
4. Rapid process spawning (children_rate > threshold)
5. High network connections (>10 concurrent)
6. High thread count (>50)
7. High file descriptor count (>100)
8. CPU/Memory spikes (>3x moving average)

Usage:
    python heuristic_baseline.py --session all2
    
Author: Assistant
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(session):
    """Load and prepare dataset."""
    dataset_path = f"logs/{session}/labeled_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"[+] Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    
    # Ensure numeric columns
    numeric_cols = ['cpu', 'mem', 'io_write_rate', 'children_rate', 
                    'num_connections', 'num_threads', 'num_fds',
                    'cpu_spike', 'mem_spike', 'cpu_burst', 'mem_burst']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
    
    return df


class HeuristicDetector:
    """
    Traditional rule-based anomaly detector.
    
    This represents the baseline approach used before ML methods.
    Uses fixed thresholds on individual metrics.
    """
    
    def __init__(self):
        # Configurable thresholds
        self.cpu_threshold = 80.0  # CPU usage > 80%
        self.mem_threshold = 70.0  # Memory usage > 70%
        self.io_write_percentile = 95  # Top 5% of I/O writes
        self.children_rate_threshold = 3  # Spawning >3 children rapidly
        self.connections_threshold = 10  # >10 network connections
        self.threads_threshold = 50  # >50 threads
        self.fds_threshold = 100  # >100 file descriptors
        self.cpu_spike_threshold = 3.0  # 3x average CPU
        self.mem_spike_threshold = 3.0  # 3x average memory
        
        # Weights for combining rules (tunable)
        self.rule_weights = {
            'high_cpu': 1.0,
            'high_mem': 1.0,
            'high_io': 1.5,
            'rapid_spawning': 2.0,
            'high_connections': 1.5,
            'high_threads': 1.0,
            'high_fds': 1.5,
            'cpu_spike': 2.0,
            'mem_spike': 1.5,
            'cpu_burst': 1.5,
            'mem_burst': 1.5
        }
        
        self.anomaly_threshold = 3.0  # Need weighted score >3 to flag as anomaly
        self.io_threshold = None  # Computed from data
    
    def fit(self, X_train, y_train=None):
        """
        'Fit' the heuristic detector (just compute data-dependent thresholds).
        """
        print("\n[+] Fitting heuristic detector (computing thresholds)...")
        
        # Compute I/O threshold from training data
        if 'io_write_rate' in X_train.columns:
            self.io_threshold = np.percentile(
                X_train['io_write_rate'].replace([np.inf, -np.inf], 0),
                self.io_write_percentile
            )
        else:
            self.io_threshold = 0
        
        print(f"    CPU threshold: >{self.cpu_threshold}%")
        print(f"    Memory threshold: >{self.mem_threshold}%")
        print(f"    I/O write threshold: >{self.io_threshold:.2f} bytes/sec")
        print(f"    Children rate threshold: >{self.children_rate_threshold}")
        print(f"    Connections threshold: >{self.connections_threshold}")
        print(f"    Threads threshold: >{self.threads_threshold}")
        print(f"    FDs threshold: >{self.fds_threshold}")
        print(f"    Anomaly score threshold: >{self.anomaly_threshold}")
        
        return self
    
    def predict(self, X):
        """Apply heuristic rules to predict anomalies."""
        scores = np.zeros(len(X))
        
        # Rule 1: High CPU
        if 'cpu' in X.columns:
            scores += (X['cpu'] > self.cpu_threshold).astype(int) * self.rule_weights['high_cpu']
        
        # Rule 2: High memory
        if 'mem' in X.columns:
            scores += (X['mem'] > self.mem_threshold).astype(int) * self.rule_weights['high_mem']
        
        # Rule 3: High I/O
        if 'io_write_rate' in X.columns:
            scores += (X['io_write_rate'] > self.io_threshold).astype(int) * self.rule_weights['high_io']
        
        # Rule 4: Rapid process spawning
        if 'children_rate' in X.columns:
            scores += (X['children_rate'] > self.children_rate_threshold).astype(int) * self.rule_weights['rapid_spawning']
        
        # Rule 5: High network connections
        if 'num_connections' in X.columns:
            scores += (X['num_connections'] > self.connections_threshold).astype(int) * self.rule_weights['high_connections']
        
        # Rule 6: High thread count
        if 'num_threads' in X.columns:
            scores += (X['num_threads'] > self.threads_threshold).astype(int) * self.rule_weights['high_threads']
        
        # Rule 7: High file descriptors
        if 'num_fds' in X.columns:
            scores += (X['num_fds'] > self.fds_threshold).astype(int) * self.rule_weights['high_fds']
        
        # Rule 8: CPU spike
        if 'cpu_spike' in X.columns:
            scores += (X['cpu_spike'] > self.cpu_spike_threshold).astype(int) * self.rule_weights['cpu_spike']
        
        # Rule 9: Memory spike
        if 'mem_spike' in X.columns:
            scores += (X['mem_spike'] > self.mem_spike_threshold).astype(int) * self.rule_weights['mem_spike']
        
        # Rule 10: CPU burst
        if 'cpu_burst' in X.columns:
            scores += X['cpu_burst'].astype(int) * self.rule_weights['cpu_burst']
        
        # Rule 11: Memory burst
        if 'mem_burst' in X.columns:
            scores += X['mem_burst'].astype(int) * self.rule_weights['mem_burst']
        
        # Final prediction: anomaly if score exceeds threshold
        predictions = (scores > self.anomaly_threshold).astype(int)
        
        return predictions, scores
    
    def predict_proba(self, X):
        """Return probability-like scores for compatibility with ML metrics."""
        _, scores = self.predict(X)
        
        # Normalize scores to [0, 1] range
        max_possible_score = sum(self.rule_weights.values())
        proba_anomaly = np.clip(scores / max_possible_score, 0, 1)
        
        # Return as 2D array: [prob_normal, prob_anomaly]
        proba_normal = 1 - proba_anomaly
        return np.column_stack([proba_normal, proba_anomaly])


def evaluate_heuristic(detector, X_test, y_test):
    """Evaluate heuristic detector with same metrics as ML models."""
    print("\n" + "="*60)
    print("HEURISTIC BASELINE EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred, scores = detector.predict(X_test)
    y_proba_full = detector.predict_proba(X_test)
    y_proba = y_proba_full[:, 1]  # Probability of anomaly
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {tn:6d}  {fp:6d}")
    print(f"       Anomaly  {fn:6d}  {tp:6d}")
    
    # Advanced metrics
    print("\nAdvanced Metrics:")
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score:        {roc_auc:.4f}")
    except:
        print(f"ROC-AUC Score:        N/A")
        roc_auc = 0.0
    
    try:
        avg_precision = average_precision_score(y_test, y_proba)
        print(f"Average Precision:    {avg_precision:.4f}")
    except:
        print(f"Average Precision:    N/A")
        avg_precision = 0.0
    
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"F1 Score:             {f1:.4f}")
    print(f"Matthews Corr Coef:   {mcc:.4f}")
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"False Positive Rate:  {fpr:.4f}")
    print(f"False Negative Rate:  {fnr:.4f}")
    
    # Rule triggering statistics
    print("\n" + "="*60)
    print("RULE TRIGGERING STATISTICS")
    print("="*60)
    
    print(f"\nSamples flagged by each rule:")
    print(f"  High CPU (>{detector.cpu_threshold}%): {(X_test['cpu'] > detector.cpu_threshold).sum()}")
    print(f"  High Memory (>{detector.mem_threshold}%): {(X_test['mem'] > detector.mem_threshold).sum()}")
    print(f"  High I/O: {(X_test['io_write_rate'] > detector.io_threshold).sum()}")
    print(f"  Rapid Spawning: {(X_test['children_rate'] > detector.children_rate_threshold).sum()}")
    print(f"  High Connections: {(X_test['num_connections'] > detector.connections_threshold).sum()}")
    print(f"  High Threads: {(X_test['num_threads'] > detector.threads_threshold).sum()}")
    print(f"  High FDs: {(X_test['num_fds'] > detector.fds_threshold).sum()}")
    
    print(f"\nAverage anomaly score: {scores.mean():.2f}")
    print(f"Max anomaly score: {scores.max():.2f}")
    
    return {
        'model': 'Heuristic_Baseline',
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1': f1,
        'mcc': mcc,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def main():
    parser = argparse.ArgumentParser(description="Heuristic baseline detector")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()
    
    print("="*60)
    print("HEURISTIC BASELINE DETECTOR")
    print("="*60)
    print(f"Session: {args.session}")
    print()
    
    # Load data
    df = load_data(args.session)
    print(f"[+] Loaded {len(df)} samples")
    print(f"[+] Class distribution:")
    print(df['label'].value_counts())
    print()
    
    # Prepare features
    exclude = ['timestamp', 'session_id', 'pid', 'ppid', 'exe', 'label', 'session']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols]
    y = df['label']
    
    # Train/test split (same as ML models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"[+] Train set: {len(X_train)} samples")
    print(f"[+] Test set:  {len(X_test)} samples")
    
    # Create and fit heuristic detector
    detector = HeuristicDetector()
    detector.fit(X_train, y_train)
    
    # Evaluate
    results = evaluate_heuristic(detector, X_test, y_test)
    
    # Save detector and results
    os.makedirs(args.output_dir, exist_ok=True)
    
    detector_path = os.path.join(args.output_dir, f"heuristic_baseline_{args.session}.pkl")
    joblib.dump(detector, detector_path)
    print(f"\n[+] Saved detector: {detector_path}")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f"heuristic_results_{args.session}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[+] Saved results: {results_path}")
    
    # Save predictions for visualization script
    y_pred, scores = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)[:, 1]
    
    predictions_df = pd.DataFrame({
        'y_test': y_test.values,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'anomaly_score': scores
    })
    
    predictions_path = os.path.join(args.output_dir, f"heuristic_predictions_{args.session}.pkl")
    joblib.dump(predictions_df, predictions_path)
    print(f"[+] Saved predictions: {predictions_path}")
    
    print("\n" + "="*60)
    print("HEURISTIC BASELINE COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Precision: {results['tp']/(results['tp']+results['fp']):.4f}")
    print(f"  Recall: {results['tp']/(results['tp']+results['fn']):.4f}")
    print("\nUse these results to compare against ML models.")
    print("="*60)


if __name__ == "__main__":
    main()
