#!/usr/bin/env python3
"""
train_eval.py
------------------------
Trains and evaluates a Random Forest classifier (and a tunable heuristic baseline)
for detecting anomalous process samples from labeled_dataset.csv.

Usage:
    python3 train_eval.py --session test1
    python3 train_eval.py --data logs/test1/labeled_dataset.csv --time-split

Notes:
 - Expects a CSV with at least: timestamp, pid, ppid, cpu, mem, delta_cpu, delta_mem, label
 - label column is expected to be 0 (normal) / 1 (anomaly)
 - Use --time-split to split by time (first 70% train, last 30% test) to avoid leakage
"""

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, f1_score
)
from sklearn.utils import compute_class_weight

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # Standard columns that should exist (protective checks)
    expected = {"timestamp", "pid", "ppid", "cpu", "mem", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    # Ensure label is integer 0/1
    df['label'] = df['label'].astype(int)
    # Convert timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    return df

def build_features(df):
    # Basic numeric features (extend here if you want)
    # If delta columns don't exist, create simple diffs per pid
    if 'delta_cpu' not in df.columns:
        df = df.sort_values(['pid', 'timestamp'])
        df['delta_cpu'] = df.groupby('pid')['cpu'].diff().fillna(0)
    if 'delta_mem' not in df.columns:
        df = df.sort_values(['pid', 'timestamp'])
        df['delta_mem'] = df.groupby('pid')['mem'].diff().fillna(0)

    # children count per parent (approx)
    df['num_children'] = df.groupby('ppid')['pid'].transform('count')

    # Normalize features relative to parent (optional)
    df['cpu_norm_parent'] = df['cpu'] / (df.groupby('ppid')['cpu'].transform('mean').replace(0, np.nan)).fillna(0)
    df['mem_norm_parent'] = df['mem'] / (df.groupby('ppid')['mem'].transform('mean').replace(0, np.nan)).fillna(0)

    feature_cols = [
        'cpu', 'mem', 'delta_cpu', 'delta_mem',
        'num_children', 'cpu_norm_parent', 'mem_norm_parent'
    ]

    # Fill NaNs and infinities
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, feature_cols

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, session, save_preds=True):
    # Compute class weight to help RF on imbalanced data
    classes = np.unique(y_train)
    class_weight = 'balanced'  # uses class frequencies

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight, n_jobs=-1))
    ])

    # Train
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')
    pr_auc = average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')
    f1 = f1_score(y_test, y_pred)

    print("\n======= RANDOM FOREST PERFORMANCE =======")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}   F1: {f1:.4f}")

    # Cross-validated F1 (stratified)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
    print("Cross-validated F1 (macro):", np.round(cv_scores, 4), " mean =", np.round(cv_scores.mean(), 4))

    # Feature importances (from RF inside pipeline)
    rf = pipeline.named_steps['rf']
    try:
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("\nTop Feature Importances:")
        print(importances.to_string())
    except Exception:
        pass

    # Save predictions for manual inspection
    if save_preds:
        out = X_test.copy()
        out['y_true'] = y_test
        out['y_pred_rf'] = y_pred
        out['y_score_rf'] = y_proba
        out.to_csv(f"predictions_{session}.csv", index=False)
        print(f"[i] Saved test predictions -> predictions_{session}.csv")

    return pipeline, (report, cm, roc_auc, pr_auc, f1)

def tune_heuristic_threshold(X_train, y_train):
    """
    Tune a simple heuristic threshold on cpu_pct or delta_cpu to maximize F1 on TRAIN.
    Heuristic will be: anomaly if cpu > T_cpu OR delta_cpu > T_delta OR num_children >= N_children.
    We search small grid for T_cpu, T_delta, N_children.
    """
    best = None
    best_score = -1
    # coarser grid; adjust as desired
    cpu_grid = [10, 20, 30, 40, 50, 60, 70, 80]
    delta_grid = [1, 5, 10, 20, 40]
    children_grid = [1, 2, 3, 4, 5, 10]

    for tc in cpu_grid:
        for td in delta_grid:
            for nc in children_grid:
                y_pred = ((X_train['cpu'] > tc) | (X_train['delta_cpu'] > td) | (X_train['num_children'] >= nc)).astype(int)
                s = f1_score(y_train, y_pred)
                if s > best_score:
                    best_score = s
                    best = (tc, td, nc, s)
    return best

def evaluate_heuristic(X_test, y_test, threshold_tuple):
    tc, td, nc, _ = threshold_tuple
    y_pred = ((X_test['cpu'] > tc) | (X_test['delta_cpu'] > td) | (X_test['num_children'] >= nc)).astype(int)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return (report, cm, f1, (tc, td, nc))

def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RF vs heuristic on labeled dataset")
    parser.add_argument("--session", default="session", help="session name (used to find logs/<session>/labeled_dataset.csv)")
    parser.add_argument("--data", default=None, help="explicit path to labeled CSV (overrides --session)")
    parser.add_argument("--time-split", action="store_true", help="split by time (first 70% train, last 30% test)")
    args = parser.parse_args()

    if args.data:
        path = args.data
        session_name = os.path.splitext(os.path.basename(path))[0]
    else:
        path = f"logs/{args.session}/labeled_dataset.csv"
        session_name = args.session

    print(f"[i] Loading dataset: {path}")
    df = load_dataset(path)
    df, feature_cols = build_features(df)

    # Drop rows with missing label or NaT timestamp
    df = df.dropna(subset=['label'])

    # Features and labels
    X = df[feature_cols].copy()
    y = df['label'].astype(int).copy()

    # Train/test split: time-aware or stratified random
    if args.time_split and 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train, y_train = train_df[feature_cols], train_df['label']
        X_test, y_test = test_df[feature_cols], test_df['label']
        print("[i] Using time-based split: first 70% train, last 30% test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        print("[i] Using stratified random split (stratify by label)")

    # If extreme imbalance, compute/print class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"[i] Class distribution (train): {np.bincount(y_train.astype(int))}  computed class weights: {dict(enumerate(cw))}")

    # Tune heuristic on TRAIN
    print("[i] Tuning heuristic threshold on training data (coarse grid)...")
    best_thresh = tune_heuristic_threshold(X_train, y_train)
    print(f"[i] Best heuristic (on train) -> cpu>{best_thresh[0]} OR delta_cpu>{best_thresh[1]} OR num_children>={best_thresh[2]}  F1={best_thresh[3]:.4f}")

    # Train RF and evaluate
    pipeline, rf_metrics = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols, session_name, save_preds=True)

    # Evaluate heuristic on test using tuned thresholds
    h_report, h_cm, h_f1, h_params = evaluate_heuristic(X_test, y_test, best_thresh)
    print("\n======= HEURISTIC BASELINE (tuned on train) =======")
    print(h_report)
    print("Confusion Matrix:\n", h_cm)
    print(f"Heuristic params: cpu>{h_params[0]}  delta_cpu>{h_params[1]}  num_children>={h_params[2]}  F1={h_f1:.4f}")

    print("\n[i] Done.")

if __name__ == '__main__':
    main()
