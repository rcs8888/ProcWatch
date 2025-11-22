#!/usr/bin/env python3
"""
train_eval.py (single-file, robust)
-----------------------------------
Usage examples:
  python3 train_eval.py --session test1
  python3 train_eval.py --data logs/test1/labeled_dataset.csv
  python3 train_eval.py --all                # concatenate logs/*/labeled_dataset.csv
  python3 train_eval.py --all --time-split

Notes:
 - Expects CSV rows with at least: timestamp, pid, ppid, cpu, mem, label
 - label: 0 for normal, 1 for anomaly
 - No hard dependency on imblearn; will use SMOTE/ADASYN if available.
"""
import argparse
import glob
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# Try optional imports (imblearn) for oversampling
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def find_all_sessions_logs():
    paths = glob.glob("logs/*/labeled_dataset.csv")
    return sorted(paths)

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # ensure label column exists
    if "label" not in df.columns:
        # some older names may use 'is_anomaly'
        if "is_anomaly" in df.columns:
            df["label"] = df["is_anomaly"].astype(int)
        else:
            raise ValueError("Dataset missing required 'label' column.")
    # cast label, parse timestamp
    df["label"] = df["label"].astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df

def build_features(df):
    # create deltas if missing
    if "delta_cpu" not in df.columns:
        if "timestamp" in df.columns:
            df = df.sort_values(["pid", "timestamp"])
        else:
            df = df.sort_values(["pid"])
        df["delta_cpu"] = df.groupby("pid")["cpu"].diff().fillna(0)
    if "delta_mem" not in df.columns:
        if "timestamp" in df.columns:
            df = df.sort_values(["pid", "timestamp"])
        else:
            df = df.sort_values(["pid"])
        df["delta_mem"] = df.groupby("pid")["mem"].diff().fillna(0)

    # count children per parent (approx)
    df["num_children"] = df.groupby("ppid")["pid"].transform("count")

    # normalize relative to parent mean to capture anomalous deviation
    df["cpu_norm_parent"] = df["cpu"] / (df.groupby("ppid")["cpu"].transform("mean").replace(0, np.nan)).fillna(0)
    df["mem_norm_parent"] = df["mem"] / (df.groupby("ppid")["mem"].transform("mean").replace(0, np.nan)).fillna(0)

    feature_cols = [
        "cpu", "mem", "delta_cpu", "delta_mem",
        "num_children", "cpu_norm_parent", "mem_norm_parent"
    ]

    # ensure numeric and safe
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, feature_cols

def tune_heuristic_threshold(X_train, y_train):
    """
    Tune a simple heuristic on TRAIN. Heuristic:
      anomaly if cpu > T_cpu OR delta_cpu > T_delta OR num_children >= N_children
    Searches a coarse grid and returns best (tc, td, nc, f1).
    """
    best = None
    best_score = -1.0
    cpu_grid = [5, 10, 20, 30, 40, 50]
    delta_grid = [0.5, 1, 2, 5, 10]
    children_grid = [1, 2, 3, 4]

    for tc in cpu_grid:
        for td in delta_grid:
            for nc in children_grid:
                y_pred = ((X_train["cpu"] > tc) | (X_train["delta_cpu"] > td) | (X_train["num_children"] >= nc)).astype(int)
                s = f1_score(y_train, y_pred, zero_division=0)
                if s > best_score:
                    best_score = s
                    best = (tc, td, nc, s)
    return best

def evaluate_heuristic(X_test, y_test, thresh):
    tc, td, nc, _ = thresh
    y_pred = ((X_test["cpu"] > tc) | (X_test["delta_cpu"] > td) | (X_test["num_children"] >= nc)).astype(int)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return report, cm, f1, (tc, td, nc)

def train_rf(X_train, y_train, X_test, y_test, feature_names, session_name, use_smote=False):
    # pipeline with scaler + RF
    class_weight = "balanced"
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight=class_weight, n_jobs=-1
        ))
    ])

    # Optionally perform oversampling on TRAIN only (if imblearn exists)
    if use_smote and IMBLEARN_AVAILABLE:
        # do small oversampling using SMOTE (or ADASYN) inside a local scope
        try:
            sm = SMOTE(random_state=42, n_jobs=-1)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"[i] SMOTE applied: {len(y_train)} -> {len(y_train_res)} samples")
        except Exception:
            # fallback: no resampling
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    pipeline.fit(X_train_res, y_train_res)
    y_pred = pipeline.predict(X_test)
    # handle case classifier might not have predict_proba
    try:
        y_score = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = np.zeros(len(y_pred))

    return pipeline, y_pred, y_score

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Train & evaluate RF vs heuristic on labeled dataset")
    p.add_argument("--session", default="session", help="session name (reads logs/<session>/labeled_dataset.csv)")
    p.add_argument("--data", default=None, help="explicit path to labeled CSV (overrides --session)")
    p.add_argument("--time-split", action="store_true", help="use time-based split (first 70% train, last 30% test)")
    p.add_argument("--all", action="store_true", help="concatenate all logs/*/labeled_dataset.csv into one dataset")
    p.add_argument("--use-smote", action="store_true", help="apply SMOTE on train if imblearn installed")
    p.add_argument("--save-preds", action="store_true", help="save predictions CSV")
    args = p.parse_args()

    # build path(s)
    if args.data:
        paths = [args.data]
        session_name = os.path.splitext(os.path.basename(args.data))[0]
    elif args.all:
        paths = find_all_sessions_logs()
        session_name = "all"
        if not paths:
            print("[!] No logs/*/labeled_dataset.csv files found.")
            sys.exit(1)
    else:
        paths = [f"logs/{args.session}/labeled_dataset.csv"]
        session_name = args.session

    # load and optionally concat
    dfs = []
    for path in paths:
        try:
            df = load_dataset(path)
            df["_source_file"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            print(f"[!] Skipping {path}: {e}")
    if not dfs:
        print("[!] No valid datasets loaded, exiting.")
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)

    print(f"[i] Loaded {len(df)} rows from {len(dfs)} file(s). Using session name: '{session_name}'")

    df, feature_cols = build_features(df)
    # drop rows with missing label
    df = df.dropna(subset=["label"])
    X = df[feature_cols].copy()
    y = df["label"].astype(int).copy()

    # split
    if args.time_split and "timestamp" in df.columns and df["timestamp"].notna().all():
        df = df.sort_values("timestamp")
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]
        print("[i] Using time-based split (first 70% train, last 30% test).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        print("[i] Using stratified random split (stratify by label).")

    # print class distribution
    print(f"[i] Class distribution (train): {np.bincount(y_train.astype(int))}")
    try:
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        print(f"[i] computed class weights: {dict(enumerate(cw))}")
    except Exception:
        pass

    # tune heuristic
    print("[i] Tuning heuristic threshold on training data (coarse grid)...")
    best_thresh = tune_heuristic_threshold(X_train, y_train)
    print(f"[i] Best heuristic (on train) -> cpu>{best_thresh[0]} OR delta_cpu>{best_thresh[1]} OR num_children>={best_thresh[2]}  F1={best_thresh[3]:.4f}")

    # train RF
    pipeline, y_pred_rf, y_score_rf = train_rf(
        X_train, y_train, X_test, y_test, feature_cols, session_name, use_smote=(args.use_smote and IMBLEARN_AVAILABLE)
    )

    # metrics RF
    print("\n======= RANDOM FOREST PERFORMANCE =======")
    rf_report = classification_report(y_test, y_pred_rf, digits=4, zero_division=0)
    print(rf_report)
    cm = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:\n", cm)
    try:
        roc_auc = roc_auc_score(y_test, y_score_rf) if len(np.unique(y_test)) > 1 else float("nan")
        pr_auc = average_precision_score(y_test, y_score_rf) if len(np.unique(y_test)) > 1 else float("nan")
    except Exception:
        roc_auc = pr_auc = float("nan")
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    print(f"ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}   F1: {f1_rf:.4f}")

    # cross-val (on TRAIN)
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        print("Cross-validated F1 (macro):", np.round(cv_scores, 4), " mean =", np.round(cv_scores.mean(), 4))
    except Exception:
        pass

    # feature importances
    try:
        rf = pipeline.named_steps["rf"]
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop Feature Importances:")
        print(importances.to_string())
    except Exception:
        pass

    # heuristic evaluation
    h_report, h_cm, h_f1, h_params = evaluate_heuristic(X_test, y_test, best_thresh)
    print("\n======= HEURISTIC BASELINE (tuned on train) =======")
    print(h_report)
    print("Confusion Matrix:\n", h_cm)
    print(f"Heuristic params: cpu>{h_params[0]}  delta_cpu>{h_params[1]}  num_children>={h_params[2]}  F1={h_f1:.4f}")

    # save predictions if asked
    if args.save_preds or True:
        preds = X_test.copy().reset_index(drop=True)
        preds["y_true"] = y_test.values
        preds["y_pred_rf"] = y_pred_rf
        preds["y_score_rf"] = y_score_rf
        preds["y_pred_heuristic"] = ((preds["cpu"] > h_params[0]) | (preds["delta_cpu"] > h_params[1]) | (preds["num_children"] >= h_params[2])).astype(int)
        outname = f"predictions_{session_name}.csv"
        preds.to_csv(outname, index=False)
        print(f"[i] Saved test predictions -> {outname}")

    print("\n[i] Done.")

if __name__ == "__main__":
    main()
