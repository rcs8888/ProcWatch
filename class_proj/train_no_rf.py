#!/usr/bin/env python3
"""
train_eval.py — XGBoost version (drop-in replacement)
-----------------------------------------------------
Replaces the Random Forest with XGBoost to improve detection of subtle anomalies.

Usage:
    python3 train_eval.py --session all
    python3 train_eval.py --data logs/all/labeled_dataset.csv
    python3 train_eval.py --time-split
"""

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight

# Try XGBoost, else fall back to GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("[!] XGBoost not installed — using sklearn GradientBoostingClassifier instead")


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    expected = {"timestamp", "pid", "ppid", "cpu", "mem", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label"] = df["label"].astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    return df


# -------------------------------------------------------------------
# Build features
# -------------------------------------------------------------------

def build_features(df):
    if "delta_cpu" not in df.columns:
        df = df.sort_values(["pid", "timestamp"])
        df["delta_cpu"] = df.groupby("pid")["cpu"].diff().fillna(0)

    if "delta_mem" not in df.columns:
        df = df.sort_values(["pid", "timestamp"])
        df["delta_mem"] = df.groupby("pid")["mem"].diff().fillna(0)

    df["num_children"] = df.groupby("ppid")["pid"].transform("count")

    df["cpu_norm_parent"] = df["cpu"] / (
        df.groupby("ppid")["cpu"].transform("mean").replace(0, np.nan)
    ).fillna(0)

    df["mem_norm_parent"] = df["mem"] / (
        df.groupby("ppid")["mem"].transform("mean").replace(0, np.nan)
    ).fillna(0)

    feature_cols = [
        "cpu", "mem",
        "delta_cpu", "delta_mem",
        "num_children",
        "cpu_norm_parent", "mem_norm_parent"
    ]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, feature_cols


# -------------------------------------------------------------------
# Train/test split + training
# -------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, session):

    # Compute class weight manually for XGBoost
    unique, counts = np.unique(y_train, return_counts=True)
    neg, pos = counts
    scale_pos = neg / pos

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Fallback model
        model = GradientBoostingClassifier()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- Metrics
    print("\n======= MODEL PERFORMANCE =======")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}   F1: {f1:.4f}")

    # CV F1 macro
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf,
                                scoring="f1_macro", n_jobs=-1)
    print("Cross-validated F1 (macro):", np.round(cv_scores, 4),
          " mean =", np.round(np.mean(cv_scores), 4))

    # --- Feature importances
    try:
        if HAS_XGB:
            booster = pipeline.named_steps["model"]
            importances = booster.feature_importances_
            s = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            print("\nTop Feature Importances:")
            print(s.to_string())
    except:
        pass

    # --- Save predictions
    out = X_test.copy()
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out["y_score"] = y_proba
    out.to_csv(f"predictions_{session}.csv", index=False)
    print(f"[i] Saved predictions -> predictions_{session}.csv")

    return pipeline


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="session")
    parser.add_argument("--data", default=None)
    parser.add_argument("--time-split", action="store_true")
    args = parser.parse_args()

    # Path logic unchanged
    if args.data:
        path = args.data
        session_name = os.path.splitext(os.path.basename(path))[0]
    else:
        path = f"logs/{args.session}/labeled_dataset.csv"
        session_name = args.session

    print(f"[i] Loading dataset: {path}")
    df = load_dataset(path)
    df, feature_cols = build_features(df)
    df = df.dropna(subset=["label"])

    X = df[feature_cols]
    y = df["label"].astype(int)

    # Time-split or stratified split (same behavior as before)
    if args.time_split and "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")
        split = int(len(df_sorted) * 0.7)
        train_df = df_sorted.iloc[:split]
        test_df = df_sorted.iloc[split:]

        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        print("[i] Using time-based split")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        print("[i] Using stratified random split")

    print("[i] Class distribution (train):", np.bincount(y_train))

    train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols, session_name)


if __name__ == "__main__":
    main()
