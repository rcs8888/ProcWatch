#!/usr/bin/env python3
"""
train_with_tuned_params.py
Train model using optimized hyperparameters from tuning.

This loads the best parameters found by hyperparameter_tuning.py
and trains a final model for evaluation and visualization.

Usage:
    python train_with_tuned_params.py --session all2 --model xgboost
    python train_with_tuned_params.py --session all2 --params-file models/tuned_params_xgboost_all2.txt
    
Author: Assistant
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data(session):
    """Load and prepare dataset."""
    dataset_path = f"logs/{session}/labeled_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"[+] Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    
    exclude = ['timestamp', 'session_id', 'pid', 'ppid', 'exe', 'label', 'session']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y, feature_cols


def parse_params_file(params_file):
    """Parse hyperparameters from tuning output file."""
    print(f"[+] Loading parameters from: {params_file}")
    
    params = {}
    
    with open(params_file, 'r') as f:
        lines = f.readlines()
        
        # Skip to parameters section
        in_params = False
        for line in lines:
            if 'Best parameters:' in line:
                in_params = True
                continue
            
            if in_params and ':' in line:
                # Parse "param_name: value"
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    # Convert to appropriate type
                    try:
                        # Try int first
                        if '.' not in param_value:
                            params[param_name] = int(param_value)
                        else:
                            # Try float
                            params[param_name] = float(param_value)
                    except ValueError:
                        # Keep as string
                        params[param_name] = param_value
    
    print(f"[+] Loaded {len(params)} parameters:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    
    return params


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {tn:6d}  {fp:6d}")
    print(f"       Anomaly  {fn:6d}  {tp:6d}")
    
    print("\nAdvanced Metrics:")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"ROC-AUC Score:        {roc_auc:.4f}")
    print(f"Average Precision:    {avg_precision:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Matthews Corr Coef:   {mcc:.4f}")
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"False Positive Rate:  {fpr:.4f}")
    print(f"False Negative Rate:  {fnr:.4f}")
    
    return {
        'model': model_name,
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
    parser = argparse.ArgumentParser(description="Train model with tuned hyperparameters")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--model", choices=['xgboost', 'lightgbm'], 
                       default='xgboost', help="Model type")
    parser.add_argument("--params-file", default=None,
                       help="Path to tuned parameters file (auto-detected if not provided)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING WITH TUNED HYPERPARAMETERS")
    print("="*60)
    print(f"Session: {args.session}")
    print(f"Model: {args.model}")
    print()
    
    # Auto-detect params file if not provided
    if args.params_file is None:
        args.params_file = os.path.join(
            args.output_dir, 
            f"tuned_params_{args.model}_{args.session}.txt"
        )
    
    if not os.path.exists(args.params_file):
        print(f"[!] ERROR: Parameters file not found: {args.params_file}")
        print(f"[!] Run hyperparameter_tuning.py first!")
        return
    
    # Load parameters
    tuned_params = parse_params_file(args.params_file)
    
    if not tuned_params:
        print("[!] ERROR: No parameters found in file!")
        return
    
    # Load data
    X, y, feature_names = load_data(args.session)
    print(f"\n[+] Loaded {len(X)} samples with {len(feature_names)} features")
    
    # Train/test split (same as tuning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"[+] Train set: {len(X_train)} samples ({y_train.sum()} anomalies)")
    print(f"[+] Test set:  {len(X_test)} samples ({y_test.sum()} anomalies)")
    
    # Add required parameters
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    tuned_params['scale_pos_weight'] = scale_pos_weight
    tuned_params['random_state'] = 42
    tuned_params['n_jobs'] = -1
    
    # Train model
    print(f"\n[+] Training {args.model.upper()} with tuned parameters...")
    
    if args.model == 'xgboost':
        tuned_params['eval_metric'] = 'logloss'
        tuned_params['early_stopping_rounds'] = 50
        model = XGBClassifier(**tuned_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        model_name = "XGBoost_Tuned"
    else:
        tuned_params['verbose'] = -1
        tuned_params['early_stopping_rounds'] = 50
        model = LGBMClassifier(**tuned_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[]
        )
        model_name = "LightGBM_Tuned"
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, model_name)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = os.path.join(args.output_dir, f"tuned_model_{args.session}.pkl")
    joblib.dump(model, model_path)
    print(f"\n[+] Saved model: {model_path}")
    
    # Save scaler (for compatibility with visualization script)
    scaler = RobustScaler()
    scaler.fit(X_train)
    scaler_path = os.path.join(args.output_dir, f"tuned_scaler_{args.session}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved scaler: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(args.output_dir, f"tuned_features_{args.session}.txt")
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"[+] Saved features: {features_path}")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f"tuned_results_{args.session}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[+] Saved results: {results_path}")
    
    # Save predictions for visualization
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    predictions = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model,
        'feature_names': feature_names
    }
    
    predictions_path = os.path.join(args.output_dir, f"tuned_predictions_{args.session}.pkl")
    joblib.dump(predictions, predictions_path)
    print(f"[+] Saved predictions: {predictions_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Matthews Correlation: {results['mcc']:.4f}")
    print("\nModel saved and ready for visualization!")
    print("="*60)


if __name__ == "__main__":
    main()

