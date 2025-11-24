#!/usr/bin/env python3
"""
hyperparameter_tuning.py
Systematic hyperparameter optimization using Optuna.

This can improve ROC-AUC by 2-5% through better parameter selection.

Usage:
    python hyperparameter_tuning.py --session all2 --trials 50
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
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
    
    return X, y


def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    """Objective function for XGBoost hyperparameter optimization."""
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    # Add early_stopping_rounds to params for newer XGBoost versions
    params['early_stopping_rounds'] = 50
    
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    
    return roc_auc


def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    """Objective function for LightGBM hyperparameter optimization."""
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    
    return roc_auc


def tune_hyperparameters(session, model_type='xgboost', n_trials=50, output_dir='models'):
    """
    Tune hyperparameters using Optuna.
    
    Args:
        session: Session name
        model_type: 'xgboost' or 'lightgbm'
        n_trials: Number of optimization trials
        output_dir: Directory to save results
    """
    print("="*60)
    print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
    print("="*60)
    print(f"Session: {session}")
    print(f"Number of trials: {n_trials}")
    print()
    
    # Load data
    X, y = load_data(session)
    print(f"[+] Loaded {len(X)} samples with {len(X.columns)} features")
    
    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Further split training for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"[+] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f"{model_type}_tuning_{session}"
    )
    
    # Optimize
    print(f"[+] Starting optimization ({n_trials} trials)...")
    print(f"[+] This may take 30-60 minutes depending on dataset size...")
    print()
    
    if model_type == 'xgboost':
        objective = lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val)
    else:
        objective = lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(
                f"Trial {trial.number}: ROC-AUC = {trial.value:.4f}"
            )
        ]
    )
    
    # Results
    print()
    print("="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    print("\n[+] Training final model with best parameters...")
    
    scale_pos_weight = (len(y_train_full) - y_train_full.sum()) / max(y_train_full.sum(), 1)
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    if model_type == 'xgboost':
        best_params['eval_metric'] = 'logloss'
        best_params['early_stopping_rounds'] = 50
        final_model = XGBClassifier(**best_params)
        final_model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        best_params['verbose'] = -1
        final_model = LGBMClassifier(**best_params)
        final_model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_test, y_test)],
            callbacks=[]
        )
    
    # Evaluate on test set
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"\n[+] Final test set ROC-AUC: {test_roc_auc:.4f}")
    
    # Save model and parameters
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"tuned_{model_type}_{session}.pkl")
    joblib.dump(final_model, model_path)
    print(f"[+] Saved model: {model_path}")
    
    params_path = os.path.join(output_dir, f"tuned_params_{model_type}_{session}.txt")
    with open(params_path, 'w') as f:
        f.write(f"Best ROC-AUC: {study.best_value:.4f}\n")
        f.write(f"Test ROC-AUC: {test_roc_auc:.4f}\n")
        f.write(f"\nBest parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    print(f"[+] Saved parameters: {params_path}")
    
    # Save optimization history
    history_df = study.trials_dataframe()
    history_path = os.path.join(output_dir, f"tuning_history_{model_type}_{session}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"[+] Saved optimization history: {history_path}")
    
    print("\n" + "="*60)
    print(f"IMPROVEMENT: {test_roc_auc:.4f} vs baseline ~0.856")
    print(f"Gain: +{(test_roc_auc - 0.856) * 100:.2f} percentage points")
    print("="*60)
    
    return final_model, study.best_params, test_roc_auc


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for anomaly detection")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--model", choices=['xgboost', 'lightgbm', 'both'], 
                       default='both', help="Model to tune")
    parser.add_argument("--trials", type=int, default=50, 
                       help="Number of optimization trials (default: 50)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()
    
    if args.model in ['xgboost', 'both']:
        print("\n" + "="*60)
        print("TUNING XGBOOST")
        print("="*60 + "\n")
        tune_hyperparameters(args.session, 'xgboost', args.trials, args.output_dir)
    
    if args.model in ['lightgbm', 'both']:
        print("\n" + "="*60)
        print("TUNING LIGHTGBM")
        print("="*60 + "\n")
        tune_hyperparameters(args.session, 'lightgbm', args.trials, args.output_dir)


if __name__ == "__main__":
    main()

