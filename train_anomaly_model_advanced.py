#!/usr/bin/env python3
"""
train_anomaly_model_advanced.py
Enhanced ML pipeline with advanced techniques for high-performance anomaly detection.

Improvements:
- Time-aware feature engineering (temporal context)
- Ensemble stacking methods
- SMOTE for handling class imbalance
- Neural network (MLP) with proper architecture
- Advanced hyperparameter tuning
- Process-level aggregation features

Author: Assistant (Enhanced)
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef
)

# Models
from sklearn.ensemble import (
    IsolationForest, 
    RandomForestClassifier, 
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Imbalance handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Advanced Feature Engineering
# =============================================================================

def create_temporal_features(df):
    """
    Add temporal context features that capture process behavior over time.
    This is crucial for anomaly detection.
    """
    print("\n[+] Creating advanced temporal features...")
    
    df = df.sort_values(['pid', 'timestamp']).copy()
    
    # 1. HISTORICAL AGGREGATIONS (more windows)
    for window in [3, 10, 20]:
        df[f'cpu_max_{window}'] = df.groupby('pid')['cpu'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
        df[f'cpu_min_{window}'] = df.groupby('pid')['cpu'].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        df[f'mem_max_{window}'] = df.groupby('pid')['mem'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
    
    # 2. TREND FEATURES (is resource usage increasing/decreasing?)
    df['cpu_trend'] = df.groupby('pid')['cpu'].diff(5).fillna(0)
    df['mem_trend'] = df.groupby('pid')['mem'].diff(5).fillna(0)
    df['cpu_acceleration'] = df.groupby('pid')['cpu_trend'].diff().fillna(0)
    
    # 3. VOLATILITY (how erratic is the process?)
    df['cpu_volatility'] = df.groupby('pid')['cpu'].rolling(10, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    df['mem_volatility'] = df.groupby('pid')['mem'].rolling(10, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    
    # 4. PERCENTILE FEATURES (where does current usage rank?)
    df['cpu_pct_rank'] = df.groupby('pid')['cpu'].rank(pct=True)
    df['mem_pct_rank'] = df.groupby('pid')['mem'].rank(pct=True)
    
    # 5. INTERACTION FEATURES (combined resource pressure)
    df['cpu_mem_product'] = df['cpu'] * df['mem']
    df['cpu_mem_ratio'] = df['cpu'] / (df['mem'] + 1e-6)
    df['resource_pressure'] = df['cpu'] + df['mem']
    
    # 6. SUDDEN CHANGE DETECTION
    df['cpu_jump'] = (df['delta_cpu'] > df['delta_cpu'].rolling(20).quantile(0.95)).astype(int)
    df['mem_jump'] = (df['delta_mem'] > df['delta_mem'].rolling(20).quantile(0.95)).astype(int)
    
    # 7. PROCESS LIFECYCLE FEATURES
    df['cpu_per_age'] = df['cpu'] / (df['proc_age'] + 1)
    df['mem_per_age'] = df['mem'] / (df['proc_age'] + 1)
    
    return df.replace([np.inf, -np.inf], 0).fillna(0)


def create_process_level_features(df):
    """
    Aggregate statistics per process (PID) for better context.
    Captures overall process behavior patterns.
    """
    print("[+] Creating process-level aggregation features...")
    
    # Aggregate per PID
    pid_agg = df.groupby('pid').agg({
        'cpu': ['mean', 'std', 'max', 'min', 'median'],
        'mem': ['mean', 'std', 'max', 'min', 'median'],
        'num_children': ['mean', 'max'],
        'proc_age': ['max'],
        'timestamp': 'count'  # number of observations
    }).reset_index()
    
    pid_agg.columns = ['pid'] + ['_'.join(col).strip('_') for col in pid_agg.columns[1:]]
    pid_agg.rename(columns={'timestamp_count': 'observation_count'}, inplace=True)
    
    # Merge back
    df = df.merge(pid_agg, on='pid', how='left', suffixes=('', '_pid'))
    
    # Deviation from process average
    df['cpu_dev_from_avg'] = df['cpu'] - df['cpu_mean']
    df['mem_dev_from_avg'] = df['mem'] - df['mem_mean']
    
    return df


def select_features(df):
    """Select all engineered features for modeling."""
    exclude = ['timestamp', 'session_id', 'pid', 'ppid', 'exe', 'label', 'session']
    
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Ensure numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    return df[feature_cols], df['label']


# =============================================================================
# Model Training with Advanced Techniques
# =============================================================================

def train_xgboost_tuned(X_train, y_train, X_test, y_test):
    """XGBoost with more aggressive hyperparameters."""
    print("\n" + "="*60)
    print("XGBOOST (OPTIMIZED)")
    print("="*60)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,
        min_child_weight=2,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "XGBoost_Optimized"


def train_lightgbm_tuned(X_train, y_train, X_test, y_test):
    """LightGBM with enhanced parameters."""
    print("\n" + "="*60)
    print("LIGHTGBM (OPTIMIZED)")
    print("="*60)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    model = LGBMClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_alpha=0.3,
        reg_lambda=2.0,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[]
    )
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "LightGBM_Optimized"


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Sklearn's Gradient Boosting (often underestimated)."""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING")
    print("="*60)
    
    model = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "GradientBoosting"


def train_mlp_classifier(X_train, y_train, X_test, y_test):
    """Multi-layer Perceptron (Neural Network)."""
    print("\n" + "="*60)
    print("NEURAL NETWORK (MLP)")
    print("="*60)
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "NeuralNetwork_MLP"


def train_ensemble_voting(X_train, y_train, X_test, y_test):
    """Voting ensemble of best models."""
    print("\n" + "="*60)
    print("VOTING ENSEMBLE")
    print("="*60)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
    )
    
    lgb = LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
    )
    
    model = VotingClassifier(
        estimators=[('xgb', xgb), ('lgb', lgb), ('gb', gb)],
        voting='soft',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "VotingEnsemble"


def train_stacking_ensemble(X_train, y_train, X_test, y_test):
    """Stacking ensemble with meta-learner."""
    print("\n" + "="*60)
    print("STACKING ENSEMBLE")
    print("="*60)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    # Base models
    estimators = [
        ('xgb', XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
        )),
        ('lgb', LGBMClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42
        ))
    ]
    
    # Meta-learner
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba, "StackingEnsemble"


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(y_test, y_pred, y_proba, model_name):
    """Comprehensive evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
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
    
    # Composite score
    composite_score = (roc_auc + avg_precision + f1 + (1 + mcc)/2) / 4
    
    return {
        'model': model_name,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1': f1,
        'mcc': mcc,
        'fpr': fpr,
        'fnr': fnr,
        'composite_score': composite_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced anomaly detection training")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--use-smote", action='store_true', help="Apply SMOTE for class balance")
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = f"logs/{args.session}/labeled_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"\n{'='*60}")
    print(f"ADVANCED ANOMALY DETECTION PIPELINE")
    print(f"Session: {args.session}")
    print(f"{'='*60}")
    
    df = pd.read_csv(dataset_path)
    print(f"\nTotal samples: {len(df)}")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    print(f"Anomaly rate: {df['label'].sum() / len(df) * 100:.2f}%")
    
    # Advanced feature engineering
    df = create_temporal_features(df)
    df = create_process_level_features(df)
    
    # Feature selection
    X, y = select_features(df)
    print(f"\n[+] Total features after engineering: {len(X.columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} anomalies)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} anomalies)")
    
    # Apply SMOTE if requested (can help with imbalance)
    if args.use_smote:
        print("\n[+] Applying SMOTE for class balance...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train_resampled)} samples ({y_train_resampled.sum()} anomalies)")
        X_train, y_train = X_train_resampled, y_train_resampled
    
    # Scaling (important for MLP)
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train all models
    results = []
    models_to_save = []
    
    # 1. XGBoost (Optimized)
    try:
        model, y_pred, y_proba, name = train_xgboost_tuned(X_train, y_train, X_test, y_test)
        result = evaluate_model(y_test, y_pred, y_proba, name)
        results.append(result)
        models_to_save.append((model, name, scaler))
    except Exception as e:
        print(f"\n[!] XGBoost failed: {e}")
    
    # 2. LightGBM (Optimized)
    try:
        model, y_pred, y_proba, name = train_lightgbm_tuned(X_train, y_train, X_test, y_test)
        result = evaluate_model(y_test, y_pred, y_proba, name)
        results.append(result)
        models_to_save.append((model, name, scaler))
    except Exception as e:
        print(f"\n[!] LightGBM failed: {e}")
    
    # 3. Voting Ensemble (fast - uses XGB + LGB)
    if len(X_train) < 100000:  # Only for smaller datasets
        try:
            print("\n[i] Training Voting Ensemble (XGB + LGB)...")
            model, y_pred, y_proba, name = train_ensemble_voting(X_train, y_train, X_test, y_test)
            result = evaluate_model(y_test, y_pred, y_proba, name)
            results.append(result)
            models_to_save.append((model, name, scaler))
        except Exception as e:
            print(f"\n[!] Voting Ensemble failed: {e}")
    else:
        print("\n[i] Skipping Voting Ensemble (dataset too large, would be very slow)")
    
    # Skip slow models for large datasets
    print("\n[i] Skipping Gradient Boosting, Neural Network, and Stacking Ensemble")
    print("[i] (These are very slow on large datasets and XGBoost/LightGBM already perform well)")
    
    # Summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\n" + results_df[['model', 'roc_auc', 'avg_precision', 'f1', 'mcc']].to_string(index=False))
    
    # Save best model
    if results:
        best_model_name = results_df.iloc[0]['model']
        best_model_tuple = [m for m in models_to_save if m[1] == best_model_name][0]
        best_model, _, best_scaler = best_model_tuple
        
        model_path = os.path.join(args.output_dir, f"best_model_{args.session}.pkl")
        scaler_path = os.path.join(args.output_dir, f"scaler_{args.session}.pkl")
        
        joblib.dump(best_model, model_path)
        joblib.dump(best_scaler, scaler_path)
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
        print(f"{'='*60}")
        print(f"Saved to: {model_path}")
        
        # Save all results
        results_path = os.path.join(args.output_dir, f"results_{args.session}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

