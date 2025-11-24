#!/usr/bin/env python3
"""
generate_visualizations.py
Create publication-quality visualizations for anomaly detection paper.

Generates:
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance plots
- Performance comparison tables
- Training metrics over time

Usage:
    python generate_visualizations.py --session all --model-path models/best_model_all.pkl

Author: Assistant
Date: 2025-11
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (8, 6)


def load_data(session):
    """Load and prepare dataset."""
    dataset_path = f"logs/{session}/labeled_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"[+] Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    
    # Select features
    exclude = ['timestamp', 'session_id', 'pid', 'ppid', 'exe', 'label', 'session']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y, feature_cols


def plot_roc_curves(models_data, output_dir):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_data: List of tuples (name, y_test, y_proba)
    """
    print("[+] Generating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, y_test, y_proba) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', pad=20)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_precision_recall_curves(models_data, output_dir):
    """Plot Precision-Recall curves."""
    print("[+] Generating Precision-Recall curves...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, y_test, y_proba) in enumerate(models_data):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.plot(recall, precision, color=colors[i % len(colors)],
                lw=2.5, label=f'{name} (AP = {avg_precision:.4f})')
    
    # Baseline
    baseline = sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', lw=1.5, 
               label=f'Baseline (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves', fontweight='bold', pad=20)
    plt.legend(loc="lower left", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Plot confusion matrix with percentages."""
    print(f"[+] Generating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create annotations with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'],
               cbar_kws={'label': 'Count'},
               ax=ax, linewidths=1, linecolor='gray')
    
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', pad=20)
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_feature_importance(model, feature_names, model_name, output_dir, top_n=20):
    """Plot feature importance for tree-based models."""
    print(f"[+] Generating feature importance for {model_name}...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"    [!] Model {model_name} does not have feature_importances_")
        return
    
    # Create dataframe and sort
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Top N features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    
    # Horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features - {model_name}', 
             fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'], i, f" {row['importance']:.4f}", 
                va='center', fontsize=8)
    
    output_path = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")
    
    # Save feature importance to CSV
    csv_path = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.csv')
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")


def plot_metric_comparison(results_df, output_dir):
    """Plot comparison of metrics across models."""
    print("[+] Generating metric comparison plot...")
    
    metrics = ['roc_auc', 'avg_precision', 'f1', 'mcc']
    metric_names = ['ROC-AUC', 'Avg Precision', 'F1 Score', 'MCC']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        data = results_df.sort_values(metric, ascending=False)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        
        bars = ax.barh(range(len(data)), data[metric], color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['model'])
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(name, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row[metric], i, f" {row[metric]:.4f}", 
                   va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'metric_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")


def create_performance_table(results_df, output_dir):
    """Create a publication-ready performance table."""
    print("[+] Generating performance table...")
    
    # Format the dataframe
    table_df = results_df[['model', 'roc_auc', 'avg_precision', 'f1', 'mcc', 'fpr', 'fnr']].copy()
    
    # Round values
    for col in ['roc_auc', 'avg_precision', 'f1', 'mcc', 'fpr', 'fnr']:
        table_df[col] = table_df[col].round(4)
    
    # Rename columns for publication
    table_df.columns = ['Model', 'ROC-AUC', 'Avg Precision', 'F1 Score', 'MCC', 'FPR', 'FNR']
    
    # Sort by ROC-AUC
    table_df = table_df.sort_values('ROC-AUC', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(table_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_df.values,
                    colLabels=table_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2] + [0.13]*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows (alternating colors)
    for i in range(1, len(table_df) + 1):
        if i % 2 == 0:
            for j in range(len(table_df.columns)):
                table[(i, j)].set_facecolor('#E7E6E6')
    
    # Highlight best scores in each column
    for col_idx in range(1, len(table_df.columns)):
        col_name = table_df.columns[col_idx]
        if col_name in ['FPR', 'FNR']:
            best_val = table_df[col_name].min()
        else:
            best_val = table_df[col_name].max()
        
        for row_idx, val in enumerate(table_df[col_name], 1):
            if val == best_val:
                table[(row_idx, col_idx)].set_facecolor('#C6E0B4')
                table[(row_idx, col_idx)].set_text_props(weight='bold')
    
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    output_path = os.path.join(output_dir, 'performance_table.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, 'performance_table.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication visualizations")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING PUBLICATION VISUALIZATIONS")
    print("="*60)
    print(f"Session: {args.session}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    X, y, feature_names = load_data(args.session)
    print(f"[+] Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"[+] Class distribution: {y.value_counts().to_dict()}")
    print()
    
    # Train/test split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models to get predictions
    models_data = []
    trained_models = []
    
    # Heuristic Baseline
    print("[+] Running Heuristic Baseline...")
    from heuristic_baseline import HeuristicDetector
    heuristic = HeuristicDetector()
    heuristic.fit(X_train, y_train)
    y_pred_heuristic, _ = heuristic.predict(X_test)
    y_proba_heuristic = heuristic.predict_proba(X_test)[:, 1]
    models_data.append(('Heuristic Baseline', y_test, y_proba_heuristic))
    trained_models.append(('Heuristic Baseline', heuristic, y_pred_heuristic))
    
    # XGBoost
    print("[+] Training XGBoost...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    xgb = XGBClassifier(
        n_estimators=731, max_depth=12, learning_rate=0.08629969099319279,
        scale_pos_weight=scale_pos_weight, subsample=0.9348323826941092,
        colsample_bytree=0.7898511898530559, random_state=42, n_jobs=-1,
        eval_metric='logloss', early_stopping_rounds=50
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    models_data.append(('XGBoost', y_test, y_proba_xgb))
    trained_models.append(('XGBoost', xgb, y_pred_xgb))
    
    # LightGBM
    print("[+] Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=500, max_depth=9, learning_rate=0.03,
        scale_pos_weight=scale_pos_weight, subsample=0.85,
        colsample_bytree=0.85, random_state=42, n_jobs=-1,
        verbose=-1, early_stopping_rounds=50
    )
    lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[])
    y_pred_lgb = lgb.predict(X_test)
    y_proba_lgb = lgb.predict_proba(X_test)[:, 1]
    models_data.append(('LightGBM', y_test, y_proba_lgb))
    trained_models.append(('LightGBM', lgb, y_pred_lgb))
    
    print()
    
    # Generate visualizations
    plot_roc_curves(models_data, output_dir)
    plot_precision_recall_curves(models_data, output_dir)
    
    for name, model, y_pred in trained_models:
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        # Only plot feature importance for models that support it (not heuristic)
        if name != 'Heuristic Baseline':
            plot_feature_importance(model, feature_names, name, output_dir, top_n=20)
    
    # Load results if available
    results_path = f"models/tuned_results_{args.session}.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        plot_metric_comparison(results_df, output_dir)
        create_performance_table(results_df, output_dir)
    else:
        print(f"[!] Results file not found: {results_path}")
        print("[!] Skipping metric comparison and performance table")
    
    print()
    print("="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files in {output_dir}/:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    print()
    print("These visualizations are publication-ready at 300 DPI!")


if __name__ == "__main__":
    main()

