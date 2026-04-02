"""
Demo utilities for CKPT3 presentation.

Essential functions for the 10-minute in-class demo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def print_demo_header(title):
    """Print formatted demo section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def quick_intro():
    """Quick introduction (1 minute)."""
    print_demo_header("CKPT3: STACKED ENSEMBLE FOR CLV PREDICTION")
    
    print("IMPLEMENTATION OVERVIEW:")
    print("├── Config A: Meta-learner uses only base predictions")
    print("├── Config B: Meta-learner uses predictions + customer features")
    print("├── Base Models: ElasticNet, RandomForest, XGBoost")
    print("└── ReAct Agent: LangGraph-based intelligent model selection\n")
    
    print("KEY INNOVATION:")
    print("  Config B enables context-aware decisions by providing customer")
    print("  features to the meta-learner for segment-specific weighting.\n")


def show_model_architecture(use_features=True):
    """Display model architecture diagram."""
    config = "Config B" if use_features else "Config A"
    
    print(f"\n{config} ARCHITECTURE:")
    print("┌─────────────────────────────────────────────────┐")
    print("│           BASE MODELS (Level 1)                 │")
    print("│  ┌───────────┐ ┌───────────┐ ┌───────────┐    │")
    print("│  │ElasticNet │ │RandomForest│ │  XGBoost  │    │")
    print("│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    │")
    print("│        └─────────────┴─────────────┘            │")
    print("│                     ▼                           │")
    
    if use_features:
        print("│        ┌─────────────────────────┐            │")
        print("│        │  Predictions + Features │            │")
        print("│        └──────────┬──────────────┘            │")
    else:
        print("│        ┌─────────────────────────┐            │")
        print("│        │   Predictions Only      │            │")
        print("│        └──────────┬──────────────┘            │")
    
    print("│                   ▼                             │")
    print("│        ┌──────────────────────┐                │")
    print("│        │ META-LEARNER (Level 2)│                │")
    print("│        │   ElasticNet         │                │")
    print("│        └──────────┬───────────┘                │")
    print("│                   ▼                             │")
    print("│        ┌──────────────────────┐                │")
    print("│        │  Final Prediction    │                │")
    print("│        └──────────────────────┘                │")
    print("└─────────────────────────────────────────────────┘\n")


def show_training_snapshot(checkpoint_dir):
    """Display training snapshot."""
    print_demo_header("TRAINING SNAPSHOT")
    
    log_path = f"{checkpoint_dir}/training_log.json"
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        if logs:
            latest = logs[-1]
            
            print("TRAINING DETAILS:")
            print(f"  Configuration: {latest['config']}")
            print(f"  Meta-features shape: {latest['meta_features_shape']}")
            print(f"  Training MAE: {latest['train_mae']:.4f}")
            print(f"  Training RMSE: {latest['train_rmse']:.4f}")
            print(f"  Training time: {latest['train_time_seconds']:.2f}s")
            print(f"  Timestamp: {latest['timestamp']}\n")
            print(f"✓ Checkpoint saved: {checkpoint_dir}/\n")
        else:
            print("⚠ No training logs found\n")
    
    except FileNotFoundError:
        print(f"⚠ Training log not found: {log_path}\n")


def show_inference_demo(model, X_test_sample, y_test_sample, base_models):
    """Run inference demo on sample batch."""
    print_demo_header("INFERENCE DEMO")
    
    print(f"Processing {len(X_test_sample)} customers...\n")
    
    # Generate predictions
    ensemble_preds, base_preds_dict = model.predict(X_test_sample, base_models)
    
    # Show sample predictions
    print("SAMPLE PREDICTIONS:")
    print("─" * 70)
    print(f"{'Customer':<12} {'True':<8} {'Ensemble':<10} {'ElasticNet':<12} {'XGBoost':<10}")
    print("─" * 70)
    
    for i in range(min(5, len(X_test_sample))):
        cust_id = X_test_sample.index[i]
        true_val = y_test_sample.iloc[i]
        ens_pred = ensemble_preds[i]
        en_pred = base_preds_dict['ElasticNet'][i]
        xgb_pred = base_preds_dict['XGBoost'][i]
        
        print(f"{cust_id:<12} {true_val:<8.2f} {ens_pred:<10.2f} {en_pred:<12.2f} {xgb_pred:<10.2f}")
    
    print("─" * 70)
    
    # Compute metrics
    mae = np.abs(ensemble_preds - y_test_sample).mean()
    rmse = np.sqrt(((ensemble_preds - y_test_sample) ** 2).mean())
    
    print(f"\nBATCH METRICS:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")


def show_comparison_results(results_df):
    """Show comparison to baselines."""
    print_demo_header("COMPARISON TO BASELINES")
    
    print("PERFORMANCE TABLE:")
    print(results_df.to_string(index=False))
    print()
    
    # Calculate improvement
    best_baseline_mae = results_df[results_df['Type'] == 'Baseline']['MAE'].min()
    best_ensemble_mae = results_df[results_df['Type'] == 'Ensemble']['MAE'].min()
    
    if best_ensemble_mae < best_baseline_mae:
        improvement = ((best_baseline_mae - best_ensemble_mae) / best_baseline_mae) * 100
        print(f"✓ ENSEMBLE IMPROVES OVER BEST BASELINE BY {improvement:.2f}%\n")
    else:
        print(f"✗ Ensemble MAE: {best_ensemble_mae:.4f} vs Best Baseline: {best_baseline_mae:.4f}\n")


def create_comparison_plot(results_df, save_path=None):
    """Create comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    colors = ['lightcoral' if t == 'Baseline' else 'lightgreen' 
              for t in results_df['Type']]
    
    # MAE plot
    axes[0].barh(results_df['Model'], results_df['MAE'], color=colors)
    axes[0].set_xlabel('MAE', fontsize=12)
    axes[0].set_title('Model Comparison: MAE', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].axvline(results_df['MAE'].min(), color='green', linestyle='--', linewidth=2)
    
    # RMSE plot
    axes[1].barh(results_df['Model'], results_df['RMSE'], color=colors)
    axes[1].set_xlabel('RMSE', fontsize=12)
    axes[1].set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].axvline(results_df['RMSE'].min(), color='green', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}\n")
    
    return fig


if __name__ == "__main__":
    print("Demo Utilities Module")
    print("Use: from src.demo_utils import quick_intro, show_training_snapshot, etc.")