# --- cell 1 ---
# ============================================================================
# CELL 1: Imports & Setup
# ============================================================================

import sys
sys.path.append('..')

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib

# Src imports
from src import *

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CKPT3: STACKED ENSEMBLE DEMO".center(70))
print("="*70)
print("\nImports successful!")
print("Ready to begin demo.\n")

# --- cell 2 ---
# ============================================================================
# CELL 2: Load CKPT2 Data & Models
# ============================================================================

print("="*70)
print("LOADING CKPT2 BASELINE OUTPUTS")
print("="*70)

# Load data splits
print("\n[1/3] Loading data splits...")
train_df = pd.read_csv('../data/train_features.csv')
val_df = pd.read_csv('../data/val_features.csv')
test_df = pd.read_csv('../data/test_features.csv')

print(f"  ✓ Train: {len(train_df)} samples")
print(f"  ✓ Val:   {len(val_df)} samples")
print(f"  ✓ Test:  {len(test_df)} samples")

# Prepare X and y
X_train = train_df[['freq', 'freq_3m', 'latetime', 'earlytime']]
y_train = train_df['target']

X_val = val_df[['freq', 'freq_3m', 'latetime', 'earlytime']]
y_val = val_df['target']

X_test = test_df[['freq', 'freq_3m', 'latetime', 'earlytime']]
y_test = test_df['target']

# Load pre-trained baseline models
print("\n[2/3] Loading pre-trained baseline models...")
en_model = joblib.load('../models/elasticnet_baseline.pkl')
rf_model = joblib.load('../models/randomforest_baseline.pkl')
xgb_model = joblib.load('../models/xgboost_baseline.pkl')

print("  ✓ ElasticNet loaded")
print("  ✓ RandomForest loaded")
print("  ✓ XGBoost loaded")

# Load baseline results
print("\n[3/3] Loading baseline results...")
with open('../results/baseline_results.json', 'r') as f:
    baseline_results = json.load(f)

print("  ✓ Baseline results loaded:")
for model, metrics in baseline_results.items():
    print(f"    {model}: MAE={metrics['MAE']:.4f}")

print("\n" + "="*70)
print("✓ ALL CKPT2 DATA LOADED SUCCESSFULLY")
print("="*70)

# --- cell 3 ---
# ============================================================================
# CELL 3: Quick Intro (Demo Part 1 - 1 min)
# ============================================================================

from src.demo_utils import quick_intro, show_model_architecture

# Show intro
quick_intro()

# Show architecture
show_model_architecture(use_features=True)

# --- cell 4 ---
# ============================================================================
# CELL 4: Define Base Models for Stacking
# ============================================================================

print("="*70)
print("PREPARING BASE MODELS FOR STACKING")
print("="*70)

# These are the SAME models from CKPT2 (loaded from .pkl)
# We DON'T retrain them - we use them as-is for stacking
base_models_dict = {
    'ElasticNet': en_model,
    'RandomForest': rf_model,
    'XGBoost': xgb_model
}

print("\nBase models ready:")
for name in base_models_dict.keys():
    print(f"  ✓ {name}")

print("\nThese models will be used to generate predictions for stacking.")
print("="*70)

# --- cell 5 ---
# ============================================================================
# CELL 5: Train Config A (Demo Part 2 - 2 min)
# ============================================================================

from src.stacking import StackedEnsemble

print("\n" + "="*70)
print("TRAINING CONFIG A (PREDICTIONS-ONLY)")
print("="*70)

# Initialize Config A
config_a = StackedEnsemble(
    n_folds=5, 
    use_features=False,  # Config A: predictions only
    random_state=42
)

# Generate out-of-fold predictions
oof_preds_a = config_a.generate_oof_predictions(X_train, y_train, base_models_dict)

# Train meta-learner
config_a.train(X_train, y_train, oof_preds_a, save_dir='../checkpoints/config_a')

# Train base models on full training data
config_a.train_base_models_final(X_train, y_train, base_models_dict)

print("\n✓ Config A training complete!")

# --- cell 6 ---
# ============================================================================
# CELL 6: Train Config B (Demo Part 2 continued - 2 min)
# ============================================================================

print("\n" + "="*70)
print("TRAINING CONFIG B (PREDICTIONS + FEATURES)")
print("="*70)

# Initialize Config B
config_b = StackedEnsemble(
    n_folds=5,
    use_features=True,  # Config B: predictions + features
    random_state=42
)

# Generate out-of-fold predictions
oof_preds_b = config_b.generate_oof_predictions(X_train, y_train, base_models_dict)

# Train meta-learner
config_b.train(X_train, y_train, oof_preds_b, save_dir='../checkpoints/config_b')

# Train base models on full training data
config_b.train_base_models_final(X_train, y_train, base_models_dict)

print("\n✓ Config B training complete!")

# --- cell 7 ---
# ============================================================================
# CELL 7: Training Snapshot (Demo Part 3 - 1 min)
# ============================================================================

from src.demo_utils import show_training_snapshot

# Show Config B training logs
show_training_snapshot('../checkpoints/config_b')

# --- cell 8 ---
# ============================================================================
# CELL 8: Inference Demo (Demo Part 4 - 2 min)
# ============================================================================

from src.demo_utils import show_inference_demo

# Take sample batch
sample_size = 20
X_sample = X_test.iloc[:sample_size]
y_sample = y_test.iloc[:sample_size]

# Run inference demo
show_inference_demo(config_b, X_sample, y_sample, config_b.base_models)

# --- cell 9 ---
# ============================================================================
# CELL 9: Full Test Results (Demo Part 5 - 2 min)
# ============================================================================

print("="*70)
print("FULL TEST SET EVALUATION")
print("="*70)

# Get predictions from both configs
print("\nGenerating predictions...")
config_a_preds, _ = config_a.predict(X_test, config_a.base_models)
config_b_preds, _ = config_b.predict(X_test, config_b.base_models)

# Compute metrics
from src.eval import compute_metrics

config_a_results = compute_metrics(y_test, config_a_preds)
config_b_results = compute_metrics(y_test, config_b_preds)

print("\nRESULTS:")
print(f"  Config A - MAE: {config_a_results['MAE']:.4f}, RMSE: {config_a_results['RMSE']:.4f}")
print(f"  Config B - MAE: {config_b_results['MAE']:.4f}, RMSE: {config_b_results['RMSE']:.4f}")

# Best baseline
best_baseline_mae = baseline_results['XGBoost']['MAE']
print(f"  Best Baseline (XGBoost) - MAE: {best_baseline_mae:.4f}")

# Check improvement
if config_b_results['MAE'] < best_baseline_mae:
    improvement = ((best_baseline_mae - config_b_results['MAE']) / best_baseline_mae) * 100
    print(f"\n✓ Config B improves over best baseline by {improvement:.2f}%!")
else:
    print(f"\n✗ Config B MAE: {config_b_results['MAE']:.4f} vs Best Baseline: {best_baseline_mae:.4f}")

print("="*70)

# --- cell 10 ---
# ============================================================================
# CELL 10: Comparison Results (Demo Part 5 continued - 2 min)
# ============================================================================

from src.stacking import compare_configs
from src.demo_utils import show_comparison_results, create_comparison_plot

# Create comparison table
results_df = compare_configs(config_a_results, config_b_results, baseline_results)

# Show results
show_comparison_results(results_df)

# Create comparison plot
fig = create_comparison_plot(results_df, save_path='../results/comparison_plot.png')
plt.show()

# --- cell 11 ---
# ============================================================================
# CELL 11: ReAct Agent Decision (Demo Part 6 - 1 min)
# ============================================================================

from src.react_agent import ReActModelSelector

# Initialize agent
agent = ReActModelSelector()

# Check agent status
status = agent.get_status()
print(f"Agent initialized: {status}\n")

# Run agent decision
agent_state = agent.run(
    baseline_results=baseline_results,
    config_a_results=config_a_results,
    config_b_results=config_b_results,
    query="Which model should I use for production deployment?"
)

# Save decision log
agent.save_decision_log('../checkpoints/agent_decisions.json')

# --- cell 12 ---
# ============================================================================
# CELL 12: Demo Summary
# ============================================================================

print("\n" + "="*70)
print("CKPT3 DEMO COMPLETE - SUMMARY")
print("="*70)

print("\nKEY RESULTS:")
print(f"  Best Baseline:    XGBoost (MAE: {baseline_results['XGBoost']['MAE']:.4f})")
print(f"  Config A:         MAE: {config_a_results['MAE']:.4f}")
print(f"  Config B:         MAE: {config_b_results['MAE']:.4f}")
print(f"  Selected Model:   {agent_state['selected_model']}")
print(f"  Confidence:       {agent_state['confidence']:.0%}")

print("\nACHIEVEMENTS:")
print("  ✓ Implemented stacked ensemble with Config A and Config B")
print("  ✓ Config B enables context-aware adaptive weighting")
print("  ✓ LangGraph ReAct agent provides intelligent model selection")
print("  ✓ Complete comparison to all CKPT2 baselines")
print("  ✓ Forward-chaining OOF predictions ensure temporal integrity")

print("\nFILES CREATED:")
print("  ✓ checkpoints/config_a/training_log.json")
print("  ✓ checkpoints/config_b/training_log.json")
print("  ✓ checkpoints/agent_decisions.json")
print("  ✓ results/comparison_plot.png")

print("\n" + "="*70)
print("THANK YOU!")
print("="*70)
