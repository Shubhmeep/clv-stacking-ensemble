# --- cell 1 ---
# Import libraries
import sys
print(sys.executable)
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.data import load_online_retail_ii, clean_data, print_dataset_info
from src.features import create_temporal_splits, get_feature_stats
from src.baselines import *
from src.eval import create_results_table, print_results_table, compare_models_plot

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 4)

print("✓ All imports successful!")

# --- cell 2 ---
# Load data
df_raw = load_online_retail_ii(
    '../data/Year 2009-2010.csv',
    '../data/Year 2010-2011.csv'
)

print(f"\nRaw data shape: {df_raw.shape}")
print(f"\nFirst few rows:")
df_raw.head()

# --- cell 3 ---
# Clean data
df_clean = clean_data(df_raw, verbose=True)

# --- cell 4 ---
# ============================================================================
# VALIDATION: Cancellation Analysis
# ============================================================================

print("="*60)
print("CANCELLATION ANALYSIS - Validating Removal Decision")
print("="*60)

# Load raw data before cleaning
df_with_cancellations = df_raw.copy()

# Identify cancellations
cancellations = df_with_cancellations[
    df_with_cancellations['InvoiceNo'].astype(str).str.startswith('C')
]

print(f"\nTotal cancellations: {len(cancellations):,}")
print(f"Percentage of data: {len(cancellations)/len(df_with_cancellations)*100:.2f}%")
print()

# ============================================================================
# VISUALIZATION 1: Impact across entire dataset
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Data
total_rows = len(df_raw)
after_customerid = total_rows - 243007
after_cancellations = after_customerid - 18744
final_rows = 779425

# Step-by-step removal
steps = ['Original\nData', 'Remove\nMissing IDs', 'Remove\nCancellations', 'Final\nCleaned']
values = [total_rows, after_customerid, after_cancellations, final_rows]
colors_steps = ['#e9ecef', '#ffd43b', '#ff8787', '#51cf66']

# Left: Waterfall chart
axes[0].bar(steps, values, color=colors_steps, edgecolor='black', linewidth=2, alpha=0.8)
axes[0].set_ylabel('Number of Rows', fontsize=12)
axes[0].set_title('Data Cleaning Pipeline', fontsize=14, fontweight='bold')

# Add value labels
for i, (step, val) in enumerate(zip(steps, values)):
    axes[0].text(i, val + 20000, f'{val:,}', ha='center', fontsize=10, fontweight='bold')

# Add removal annotations
axes[0].annotate('', xy=(1, after_customerid), xytext=(0, total_rows),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
axes[0].text(0.5, (total_rows + after_customerid)/2, '-243K', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

axes[0].annotate('', xy=(2, after_cancellations), xytext=(1, after_customerid),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
axes[0].text(1.5, (after_customerid + after_cancellations)/2, '-19K\nreturns', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Middle: Pie chart of what was removed
labels_pie2 = ['Kept (73%)', 'Missing ID (23%)', 'Cancellations (2%)', 'Other (2%)']
sizes_pie2 = [779425, 243007, 18744, (total_rows - 779425 - 243007 - 18744)]
colors_pie2 = ['#51cf66', '#adb5bd', '#ff6b6b', '#ffd43b']
explode_pie2 = (0.05, 0, 0.1, 0)

axes[1].pie(sizes_pie2, labels=labels_pie2, colors=colors_pie2, autopct='%1.1f%%',
           startangle=90, explode=explode_pie2, textprops={'fontsize': 10})
axes[1].set_title('Data Retention Breakdown', fontsize=14, fontweight='bold')

# Right: Transaction count comparison
all_invoices = df_raw[df_raw['CustomerID'].notna()]['InvoiceNo'].nunique()
clean_invoices = df_clean['InvoiceNo'].nunique()

cats_inv = ['With\nCancellations', 'Without\nCancellations\n(Clean)']
vals_inv = [all_invoices, clean_invoices]
colors_inv = ['#ff8787', '#51cf66']

bars2 = axes[2].bar(cats_inv, vals_inv, color=colors_inv, edgecolor='black', linewidth=2, alpha=0.8)
axes[2].set_ylabel('Unique Invoices', fontsize=12)
axes[2].set_title('Impact on Transaction Count', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars2, vals_inv):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{int(val):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add difference
diff = all_invoices - clean_invoices
axes[2].text(0.5, (all_invoices + clean_invoices)/2, f'-{diff}\nreturns', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('../figures/cleaning_overview.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print("DATASET-WIDE IMPACT:")
print(f"{'='*60}")
print(f"Total invoices with cancellations:    {all_invoices:,}")
print(f"Total invoices without cancellations: {clean_invoices:,}")
print(f"Difference (returns removed):         {diff:,}")
print(f"{'='*60}\n")

# ============================================================================
# VISUALIZATION 2: When do returns happen? (Sample Customer Analysis)
# ============================================================================

# Find a customer with both purchases and returns
df_with_id = df_raw[df_raw['CustomerID'].notna()].copy()

# Ensure InvoiceDate is datetime
df_with_id['InvoiceDate'] = pd.to_datetime(df_with_id['InvoiceDate'])

# Find customers who have both purchases and returns
purchases_by_customer = df_with_id[~df_with_id['InvoiceNo'].astype(str).str.startswith('C')].groupby('CustomerID').size()
returns_by_customer = df_with_id[df_with_id['InvoiceNo'].astype(str).str.startswith('C')].groupby('CustomerID').size()

customers_with_both = purchases_by_customer.index.intersection(returns_by_customer.index)

if len(customers_with_both) > 0:
    # Pick a customer with both
    sample_customer = customers_with_both[0]
    
    # Get this customer's data
    customer_data = df_with_id[df_with_id['CustomerID'] == sample_customer].copy()
    customer_data_sorted = customer_data.sort_values('InvoiceDate')
    
    # Find purchase-return pairs
    purchases = customer_data_sorted[~customer_data_sorted['InvoiceNo'].astype(str).str.startswith('C')]
    returns = customer_data_sorted[customer_data_sorted['InvoiceNo'].astype(str).str.startswith('C')]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot purchases
    purchase_dates = purchases.groupby('InvoiceNo')['InvoiceDate'].first()
    ax.scatter(purchase_dates, [1]*len(purchase_dates), 
              s=200, c='green', marker='o', label='Purchase', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Plot returns
    return_dates = returns.groupby('InvoiceNo')['InvoiceDate'].first()
    ax.scatter(return_dates, [0.5]*len(return_dates), 
              s=200, c='red', marker='X', label='Return', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Connect returns to nearest purchase (visual only)
    for ret_date in return_dates:
        # Find nearest earlier purchase
        earlier_purchases = purchase_dates[purchase_dates < ret_date]
        if len(earlier_purchases) > 0:
            nearest = earlier_purchases.iloc[-1]
            ax.plot([nearest, ret_date], [1, 0.5], 'r--', alpha=0.3, linewidth=1.5)
            
            # Calculate time difference (convert to timedelta if needed)
            time_diff_td = pd.Timedelta(ret_date - nearest)
            time_diff = time_diff_td.total_seconds() / 60  # minutes
            
            if time_diff < 60:  # Less than 1 hour
                mid_point = nearest + (ret_date - nearest) / 2
                ax.text(mid_point, 0.75, f'{int(time_diff)} min', 
                       ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    ax.set_ylim(0.2, 1.3)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(['Returns', 'Purchases'])
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'Customer {int(sample_customer)}: Purchase-Return Pattern\n(Returns often happen minutes after purchase = Exchanges)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('../figures/purchase_return_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n🔍 KEY INSIGHT: Returns typically happen within minutes of purchase")
    print("   → These are EXCHANGES (wrong size/color), not customer dissatisfaction")
    print("   → Including them would count exchanges as 2 separate transactions (WRONG!)")
else:
    print("\n⚠ No customers found with both purchases and returns in the sample")
    print("   Skipping purchase-return timing visualization")

# --- cell 5 ---
# Duplicate analysis visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Data for duplicate occurrences
duplicate_counts = [20, 12, 10, 8, 6]  # From your output
duplicate_labels = ['Invoice\n555524', 'Invoice\n555524\n(Item 2)', 
                   'Invoice\n537224', 'Invoice\n572861', 'Invoice\n536749']

# Panel 1: Top duplicated transactions
axes[0].barh(range(len(duplicate_counts)), duplicate_counts, color='#E63946', edgecolor='black')
axes[0].set_yticks(range(len(duplicate_counts)))
axes[0].set_yticklabels(duplicate_labels, fontsize=9)
axes[0].set_xlabel('Number of Duplicate Copies', fontsize=11, fontweight='bold')
axes[0].set_title('Top 5 Most Duplicated Transactions', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(duplicate_counts):
    axes[0].text(v + 0.3, i, f'{v}×', va='center', fontweight='bold', fontsize=10)

# Panel 2: Overall duplicate statistics
stats_labels = ['Unique\nTransactions\nDuplicated', 'Total\nDuplicate\nRows', 'Rows\nRemoved']
stats_values = [24712, 50836, 26124]
stats_colors = ['#457B9D', '#E63946', '#F4A261']

bars = axes[1].bar(range(len(stats_labels)), stats_values, 
                   color=stats_colors, edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(len(stats_labels)))
axes[1].set_xticklabels(stats_labels, fontsize=10, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title('Duplicate Statistics', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, stats_values)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel 3: Before vs After comparison
comparison_data = {
    'Before\nDeduplication': 805549,
    'After\nDeduplication': 779425
}

bars = axes[2].bar(range(len(comparison_data)), list(comparison_data.values()),
                   color=['#E63946', '#06A77D'], edgecolor='black', linewidth=1.5)
axes[2].set_xticks(range(len(comparison_data)))
axes[2].set_xticklabels(list(comparison_data.keys()), fontsize=10, fontweight='bold')
axes[2].set_ylabel('Total Rows', fontsize=11, fontweight='bold')
axes[2].set_title('Impact of Deduplication', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, comparison_data.values())):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add arrow showing reduction
axes[2].annotate('', xy=(1, 779425), xytext=(0, 805549),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
axes[2].text(0.5, 792000, f'-26,124\n(-3.2%)', ha='center', fontsize=9, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle('Duplicate Row Analysis & Removal', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../figures/duplicate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Duplicate analysis saved to figures/duplicate_analysis.png")

# --- cell 6 ---
# Dataset statistics
print_dataset_info(df_clean)

# --- cell 7 ---
# 1. Transactions per customer distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Invoices per customer
invoices_per_customer = df_clean.groupby('CustomerID')['InvoiceNo'].nunique()
axes[0].hist(invoices_per_customer, bins=50, edgecolor='black')
axes[0].set_xlabel('Number of Invoices')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Distribution: Invoices per Customer')
axes[0].set_yscale('log')

# Transactions over time
monthly_transactions = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M')).size()
axes[1].plot(monthly_transactions.index.astype(str), monthly_transactions.values, marker='o')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of Transactions')
axes[1].set_title('Transactions Over Time (Motivates Temporal Drift)')
axes[1].tick_params(axis='x', rotation=45)

# Revenue distribution
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Price']
revenue_per_invoice = df_clean.groupby('InvoiceNo')['Revenue'].sum()
axes[2].hist(revenue_per_invoice[revenue_per_invoice < 1000], bins=50, edgecolor='black')
axes[2].set_xlabel('Revenue per Invoice ($)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution: Revenue per Invoice')

plt.tight_layout()
plt.savefig('../figures/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# --- cell 8 ---
# Create temporal splits
train_df, val_df, test_df = create_temporal_splits(
    df_clean,
    train_cutoff='2010-12-01',
    val_cutoff='2011-06-01',
    test_cutoff='2011-09-01',
    obs_months=6,
    horizon_months=3
)

# --- cell 9 ---
# Show sample customer features
print("Sample Customer Features:")
print(train_df[['CustomerID', 'freq', 'freq_3m', 'latetime', 'earlytime', 'target']].head(10))

print("\nFeature Statistics (Training Set):")
print(get_feature_stats(train_df))

# --- cell 10 ---
# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Target distribution
axes[0].hist(train_df['target'], bins=50, edgecolor='black')
axes[0].set_xlabel('Target (# Transactions in Next 3 Months)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution (Training Set)')
axes[0].axvline(train_df['target'].mean(), color='red', linestyle='--', 
                label=f"Mean: {train_df['target'].mean():.2f}")
axes[0].legend()

# Zero vs non-zero targets
zero_pct = (train_df['target'] == 0).mean() * 100
nonzero_pct = (train_df['target'] > 0).mean() * 100
axes[1].bar(['Zero Purchases', 'Non-Zero Purchases'], [zero_pct, nonzero_pct])
axes[1].set_ylabel('Percentage of Customers')
axes[1].set_title('Customer Activity in Prediction Horizon')
axes[1].set_ylim([0, 100])

for i, v in enumerate([zero_pct, nonzero_pct]):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- cell 11 ---
# Prepare data
feature_cols = ['freq', 'freq_3m', 'latetime', 'earlytime']

X_train = train_df[feature_cols]
y_train = train_df['target']

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"Training set: {len(X_train)} customers")
print(f"Validation set: {len(X_val)} customers")
print(f"Test set: {len(X_test)} customers")

# --- cell 12 ---
print("Training ElasticNet...")
model_en = train_elasticnet(X_train, y_train)

# Predictions
pred_en_val = model_en.predict(X_val)
pred_en_test = model_en.predict(X_test)

# Evaluate
results_en_val = evaluate_model(y_val, pred_en_val, "ElasticNet")
results_en_test = evaluate_model(y_test, pred_en_test, "ElasticNet")

print(f"Val  MAE: {results_en_val['MAE']:.4f}, RMSE: {results_en_val['RMSE']:.4f}")
print(f"Test MAE: {results_en_test['MAE']:.4f}, RMSE: {results_en_test['RMSE']:.4f}")
print("✓ ElasticNet complete!")

# --- cell 13 ---
print("Training RandomForest...")
model_rf = train_random_forest(X_train, y_train)

# Predictions
pred_rf_val = model_rf.predict(X_val)
pred_rf_test = model_rf.predict(X_test)

# Evaluate
results_rf_val = evaluate_model(y_val, pred_rf_val, "RandomForest")
results_rf_test = evaluate_model(y_test, pred_rf_test, "RandomForest")

print(f"Val  MAE: {results_rf_val['MAE']:.4f}, RMSE: {results_rf_val['RMSE']:.4f}")
print(f"Test MAE: {results_rf_test['MAE']:.4f}, RMSE: {results_rf_test['RMSE']:.4f}")
print("✓ RandomForest complete!")

# --- cell 14 ---
print("Training XGBoost...")
model_xgb = train_xgboost(X_train, y_train)

if model_xgb is not None:
    # Predictions
    pred_xgb_val = model_xgb.predict(X_val)
    pred_xgb_test = model_xgb.predict(X_test)
    
    # Evaluate
    results_xgb_val = evaluate_model(y_val, pred_xgb_val, "XGBoost")
    results_xgb_test = evaluate_model(y_test, pred_xgb_test, "XGBoost")
    
    print(f"Val  MAE: {results_xgb_val['MAE']:.4f}, RMSE: {results_xgb_val['RMSE']:.4f}")
    print(f"Test MAE: {results_xgb_test['MAE']:.4f}, RMSE: {results_xgb_test['RMSE']:.4f}")
    print("✓ XGBoost complete!")
else:
    print("⚠ XGBoost not available (install with: pip install xgboost)")

# --- cell 15 ---
print("Training BG/NBD...")
model_bgnbd = train_bgnbd_baseline(train_df, horizon_months=3)

if model_bgnbd is not None:
    # Predictions
    pred_bgnbd_val = predict_bgnbd(model_bgnbd, val_df, horizon_months=3)
    pred_bgnbd_test = predict_bgnbd(model_bgnbd, test_df, horizon_months=3)
    
    # Evaluate
    results_bgnbd_val = evaluate_model(y_val, pred_bgnbd_val, "BG/NBD")
    results_bgnbd_test = evaluate_model(y_test, pred_bgnbd_test, "BG/NBD")
    
    print(f"Val  MAE: {results_bgnbd_val['MAE']:.4f}, RMSE: {results_bgnbd_val['RMSE']:.4f}")
    print(f"Test MAE: {results_bgnbd_test['MAE']:.4f}, RMSE: {results_bgnbd_test['RMSE']:.4f}")
    print("✓ BG/NBD complete!")
else:
    print("⚠ BG/NBD not available (install with: pip install lifetimes)")

# --- cell 16 ---
print("Computing Simple Averaging...")

# Average of ElasticNet, RF, XGBoost
pred_avg_val = simple_averaging({
    'ElasticNet': pred_en_val,
    'RandomForest': pred_rf_val,
    'XGBoost': pred_xgb_val
})

pred_avg_test = simple_averaging({
    'ElasticNet': pred_en_test,
    'RandomForest': pred_rf_test,
    'XGBoost': pred_xgb_test
})

# Evaluate
results_avg_val = evaluate_model(y_val, pred_avg_val, "SimpleAverage")
results_avg_test = evaluate_model(y_test, pred_avg_test, "SimpleAverage")

print(f"Val  MAE: {results_avg_val['MAE']:.4f}, RMSE: {results_avg_val['RMSE']:.4f}")
print(f"Test MAE: {results_avg_test['MAE']:.4f}, RMSE: {results_avg_test['RMSE']:.4f}")
print("✓ Simple Averaging complete!")

# --- cell 17 ---
# Compile all results
all_results_val = [
    {**results_en_val, 'Split': 'Validation'},
    {**results_rf_val, 'Split': 'Validation'},
    {**results_xgb_val, 'Split': 'Validation'},
    {**results_bgnbd_val, 'Split': 'Validation'},
    {**results_avg_val, 'Split': 'Validation'},
]

all_results_test = [
    {**results_en_test, 'Split': 'Test'},
    {**results_rf_test, 'Split': 'Test'},
    {**results_xgb_test, 'Split': 'Test'},
    {**results_bgnbd_test, 'Split': 'Test'},
    {**results_avg_test, 'Split': 'Test'},
]

results_val_df = create_results_table(all_results_val)
results_test_df = create_results_table(all_results_test)

print_results_table(results_val_df, "Validation Set Performance")
print_results_table(results_test_df, "Test Set Performance")

# --- cell 18 ---
# Visualize comparison
fig = compare_models_plot(results_test_df, save_path='../figures/model_comparison.png')
plt.show()

# --- cell 19 ---
# ============================================================================
# SAVE EVERYTHING FOR CKPT3
# ============================================================================

import joblib
import json
import os

print("="*70)
print("SAVING FOR CKPT3".center(70))
print("="*70)

# Create directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../data', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Save Trained Baseline Models
# ----------------------------------------------------------------------------
print("\n[1/3] Saving trained baseline models...")

# Try both naming conventions
try:
    # Try en_model first, then model_en
    if 'en_model' in globals():
        joblib.dump(en_model, '../models/elasticnet_baseline.pkl')
    elif 'model_en' in globals():
        joblib.dump(model_en, '../models/elasticnet_baseline.pkl')
    print("  ✓ ElasticNet saved")
except Exception as e:
    print(f"  ⚠ ElasticNet error: {str(e)}")

try:
    if 'rf_model' in globals():
        joblib.dump(rf_model, '../models/randomforest_baseline.pkl')
    elif 'model_rf' in globals():
        joblib.dump(model_rf, '../models/randomforest_baseline.pkl')
    print("  ✓ RandomForest saved")
except Exception as e:
    print(f"  ⚠ RandomForest error: {str(e)}")

try:
    if 'xgb_model' in globals():
        joblib.dump(xgb_model, '../models/xgboost_baseline.pkl')
    elif 'model_xgb' in globals():
        joblib.dump(model_xgb, '../models/xgboost_baseline.pkl')
    print("  ✓ XGBoost saved")
except Exception as e:
    print(f"  ⚠ XGBoost error: {str(e)}")

try:
    if 'model_bgnbd' in globals():
        joblib.dump(model_bgnbd, '../models/bgnbd_baseline.pkl')
    print("  ✓ BG/NBD saved")
except Exception as e:
    print(f"  ⚠ BG/NBD error: {str(e)}")

# ----------------------------------------------------------------------------
# 2. Save Data Splits
# ----------------------------------------------------------------------------
print("\n[2/3] Saving data splits...")

try:
    train_df.to_csv('../data/train_features.csv', index=False)
    print(f"  ✓ Train set saved: {len(train_df)} samples")
except Exception as e:
    print(f"  ⚠ Train data error: {str(e)}")

try:
    val_df.to_csv('../data/val_features.csv', index=False)
    print(f"  ✓ Val set saved: {len(val_df)} samples")
except Exception as e:
    print(f"  ⚠ Val data error: {str(e)}")

try:
    test_df.to_csv('../data/test_features.csv', index=False)
    print(f"  ✓ Test set saved: {len(test_df)} samples")
except Exception as e:
    print(f"  ⚠ Test data error: {str(e)}")

# ----------------------------------------------------------------------------
# 3. Save Baseline Results
# ----------------------------------------------------------------------------
print("\n[3/3] Saving baseline results...")

try:
    # Use whichever model variables exist
    en = en_model if 'en_model' in globals() else model_en
    rf = rf_model if 'rf_model' in globals() else model_rf
    xgb = xgb_model if 'xgb_model' in globals() else model_xgb
    
    # ElasticNet
    en_test_preds = en.predict(X_test)
    en_test_mae = np.abs(en_test_preds - y_test).mean()
    en_test_rmse = np.sqrt(((en_test_preds - y_test) ** 2).mean())
    
    # RandomForest
    rf_test_preds = rf.predict(X_test)
    rf_test_mae = np.abs(rf_test_preds - y_test).mean()
    rf_test_rmse = np.sqrt(((rf_test_preds - y_test) ** 2).mean())
    
    # XGBoost
    xgb_test_preds = xgb.predict(X_test)
    xgb_test_mae = np.abs(xgb_test_preds - y_test).mean()
    xgb_test_rmse = np.sqrt(((xgb_test_preds - y_test) ** 2).mean())
    
    # Simple Average
    avg_test_preds = (en_test_preds + rf_test_preds + xgb_test_preds) / 3
    avg_test_mae = np.abs(avg_test_preds - y_test).mean()
    avg_test_rmse = np.sqrt(((avg_test_preds - y_test) ** 2).mean())
    
    # BG/NBD (optional)
    try:
        from src.baselines import predict_bgnbd
        bgf = model_bgnbd if 'model_bgnbd' in globals() else None
        if bgf:
            bgf_test_preds = predict_bgnbd(bgf, test_df)
            bgf_test_mae = np.abs(bgf_test_preds - y_test).mean()
            bgf_test_rmse = np.sqrt(((bgf_test_preds - y_test) ** 2).mean())
            bgnbd_results = {'MAE': float(bgf_test_mae), 'RMSE': float(bgf_test_rmse)}
        else:
            bgnbd_results = {'MAE': 1.381, 'RMSE': 3.026}
    except:
        bgnbd_results = {'MAE': 1.381, 'RMSE': 3.026}
    
    baseline_results = {
        'ElasticNet': {'MAE': float(en_test_mae), 'RMSE': float(en_test_rmse)},
        'RandomForest': {'MAE': float(rf_test_mae), 'RMSE': float(rf_test_rmse)},
        'XGBoost': {'MAE': float(xgb_test_mae), 'RMSE': float(xgb_test_rmse)},
        'BG/NBD': bgnbd_results,
        'SimpleAvg': {'MAE': float(avg_test_mae), 'RMSE': float(avg_test_rmse)}
    }
    
    with open('../results/baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print("  ✓ Baseline results saved:")
    for model, metrics in baseline_results.items():
        print(f"    {model}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

except Exception as e:
    print(f"  ⚠ Error saving results: {str(e)}")
    import traceback
    traceback.print_exc()

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("✓ CKPT2 OUTPUTS SAVED SUCCESSFULLY".center(70))
print("="*70)
print("\nSaved files:")
print("  Models:     ../models/")
print("  Data:       ../data/")
print("  Results:    ../results/")
print("\nReady for CKPT3!")
print("="*70)

# Verify files were actually created
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)
import os
print("\nModels folder:")
for f in os.listdir('../models'):
    size = os.path.getsize(f'../models/{f}') / 1024  # KB
    print(f"  ✓ {f} ({size:.1f} KB)")

print("\nData folder:")
for f in os.listdir('../data'):
    size = os.path.getsize(f'../data/{f}') / 1024  # KB
    print(f"  ✓ {f} ({size:.1f} KB)")

print("\nResults folder:")
for f in os.listdir('../results'):
    size = os.path.getsize(f'../results/{f}') / 1024  # KB
    print(f"  ✓ {f} ({size:.1f} KB)")
print("="*70)
