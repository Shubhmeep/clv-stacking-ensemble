"""
Feature engineering for CLV prediction with temporal windowing.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def make_window(df, cutoff_date, obs_months=6, horizon_months=3, min_purchases=1):
    """
    Create customer-level features and targets using temporal windowing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction data with InvoiceDate, CustomerID, InvoiceNo
    cutoff_date : str or pd.Timestamp
        Cutoff date T (e.g., '2010-12-01')
    obs_months : int
        Observation window length in months (default: 6)
    horizon_months : int
        Prediction horizon length in months (default: 3)
    min_purchases : int
        Minimum purchases in observation window to include customer
        
    Returns:
    --------
    pd.DataFrame
        Customer-level dataframe with features and target
    """
    cutoff_date = pd.to_datetime(cutoff_date)
    
    # Define time windows
    obs_start = cutoff_date - pd.DateOffset(months=obs_months)
    obs_end = cutoff_date
    horizon_end = cutoff_date + pd.DateOffset(months=horizon_months)
    
    print(f"\nCreating window:")
    print(f"  Observation: {obs_start.date()} to {obs_end.date()} ({obs_months} months)")
    print(f"  Horizon: {obs_end.date()} to {horizon_end.date()} ({horizon_months} months)")
    
    # Filter data for observation window
    obs_data = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] < obs_end)]
    print(f"  Observation transactions: {len(obs_data):,}")
    
    # Filter data for horizon window  
    horizon_data = df[(df['InvoiceDate'] >= obs_end) & (df['InvoiceDate'] < horizon_end)]
    print(f"  Horizon transactions: {len(horizon_data):,}")
    
    # Compute features for each customer
    features_list = []
    
    # Get customers who made purchases in observation window
    obs_customers = obs_data['CustomerID'].unique()
    print(f"  Customers in observation: {len(obs_customers):,}")
    
    for customer_id in obs_customers:
        cust_obs = obs_data[obs_data['CustomerID'] == customer_id]
        
        # Count unique invoices (transactions)
        unique_invoices = cust_obs['InvoiceNo'].nunique()
        
        # Skip if below minimum
        if unique_invoices < min_purchases:
            continue
        
        # Feature 1: freq - total transactions in observation window
        freq = unique_invoices
        
        # Feature 2: freq_3m - transactions in most recent 3 months
        recent_start = obs_end - pd.DateOffset(months=3)
        recent_data = cust_obs[cust_obs['InvoiceDate'] >= recent_start]
        freq_3m = recent_data['InvoiceNo'].nunique()
        
        # Feature 3: latetime - days since last purchase (recency)
        last_purchase = cust_obs['InvoiceDate'].max()
        latetime = (obs_end - last_purchase).days
        
        # Feature 4: earlytime - days since first purchase (tenure)
        # Use first purchase across entire dataset
        cust_all = df[df['CustomerID'] == customer_id]
        first_purchase = cust_all['InvoiceDate'].min()
        earlytime = (obs_end - first_purchase).days
        
        # Compute target: transactions in horizon window
        cust_horizon = horizon_data[horizon_data['CustomerID'] == customer_id]
        target = cust_horizon['InvoiceNo'].nunique()
        
        features_list.append({
            'CustomerID': customer_id,
            'freq': freq,
            'freq_3m': freq_3m,
            'latetime': latetime,
            'earlytime': earlytime,
            'target': target,
            'cutoff_date': obs_end
        })
    
    features_df = pd.DataFrame(features_list)
    print(f"  Final customers with features: {len(features_df):,}")
    print(f"  Customers with target > 0: {(features_df['target'] > 0).sum():,} ({(features_df['target'] > 0).mean()*100:.1f}%)")
    
    return features_df


def create_temporal_splits(df, train_cutoff, val_cutoff, test_cutoff, 
                          obs_months=6, horizon_months=3):
    """
    Create chronological train/validation/test splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction data
    train_cutoff : str
        Cutoff date for training (e.g., '2010-12-01')
    val_cutoff : str
        Cutoff date for validation (e.g., '2011-06-01')
    test_cutoff : str
        Cutoff date for test (e.g., '2011-09-01')
    obs_months : int
        Observation window length
    horizon_months : int
        Prediction horizon length
        
    Returns:
    --------
    tuple of pd.DataFrame
        (train_df, val_df, test_df) with features and targets
    """
    print("\n" + "="*60)
    print("CREATING TEMPORAL SPLITS")
    print("="*60)
    
    print("\n[1/3] Training Set")
    train_df = make_window(df, train_cutoff, obs_months, horizon_months)
    
    print("\n[2/3] Validation Set")
    val_df = make_window(df, val_cutoff, obs_months, horizon_months)
    
    print("\n[3/3] Test Set")
    test_df = make_window(df, test_cutoff, obs_months, horizon_months)
    
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print(f"Train: {len(train_df):,} customers")
    print(f"Val:   {len(val_df):,} customers")
    print(f"Test:  {len(test_df):,} customers")
    print("="*60 + "\n")
    
    return train_df, val_df, test_df


def get_feature_stats(features_df):
    """
    Get statistics for features.
    """
    stats = features_df[['freq', 'freq_3m', 'latetime', 'earlytime', 'target']].describe()
    return stats


if __name__ == "__main__":
    print("Feature engineering module")
    print("Use: from src.features import make_window, create_temporal_splits")
