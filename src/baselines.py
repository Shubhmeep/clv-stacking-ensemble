"""
Baseline models for CLV prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def train_elasticnet(X_train, y_train, alpha=0.1, l1_ratio=0.5):
    """
    Train ElasticNet baseline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training targets
    alpha : float
        Regularization strength
    l1_ratio : float
        L1 vs L2 ratio (0=Ridge, 1=Lasso)
        
    Returns:
    --------
    sklearn model
        Trained ElasticNet model
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Train RandomForest baseline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training targets
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
        
    Returns:
    --------
    sklearn model
        Trained RandomForest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1):
    """
    Train XGBoost baseline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training targets
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
        
    Returns:
    --------
    xgboost model
        Trained XGBoost model
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return None
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_bgnbd_baseline(train_df, horizon_months=3):
    """
    Train BG/NBD probabilistic baseline using lifetimes library.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with freq, latetime, earlytime columns
    horizon_months : int
        Prediction horizon in months (for expected purchases)
        
    Returns:
    --------
    lifetimes model
        Trained BG/NBD model or None if convergence fails
    """
    try:
        from lifetimes import BetaGeoFitter
    except ImportError:
        print("lifetimes not installed. Install with: pip install lifetimes")
        return None
    
    # Prepare data in lifetimes format
    rfm_df = train_df.copy()
    
    # frequency: repeat purchases (must be >= 0)
    rfm_df['frequency'] = (rfm_df['freq'] - 1).clip(lower=0)
    
    # recency: time between first and last purchase
    rfm_df['recency'] = rfm_df['earlytime'] - rfm_df['latetime']
    
    # For customers with only 1 purchase, recency MUST be 0
    rfm_df.loc[rfm_df['freq'] == 1, 'recency'] = 0
    
    # T: customer age (time from first purchase to observation end)
    rfm_df['T'] = rfm_df['earlytime']
    
    # Additional validation
    rfm_df['recency'] = rfm_df['recency'].clip(lower=0)
    rfm_df.loc[rfm_df['frequency'] == 0, 'recency'] = 0
    
    # Remove any invalid rows
    valid_mask = (
        (rfm_df['frequency'] >= 0) &
        (rfm_df['recency'] >= 0) &
        (rfm_df['T'] > 0) &
        (rfm_df['recency'] <= rfm_df['T'])
    )
    
    rfm_df = rfm_df[valid_mask]
    
    print(f"Training BG/NBD on {len(rfm_df)} valid customers...")
    
    # Train model with regularization to help convergence
    try:
        bgf = BetaGeoFitter(penalizer_coef=0.01)  # Add regularization
        bgf.fit(
            rfm_df['frequency'], 
            rfm_df['recency'], 
            rfm_df['T'],
            verbose=False,  # Suppress convergence warnings
            tol=1e-6  # Relaxed tolerance
        )
        print("✓ BG/NBD training successful")
        return bgf
    
    except Exception as e:
        print(f"⚠ BG/NBD failed to converge: {str(e)[:100]}")
        print("  This is common with this dataset. Skipping BG/NBD baseline.")
        return None

def predict_bgnbd(model, test_df, horizon_months=3):
    """
    Generate predictions from BG/NBD model.
    
    Parameters:
    -----------
    model : lifetimes.BetaGeoFitter
        Trained BG/NBD model
    test_df : pd.DataFrame
        Test data with freq, latetime, earlytime
    horizon_months : int
        Prediction horizon
        
    Returns:
    --------
    np.array
        Predicted transaction counts
    """
    # Convert to lifetimes format
    rfm_df = test_df.copy()
    
    # frequency: repeat purchases
    rfm_df['frequency'] = (rfm_df['freq'] - 1).clip(lower=0)
    
    # recency: time between first and last purchase
    rfm_df['recency'] = rfm_df['earlytime'] - rfm_df['latetime']
    
    # For customers with only 1 purchase, recency MUST be 0
    rfm_df.loc[rfm_df['freq'] == 1, 'recency'] = 0
    
    # T: customer age
    rfm_df['T'] = rfm_df['earlytime']
    
    # Validation
    rfm_df['recency'] = rfm_df['recency'].clip(lower=0)
    rfm_df.loc[rfm_df['frequency'] == 0, 'recency'] = 0
    
    # Predict expected purchases in next horizon_months * 30 days
    t = horizon_months * 30
    predictions = model.predict(t, rfm_df['frequency'], rfm_df['recency'], rfm_df['T'])
    
    # Make a writable copy
    predictions = np.array(predictions.values, dtype=np.float64)
    
    # Handle infinite or invalid values
    valid_mask = np.isfinite(predictions)
    if valid_mask.sum() > 0:
        mean_pred = predictions[valid_mask].mean()
    else:
        mean_pred = 0.0
    
    predictions[~valid_mask] = mean_pred
    
    # Clip to reasonable range (0 to 100 transactions)
    predictions = np.clip(predictions, 0, 100)
    
    return predictions


def simple_averaging(predictions_dict):
    """
    Simple averaging baseline: mean of base model predictions.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
        
    Returns:
    --------
    np.array
        Averaged predictions
    """
    predictions_array = np.array(list(predictions_dict.values()))
    return predictions_array.mean(axis=0)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str
        Model name for display
        
    Returns:
    --------
    dict
        Dictionary with MAE and RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    results = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse
    }
    
    return results


if __name__ == "__main__":
    print("Baseline models module")
    print("Use: from src.baselines import train_elasticnet, train_random_forest, etc.")
