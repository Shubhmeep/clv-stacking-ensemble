"""
Evaluation metrics and results presentation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred):
    """
    Compute MAE and RMSE.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with MAE and RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {'MAE': mae, 'RMSE': rmse}


def create_results_table(results_list):
    """
    Create formatted results table.
    
    Parameters:
    -----------
    results_list : list of dict
        List of result dictionaries with 'Model', 'MAE', 'RMSE' keys
        
    Returns:
    --------
    pd.DataFrame
        Formatted results table
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('MAE')
    return df


def print_results_table(results_df, title="Model Performance"):
    """
    Print formatted results table.
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")


def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Model name for title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def compare_models_plot(results_df, save_path=None):
    """
    Create bar chart comparing model performance.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with Model, MAE, RMSE columns
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MAE comparison
    axes[0].barh(results_df['Model'], results_df['MAE'])
    axes[0].set_xlabel('MAE (Mean Absolute Error)')
    axes[0].set_title('Model Comparison: MAE')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # RMSE comparison
    axes[1].barh(results_df['Model'], results_df['RMSE'])
    axes[1].set_xlabel('RMSE (Root Mean Squared Error)')
    axes[1].set_title('Model Comparison: RMSE')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def print_comparison_summary(results_df):
    """
    Print summary of model comparison.
    """
    best_mae = results_df.loc[results_df['MAE'].idxmin()]
    best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
    
    print("\n" + "="*60)
    print("BEST MODELS")
    print("="*60)
    print(f"Best MAE:  {best_mae['Model']} ({best_mae['MAE']:.4f})")
    print(f"Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Evaluation module")
    print("Use: from src.eval import compute_metrics, create_results_table, etc.")
