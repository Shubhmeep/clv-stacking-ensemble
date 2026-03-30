"""
CLV Stacking Project - Source Code Package
"""

from .data import load_online_retail_ii, clean_data, get_dataset_stats, print_dataset_info
from .features import make_window, create_temporal_splits, get_feature_stats
from .baselines import (
    train_elasticnet, 
    train_random_forest, 
    train_xgboost,
    train_bgnbd_baseline,
    predict_bgnbd,
    simple_averaging,
    evaluate_model
)
from .eval import (
    compute_metrics,
    create_results_table,
    print_results_table,
    plot_predictions_vs_actual,
    compare_models_plot,
    print_comparison_summary
)

__all__ = [
    # Data
    'load_online_retail_ii',
    'clean_data',
    'get_dataset_stats',
    'print_dataset_info',
    # Features
    'make_window',
    'create_temporal_splits',
    'get_feature_stats',
    # Baselines
    'train_elasticnet',
    'train_random_forest',
    'train_xgboost',
    'train_bgnbd_baseline',
    'predict_bgnbd',
    'simple_averaging',
    'evaluate_model',
    # Evaluation
    'compute_metrics',
    'create_results_table',
    'print_results_table',
    'plot_predictions_vs_actual',
    'compare_models_plot',
    'print_comparison_summary'
]
