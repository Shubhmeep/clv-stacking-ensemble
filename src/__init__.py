"""
CLV Stacking Project - Source Code Package
"""

from .data import load_online_retail_ii, clean_data, get_dataset_stats, print_dataset_info
from .features import (
    make_window,
    create_temporal_splits,
    create_temporal_splits_multi,
    create_temporal_splits_multi_extended,
    get_feature_stats
)
from .baselines import (
    train_elasticnet, 
    train_random_forest, 
    train_xgboost,
    train_extra_trees,
    train_hist_gb,
    train_poisson,
    train_knn,
    train_svr,
    train_mlp,
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
from .stacking import StackedEnsemble, compare_configs
from .react_agent import LangGraphCLVOrchestrator, OrchestratorState
from .two_stage import TwoStageModel
from .analysis import feature_importance_table, segment_error_table
from .demo_utils import (
    quick_intro,
    show_model_architecture,
    show_training_snapshot,
    show_inference_demo,
    show_comparison_results,
    create_comparison_plot
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
    'create_temporal_splits_multi',
    'create_temporal_splits_multi_extended',
    'get_feature_stats',
    # Baselines
    'train_elasticnet',
    'train_random_forest',
    'train_xgboost',
    'train_extra_trees',
    'train_hist_gb',
    'train_poisson',
    'train_knn',
    'train_svr',
    'train_mlp',
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
    'print_comparison_summary',
    # CKPT3: Stacking
    'StackedEnsemble',
    'compare_configs',
    # CKPT5: LangGraph Orchestrator
    'LangGraphCLVOrchestrator',
    'OrchestratorState',
    # CKPT4: Two-Stage Model
    'TwoStageModel',
    # Analysis
    'feature_importance_table',
    'segment_error_table',
    # CKPT3: Demo Utils
    'quick_intro',
    'show_model_architecture',
    'show_training_snapshot',
    'show_inference_demo',
    'show_comparison_results',
    'create_comparison_plot'
]
