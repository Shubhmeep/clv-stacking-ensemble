"""
Stacked Ensemble Implementation for CLV Prediction.

Two configurations:
- Config A: Meta-learner uses only base model predictions
- Config B: Meta-learner uses predictions + original features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
import json
import os
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')


class StackedEnsemble:
    """
    Stacked ensemble with forward-chaining cross-validation.
    
    Parameters:
    -----------
    meta_learner : sklearn model
        Meta-model to combine base predictions (default: ElasticNet)
    n_folds : int
        Number of forward-chaining folds for OOF predictions
    use_features : bool
        If True (Config B), concatenate original features with predictions
        If False (Config A), use only predictions
    """
    
    def __init__(self, meta_learner=None, n_folds=5, use_features=True, random_state=42):
        # Avoid truthiness checks on sklearn estimators (can access unfitted attributes)
        self.meta_learner = meta_learner if meta_learner is not None else ElasticNet(
            alpha=0.1, l1_ratio=0.5, random_state=random_state
        )
        self.n_folds = n_folds
        self.use_features = use_features
        self.random_state = random_state
        self.base_models = {}
        self.training_log = []
        
        # Configuration name
        self.config_name = "Config_B" if use_features else "Config_A"
        
    def generate_oof_predictions(self, X, y, base_models_dict):
        """
        Generate out-of-fold predictions using forward-chaining CV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training targets
        base_models_dict : dict
            Dictionary with model names and untrained model objects
            Example: {'ElasticNet': ElasticNet(), 'XGBoost': XGBRegressor()}
            
        Returns:
        --------
        pd.DataFrame
            Out-of-fold predictions from each base model
        """
        print(f"\n{'='*60}")
        print(f"GENERATING OUT-OF-FOLD PREDICTIONS ({self.config_name})")
        print(f"{'='*60}")
        print(f"Data shape: {X.shape}")
        print(f"Number of folds: {self.n_folds}")
        print(f"Base models: {list(base_models_dict.keys())}")
        
        # Initialize OOF prediction storage
        oof_predictions = pd.DataFrame(index=X.index, columns=list(base_models_dict.keys()), dtype=float)

        # Forward-chaining with proper time-series splits
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\n[Fold {fold_idx}/{self.n_folds}]")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            print(f"  Train size: {len(X_train_fold)}, Val size: {len(X_val_fold)}")

            # Train each base model and generate predictions
            for model_name, model_template in base_models_dict.items():
                # Clone the model
                from sklearn.base import clone
                model = clone(model_template)

                # Train
                model.fit(X_train_fold, y_train_fold)

                # Predict on validation fold
                val_preds = model.predict(X_val_fold)

                # Store OOF predictions
                oof_predictions.loc[X_val_fold.index, model_name] = val_preds

                print(f"    {model_name}: MAE={np.abs(val_preds - y.iloc[val_idx]).mean():.4f}")
        
        # Verify no missing predictions
        missing_preds = oof_predictions.isna().sum().sum()
        if missing_preds > 0:
            print(f"\n⚠ Warning: {missing_preds} missing OOF predictions")

        valid_rows = oof_predictions.dropna().shape[0]
        print(f"\n✓ OOF predictions generated: {oof_predictions.shape} (valid rows: {valid_rows})")
        return oof_predictions
    
    def train(self, X_train, y_train, oof_predictions, save_dir=None, save_models=True):
        """
        Train the meta-learner on out-of-fold predictions.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Original training features
        y_train : pd.Series
            Training targets
        oof_predictions : pd.DataFrame
            Out-of-fold predictions from base models
        save_dir : str, optional
            Directory to save model checkpoint
            
        Returns:
        --------
        self
        """
        print(f"\n{'='*60}")
        print(f"TRAINING META-LEARNER ({self.config_name})")
        print(f"{'='*60}")
        
        # Align training data to rows with valid OOF predictions
        missing_rows = oof_predictions.isna().any(axis=1)
        if missing_rows.any():
            valid_idx = oof_predictions.index[~missing_rows]
            dropped = len(X_train) - len(valid_idx)
            if dropped > 0:
                print(f"⚠ Dropping {dropped} rows without full OOF predictions")
            oof_predictions = oof_predictions.loc[valid_idx]
            X_train_aligned = X_train.loc[valid_idx]
            y_train_aligned = y_train.loc[valid_idx]
        else:
            X_train_aligned = X_train
            y_train_aligned = y_train

        # Prepare meta-features
        if self.use_features:
            # Config B: Concatenate predictions + original features
            meta_features = pd.concat([oof_predictions, X_train_aligned], axis=1)
            print(f"Config B: Using predictions + features")
        else:
            # Config A: Use only predictions
            meta_features = oof_predictions
            print(f"Config A: Using predictions only")
        
        print(f"Meta-features shape: {meta_features.shape}")
        
        # Train meta-learner
        start_time = datetime.now()
        self.meta_learner.fit(meta_features, y_train_aligned)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Log training
        train_preds = self.meta_learner.predict(meta_features)
        train_mae = np.abs(train_preds - y_train_aligned).mean()
        train_rmse = np.sqrt(((train_preds - y_train_aligned) ** 2).mean())
        
        log_entry = {
            'config': self.config_name,
            'timestamp': datetime.now().isoformat(),
            'meta_features_shape': list(meta_features.shape),
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_time_seconds': train_time,
            'meta_model': str(self.meta_learner.__class__.__name__)
        }
        
        self.training_log.append(log_entry)
        
        print(f"✓ Training complete:")
        print(f"  Train MAE:  {train_mae:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Time: {train_time:.2f}s")
        
        # Save checkpoint
        if save_dir:
            self.save_checkpoint(save_dir, save_models=save_models)
        
        return self
    
    def train_base_models_final(self, X_train, y_train, base_models_dict):
        """
        Train base models on full training data for final predictions.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        base_models_dict : dict
            Dictionary with model names and untrained model objects
        """
        print(f"\nTraining base models on full training set...")
        
        for model_name, model_template in base_models_dict.items():
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train, y_train)
            self.base_models[model_name] = model
            print(f"  ✓ {model_name} trained")
        
        print(f"✓ All {len(self.base_models)} base models trained")
    
    def predict(self, X_test, base_models=None):
        """
        Generate predictions on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        base_models : dict, optional
            Trained base models (if not provided, uses self.base_models)
            
        Returns:
        --------
        np.array
            Meta-learner predictions
        dict
            Base model predictions
        """
        if base_models is None:
            base_models = self.base_models
        
        if not base_models:
            raise ValueError("No trained base models available. Call train_base_models_final first.")
        
        # Generate base predictions
        base_preds_df = pd.DataFrame(index=X_test.index)
        
        for model_name, model in base_models.items():
            base_preds_df[model_name] = model.predict(X_test)
        
        # Prepare meta-features
        if self.use_features:
            # Config B: Concatenate predictions + original features
            meta_features = pd.concat([base_preds_df, X_test], axis=1)
        else:
            # Config A: Use only predictions
            meta_features = base_preds_df
        
        # Meta-learner predictions
        ensemble_preds = self.meta_learner.predict(meta_features)
        
        return ensemble_preds, base_preds_df.to_dict('list')
    
    def save_checkpoint(self, save_dir, save_models=True):
        """Save model checkpoint and training log."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save training log
        log_path = os.path.join(save_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"\n✓ Checkpoint saved to {save_dir}/")
        print(f"  - training_log.json")

        if save_models:
            meta_path = os.path.join(save_dir, 'meta_learner.pkl')
            joblib.dump(self.meta_learner, meta_path)
            print(f"  - meta_learner.pkl")

            if self.base_models:
                base_path = os.path.join(save_dir, 'base_models.pkl')
                joblib.dump(self.base_models, base_path)
                print(f"  - base_models.pkl")

    def load_models(self, save_dir):
        """Load meta-learner and base models from a checkpoint directory."""
        meta_path = os.path.join(save_dir, 'meta_learner.pkl')
        base_path = os.path.join(save_dir, 'base_models.pkl')

        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
        else:
            raise FileNotFoundError(f"Missing meta_learner.pkl at {meta_path}")

        if os.path.exists(base_path):
            self.base_models = joblib.load(base_path)
        else:
            raise FileNotFoundError(f"Missing base_models.pkl at {base_path}")

        print(f"✓ Loaded models from {save_dir}")
        return self
    
    def load_checkpoint(self, save_dir):
        """Load training log from checkpoint."""
        log_path = os.path.join(save_dir, 'training_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.training_log = json.load(f)
            print(f"✓ Loaded training log from {log_path}")


def compare_configs(config_a_results, config_b_results, baseline_results):
    """
    Compare Config A, Config B, and baselines.
    
    Parameters:
    -----------
    config_a_results : dict
        Results from Config A
    config_b_results : dict
        Results from Config B
    baseline_results : dict
        Results from CKPT2 baselines
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    all_results = []
    
    # Baselines
    for name, metrics in baseline_results.items():
        all_results.append({
            'Model': name,
            'Type': 'Baseline',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })
    
    # Config A
    all_results.append({
        'Model': 'Stacking (Config A)',
        'Type': 'Ensemble',
        'MAE': config_a_results['MAE'],
        'RMSE': config_a_results['RMSE']
    })
    
    # Config B
    all_results.append({
        'Model': 'Stacking (Config B)',
        'Type': 'Ensemble',
        'MAE': config_b_results['MAE'],
        'RMSE': config_b_results['RMSE']
    })
    
    df = pd.DataFrame(all_results)
    df = df.sort_values('MAE')
    
    return df


if __name__ == "__main__":
    print("Stacked Ensemble Module")
    print("Use: from src.stacking import StackedEnsemble, compare_configs")
