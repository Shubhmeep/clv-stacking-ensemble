"""
Lightweight analysis helpers for feature importance and error segmentation.
"""

import pandas as pd
import numpy as np


def feature_importance_table(model, feature_names, top_n=10):
    """
    Return a DataFrame of top feature importances for tree or linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef)
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def segment_error_table(y_true, y_pred, X, feature, q=0.75):
    """
    Compute MAE for high vs low segments based on a feature quantile.
    """
    thresh = X[feature].quantile(q)
    high = X[feature] >= thresh
    low = ~high

    def mae(a, b):
        return float(np.abs(a - b).mean())

    data = [
        {"segment": f"{feature} >= {q:.2f}q", "size": int(high.sum()), "MAE": mae(y_true[high], y_pred[high])},
        {"segment": f"{feature} < {q:.2f}q", "size": int(low.sum()), "MAE": mae(y_true[low], y_pred[low])},
    ]

    return pd.DataFrame(data)

