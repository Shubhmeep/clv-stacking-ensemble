"""
Two-stage (hurdle) model for zero-inflated count targets.

Stage 1: Classifier predicts P(nonzero)
Stage 2: Regressor predicts count given nonzero
Final prediction = P(nonzero) * E(count | nonzero)
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TwoStageModel:
    """
    Two-stage model with RandomForest classifier + regressor by default.
    """

    def __init__(
        self,
        classifier=None,
        regressor=None,
        random_state=42,
        n_estimators=300,
        max_depth=6,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.classifier = classifier or RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.regressor = regressor or RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        y_bin = (y > 0).astype(int)
        self.classifier.fit(X, y_bin)

        nz_mask = y > 0
        X_nz = X[nz_mask]
        y_nz = y[nz_mask]

        self.regressor.fit(X_nz, y_nz)
        return self

    def predict(self, X):
        p_nonzero = self.classifier.predict_proba(X)[:, 1]
        reg_pred = self.regressor.predict(X)
        return p_nonzero * reg_pred

