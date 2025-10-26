from sklearn.linear_model import ARDRegression
import numpy as np
import pandas as pd


class ARDRegressor:
    """Wrapper around sklearn's ARDRegression that remembers feature names.

    Behavior:
      - fit(X, y): accepts DataFrame or ndarray. If DataFrame, stores feature names.
      - predict(X): accepts DataFrame or ndarray. If DataFrame, selects/reorders columns to match training features.
    """
    def __init__(self):
        self.model = ARDRegression(compute_score=True)
        self.feature_names = None

    def fit(self, X, y):
        """Train the ARD model and remember feature names when provided as DataFrame."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_train = X.values
        else:
            X_train = X
        self.model.fit(X_train, y)
        return self

    def predict(self, X):
        """Make predictions using the trained model.

        If X is a DataFrame and feature names were recorded during fit, the DataFrame will be re-ordered to match.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise KeyError(f"Missing feature columns for prediction: {missing}")
                X_pred = X[self.feature_names].values
            else:
                X_pred = X.values
        else:
            X_pred = X
        return self.model.predict(X_pred)

    def get_relevance(self):
        """Get the relevance scores for features (coef_)."""
        return getattr(self.model, 'coef_', None)

    def get_score(self, X=None, y=None):
        """Return model score. If X and y provided, compute score on that data."""
        if X is not None and y is not None:
            return self.model.score(X, y)
        return getattr(self.model, 'score_', None)