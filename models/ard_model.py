from sklearn.linear_model import ARDRegression
import numpy as np

class ARDRegressor:
    def __init__(self):
        self.model = ARDRegression(compute_score=True)
    
    def fit(self, X, y):
        """Train the ARD model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def get_relevance(self):
        """Get the relevance scores for features"""
        return self.model.coef_
    
    def get_score(self):
        """Get the model score"""
        return self.model.score_