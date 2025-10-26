# Automatic Relevance Determination (ARD) Regression Project

## Overview
This project implements an interactive web application for Automatic Relevance Determination (ARD) Regression, a Bayesian approach to linear regression that automatically determines the relevance of different input features.

## Theory Behind ARD Regression

### What is ARD Regression?
ARD Regression is a Bayesian linear regression model that performs automatic feature selection by learning the relevance of each input feature. It's particularly useful when:
- You have many potential input features
- You want to automatically identify which features are most relevant
- You need a model that can handle uncertainty in feature selection

### Key Concepts

1. **Bayesian Framework**
   - ARD uses a Bayesian approach to determine feature relevance
   - Each feature gets its own precision parameter (inverse of variance)
   - These parameters are automatically tuned during model training

2. **Automatic Feature Selection**
   - Features with low relevance get effectively "turned off" (weights → 0)
   - Relevant features maintain significant weights
   - This built-in feature selection helps prevent overfitting

3. **Mathematical Foundation**
   ```
   y = Xw + ε
   w ~ N(0, diag(α₁⁻¹, ..., αₚ⁻¹))
   ε ~ N(0, σ²I)
   ```
   where:
   - y is the target variable
   - X is the feature matrix
   - w are the weights
   - α are the feature precision parameters
   - σ² is the noise variance

## Project Implementation

### Architecture
1. **Backend (Flask)**
   - Model training and prediction API endpoints
   - Data validation and processing
   - Statistical analysis and visualization data preparation

2. **Frontend**
   - Interactive web interface
   - Real-time data visualization
   - User-friendly model interaction

### Key Features

1. **Data Management**
   - CSV file upload support
   - Automatic data validation
   - Feature type detection (numeric/categorical)

2. **Model Training**
   - Interactive feature selection
   - Real-time model training
   - Performance metrics visualization

3. **Prediction Interface**
   - Batch prediction support
   - Results visualization
   - Prediction confidence intervals

4. **Data Analysis Tools**
   - Distribution plots (histograms, box plots)
   - Correlation analysis
   - Feature importance visualization
   - Time series analysis (if applicable)

## Usage Guide

### Setup
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Workflow

1. **Data Upload and Validation**
   - Upload your CSV dataset
   - System validates data structure
   - Displays basic statistics

2. **Model Training**
   - Select target variable
   - Configure model parameters (optional)
   - Train the model
   - View feature relevance scores

3. **Making Predictions**
   - Upload new data
   - Get predictions
   - Analyze results

4. **Data Analysis**
   - Explore feature distributions
   - Analyze correlations
   - Visualize relationships

## Technical Details

### ARD Model Implementation
```python
from sklearn.linear_model import ARDRegression

class ARDRegressor:
    def __init__(self):
        self.model = ARDRegression(compute_score=True)
        self.feature_names = None

    def fit(self, X, y):
        """Train model and store feature names"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self
```

### Key Parameters
- **threshold_lambda**: Threshold for determining relevance
- **alpha**: Initial value of noise precision
- **lambda**: Initial values of feature precision

### Advantages of ARD
1. **Automatic Feature Selection**
   - No need for manual feature selection
   - Reduces model complexity automatically
   - Helps prevent overfitting

2. **Interpretability**
   - Clear feature relevance scores
   - Uncertainty quantification
   - Transparent decision-making process

3. **Robust Performance**
   - Handles multicollinearity well
   - Works with high-dimensional data
   - Provides probabilistic predictions

## Example Use Case: Real Estate Price Prediction

### Dataset Features
- Square footage
- Number of bedrooms/bathrooms
- Lot size
- Year built
- Distance to city center
- Neighborhood rating
- Average income in area
- Crime rate

### Model Performance
The ARD model automatically:
1. Identifies most relevant features for price prediction
2. Reduces impact of irrelevant features
3. Provides uncertainty estimates for predictions

### Visualization Examples
1. **Feature Importance Plot**
   - Shows relative importance of each feature
   - Indicates confidence intervals
   - Helps in understanding model decisions

2. **Prediction vs Actual Plot**
   - Demonstrates model accuracy
   - Shows prediction uncertainty
   - Identifies potential outliers

## Resources and References

### Documentation
- [scikit-learn ARD Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
- [Theory Behind ARD](https://www.microsoft.com/en-us/research/publication/sparse-bayesian-learning-and-the-relevance-vector-machine/)

### Related Concepts
- Bayesian Linear Regression
- Relevance Vector Machines
- Sparse Bayesian Learning

## Future Improvements
1. Add cross-validation support
2. Implement feature interaction analysis
3. Add more visualization options
4. Support for categorical feature encoding
5. Model comparison tools