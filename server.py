from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from src.data_validation import validate_dataset, prepare_dataset
from src.model import ARDModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
MODEL = None

def create_feature_importance_plot(model, feature_names):
    """Create a feature importance visualization"""
    relevance_scores = model._get_relevant_features()
    fig = px.bar(
        x=list(relevance_scores.keys()),
        y=list(relevance_scores.values()),
        title='Feature Importance in ADR Model',
        labels={'x': 'Features', 'y': 'Relevance Score'}
    )
    return json.loads(fig.to_json())

def create_prediction_vs_actual_plot(y_true, y_pred):
    """Create a scatter plot of predicted vs actual values"""
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        title='Predicted vs Actual Values',
        labels={'x': 'Actual Values', 'y': 'Predicted Values'}
    )
    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[min(y_true), max(y_true)],
            y=[min(y_true), max(y_true)],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        )
    )
    return json.loads(fig.to_json())

@app.route('/api/validate', methods=['POST'])
def validate_data():
    """Validate uploaded dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Validate the dataset
        is_valid, validation_results = validate_dataset(df)
        
        # Add column information
        validation_results['columns'] = {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'non_numeric': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        return jsonify({
            'is_valid': is_valid,
            'validation_results': validation_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the ADR model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'No target column specified'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Prepare dataset
        X, y = prepare_dataset(df, target_column)
        
        # Initialize and train model
        global MODEL
        MODEL = ARDModel()
        results = MODEL.train(X, y)
        
        # Create visualizations
        feature_importance_plot = create_feature_importance_plot(MODEL, X.columns)
        prediction_plot = create_prediction_vs_actual_plot(y, MODEL.predict(X))
        
        return jsonify({
            'training_results': results,
            'visualizations': {
                'feature_importance': feature_importance_plot,
                'prediction_plot': prediction_plot
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Make predictions
        predictions = MODEL.predict(df)
        
        # Create prediction response
        prediction_data = {
            'predictions': predictions.tolist(),
            'feature_names': MODEL.feature_names
        }
        
        return jsonify(prediction_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)