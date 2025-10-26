from flask import Flask, render_template, request, jsonify
from models.ard_model import ARDRegressor
import numpy as np
import pandas as pd

app = Flask(__name__)
model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/validate', methods=['POST'])
def validate_dataset():
    try:
        file = request.files['file']
        if file:
            # Read the dataset
            df = pd.read_csv(file)
            
            # Basic validation
            response = {
                'status': 'success',
                'message': 'Dataset is valid',
                'info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'features': list(df.columns)
                }
            }
            return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if file and target_column:
            # Read the dataset
            df = pd.read_csv(file)
            
            # Split features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Initialize and train the model
            global model
            model = ARDRegressor()
            model.fit(X, y)
            
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'features': list(X.columns)
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not trained yet'})
        
        file = request.files['file']
        if file:
            # Read the prediction dataset
            pred_df = pd.read_csv(file)
            
            # Make predictions
            predictions = model.predict(pred_df)
            
            # Create results with input data and predictions
            results = pred_df.copy()
            results['prediction'] = predictions
            
            # Convert to dictionary for JSON response
            predictions_dict = {
                'status': 'success',
                'predictions': results.to_dict(orient='records'),
                'summary': {
                    'count': len(predictions),
                    'mean': float(np.mean(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions))
                }
            }
            
            return jsonify(predictions_dict)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)