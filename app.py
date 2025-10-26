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
        
        if not file:
            return jsonify({'status': 'error', 'message': 'No training file provided'})
        if not target_column:
            return jsonify({'status': 'error', 'message': 'No target column provided'})

        # Read the dataset
        df = pd.read_csv(file)

        if target_column not in df.columns:
            return jsonify({'status': 'error', 'message': f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"})

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
            
            # Make predictions (validate columns)
            try:
                predictions = model.predict(pred_df)
            except KeyError as ke:
                # Model expected columns missing in prediction df
                return jsonify({'status': 'error', 'message': str(ke)})
            
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


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded CSV and return data for frontend plotting.

    This function is defensive: it tries to coerce columns to numeric when possible,
    guards subsetting operations, and returns helpful error messages instead of raw
    pandas errors like "['col'] not in index".
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No file provided'})

        df = pd.read_csv(file)
        df = df.dropna(axis=1, how='all')

        nrows = max(1, len(df))

        # Detect datetime-like columns first — only check object/categorical columns to avoid
        # misclassifying numeric columns (years/ints) as datetimes.
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype).startswith('category'):
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    non_null = parsed.notna().sum()
                    if non_null / nrows > 0.5:
                        datetime_cols.append(col)
                except Exception:
                    pass

        # Detect numeric columns by coercion (more robust than dtype alone)
        numeric_cols = []
        categorical_cols = []
        for col in df.columns:
            if col in datetime_cols:
                continue
            coerced = pd.to_numeric(df[col], errors='coerce')
            non_null = coerced.notna().sum()
            if non_null / nrows > 0.5:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # Histograms and boxplots
        histograms = {}
        boxplots = {}
        for col in numeric_cols:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
                if col_data.empty or col_data.nunique() <= 1:
                    continue
                counts, bin_edges = np.histogram(col_data, bins='auto')
                histograms[col] = {'bins': bin_edges.tolist(), 'counts': counts.tolist()}
                q1 = float(np.percentile(col_data, 25))
                median = float(np.percentile(col_data, 50))
                q3 = float(np.percentile(col_data, 75))
                mn = float(col_data.min())
                mx = float(col_data.max())
                boxplots[col] = {'min': mn, 'q1': q1, 'median': median, 'q3': q3, 'max': mx}
            except Exception:
                # Skip problematic numeric column
                continue

        # Correlation and top pairs
        corr_matrix = None
        top_pairs = []
        pair_samples = []
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr().fillna(0)
                corr_matrix = corr.values.tolist()
                pairs = []
                for i, a in enumerate(numeric_cols):
                    for j, b in enumerate(numeric_cols):
                        if j <= i:
                            continue
                        val = corr.iloc[i, j]
                        pairs.append((abs(val), a, b, float(val)))
                pairs.sort(reverse=True, key=lambda x: x[0])
                for absval, a, b, raw in pairs[:6]:
                    if absval < 0.1:
                        continue
                    top_pairs.append({'pair': [a, b], 'corr': raw})
                    try:
                        sample = df.loc[:, [a, b]].dropna()
                        if len(sample) > 1000:
                            sample = sample.sample(1000, random_state=42)
                        pair_samples.append({'x_name': a, 'y_name': b, 'x': pd.to_numeric(sample[a], errors='coerce').dropna().astype(float).tolist(), 'y': pd.to_numeric(sample[b], errors='coerce').dropna().astype(float).tolist()})
                    except KeyError:
                        # If columns are unexpectedly missing, skip this pair
                        continue
            except Exception:
                corr_matrix = None

        # Categorical counts
        categorical_counts = {}
        for col in categorical_cols:
            try:
                vc = df[col].value_counts(dropna=True).head(20)
                categorical_counts[col] = {'labels': vc.index.astype(str).tolist(), 'counts': vc.tolist()}
            except Exception:
                continue

        # Time series aggregation
        time_series = None
        if datetime_cols and len(numeric_cols) > 0:
            dcol = datetime_cols[0]
            try:
                parsed = pd.to_datetime(df[dcol], errors='coerce', infer_datetime_format=True)
                ts_df = df.copy()
                ts_df[dcol] = parsed
                ts_df = ts_df.dropna(subset=[dcol])
                if not ts_df.empty:
                    ts_df.set_index(dcol, inplace=True)
                    agg = ts_df[numeric_cols].apply(pd.to_numeric, errors='coerce').resample('D').mean().fillna(method='ffill').fillna(0)
                    agg_sample = agg.tail(500)
                    time_series = {'datetime_col': dcol, 'index': [t.isoformat() for t in agg_sample.index], 'values': {col: agg_sample[col].astype(float).tolist() for col in agg_sample.columns}}
            except Exception:
                time_series = None

        return jsonify({
            'status': 'success',
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'histograms': histograms,
            'boxplots': boxplots,
            'correlation': {'columns': numeric_cols, 'matrix': corr_matrix},
            'top_pairs': top_pairs,
            'pair_samples': pair_samples,
            'categorical_counts': categorical_counts,
            'time_series': time_series
        })

    except KeyError as ke:
        # Return a clearer JSON error when requested columns are missing
        return jsonify({'status': 'error', 'message': f'Missing columns: {ke}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)