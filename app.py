import streamlit as st
import pandas as pd
import numpy as np
from src.data_validation import validate_dataset, prepare_dataset
from src.model import ARDModel
import os

# Set page config
st.set_page_config(
    page_title="ADR Regression App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

def main():
    st.title("ADR Regression Analysis Tool")
    st.write("Upload your dataset and train an Automatic Relevance Determination regression model.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and display the dataset
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            
            # Dataset validation
            is_valid, validation_results = validate_dataset(df)
            
            # Display validation results
            if validation_results["errors"]:
                st.error("Dataset Validation Errors:")
                for error in validation_results["errors"]:
                    st.write(f"- {error}")
            
            if validation_results["warnings"]:
                st.warning("Dataset Validation Warnings:")
                for warning in validation_results["warnings"]:
                    st.write(f"- {warning}")
            
            if is_valid:
                # Target column selection
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                target_column = st.selectbox(
                    "Select target variable",
                    options=numeric_columns
                )
                
                # Model training
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        # Prepare dataset
                        X, y = prepare_dataset(df, target_column)
                        
                        # Initialize and train model
                        model = ARDModel()
                        results = model.train(X, y)
                        
                        # Store model in session state
                        st.session_state.model = model
                        
                        # Display results
                        st.success("Model trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("### Training Metrics")
                            st.write(f"Train RMSE: {results['train_rmse']:.4f}")
                            st.write(f"Test RMSE: {results['test_rmse']:.4f}")
                            st.write(f"Train R²: {results['train_r2']:.4f}")
                            st.write(f"Test R²: {results['test_r2']:.4f}")
                        
                        with col2:
                            st.write("### Feature Relevance")
                            relevance_df = pd.DataFrame(
                                results['relevant_features'].items(),
                                columns=['Feature', 'Relevance Score']
                            )
                            st.dataframe(relevance_df)
                        
                        # Save model
                        model.save_model('model.pkl')
                        st.info("Model saved as 'model.pkl'")
                
                # Prediction section
                if st.session_state.model is not None:
                    st.write("### Make Predictions")
                    st.write("Upload a CSV file with the same features for prediction")
                    
                    pred_file = st.file_uploader("Choose a CSV file for prediction", type="csv", key="pred_file")
                    
                    if pred_file is not None:
                        pred_df = pd.read_csv(pred_file)
                        st.write("Preview of prediction data:")
                        st.dataframe(pred_df.head())
                        
                        if st.button("Generate Predictions"):
                            try:
                                predictions = st.session_state.model.predict(pred_df)
                                pred_df['Predictions'] = predictions
                                
                                st.write("### Predictions")
                                st.dataframe(pred_df)
                                
                                # Download predictions
                                csv = pred_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Predictions",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

if __name__ == "__main__":
    main()