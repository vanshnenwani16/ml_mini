import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the input dataset for ADR regression.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_results)
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check if dataset is empty
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Dataset is empty")
        return False, validation_results
    
    # Check for minimum number of rows
    if len(df) < 10:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Dataset must have at least 10 rows")
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Missing values found in columns: {', '.join(missing_cols)}")
    
    # Check for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        validation_results["warnings"].append(
            f"Non-numeric columns found: {', '.join(non_numeric_cols)}. These will be excluded from analysis."
        )
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        validation_results["warnings"].append(
            f"Constant columns found: {', '.join(constant_cols)}. These will be excluded from analysis."
        )
    
    return validation_results["is_valid"], validation_results

def prepare_dataset(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the dataset for ADR regression by handling non-numeric columns
    and splitting into features and target.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) - features and target
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column from features if it's in numeric_cols
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Select features and target
    X = df[numeric_cols]
    y = df[target_column]
    
    return X, y