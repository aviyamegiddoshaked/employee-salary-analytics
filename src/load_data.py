"""
Module for loading employee salary data.
"""

import pandas as pd
import os
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_raw_data(file_path):
    """
    Load raw employee salary data from a CSV file.
    
    Parameters:
    -----------
    file_path : str or Path
        Full path to the CSV file to load
        
    Returns:
    --------
    pd.DataFrame
        Raw employee salary dataset
    """
    # Convert to Path object if string
    data_path = Path(file_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load CSV into pandas DataFrame
    df = pd.read_csv(data_path)
    
    # Print shape and basic info
    print(f"✓ Loaded data from: {data_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Data types:\n{df.dtypes}")
    
    return df


def load_processed_data(filename='cleaned_data.csv'):
    """
    Load processed/cleaned data from the data/processed directory.
    
    Parameters:
    -----------
    filename : str, default='cleaned_data.csv'
        Name of the processed CSV file to load
        
    Returns:
    --------
    pd.DataFrame
        Processed employee salary dataset
    """
    project_root = get_project_root()
    data_path = project_root / 'data' / 'processed' / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")
    return df


def save_data(df, path):
    """
    Save a DataFrame to a CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    path : str or Path
        Full path where the CSV file should be saved
    """
    # Convert to Path object if string
    output_path = Path(path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df)} rows and {len(df.columns)} columns to {output_path}")


def save_processed_data(df, filename='cleaned_data.csv'):
    """
    Save processed data to the data/processed directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filename : str, default='cleaned_data.csv'
        Name of the file to save
    """
    project_root = get_project_root()
    output_path = project_root / 'data' / 'processed' / filename
    
    # Create processed directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df)} rows to {output_path}")

