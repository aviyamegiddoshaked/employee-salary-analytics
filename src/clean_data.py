"""
Module for cleaning employee salary data.
"""

import pandas as pd
import numpy as np


def get_data_info(df):
    """
    Display comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'dtypes': df.dtypes,
        'numeric_summary': df.describe() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None,
        'categorical_summary': df.describe(include=['object']) if len(df.select_dtypes(include=['object']).columns) > 0 else None
    }
    return info


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        print(f"✓ Removed {removed} duplicate row(s)")
    else:
        print("✓ No duplicates found")
    
    return df_clean


def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, default='drop'
        Strategy to handle missing values: 'drop', 'fill', or 'forward_fill'
    fill_value : any, default=None
        Value to fill missing values with (if strategy='fill')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        print("✓ No missing values found")
        return df
    
    if strategy == 'drop':
        df_clean = df.dropna()
        print(f"✓ Dropped rows with missing values ({missing_before} missing values removed)")
    elif strategy == 'fill':
        if fill_value is None:
            # Fill numeric columns with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        else:
            df.fillna(fill_value, inplace=True)
        df_clean = df
        print(f"✓ Filled {missing_before} missing values")
    elif strategy == 'forward_fill':
        df_clean = df.fillna(method='ffill')
        print(f"✓ Forward filled {missing_before} missing values")
    else:
        raise ValueError("Strategy must be 'drop', 'fill', or 'forward_fill'")
    
    return df_clean


def clean_column_names(df):
    """
    Clean column names by converting to lowercase and replacing spaces with underscores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    print("✓ Cleaned column names")
    return df_clean


def validate_data_types(df):
    """
    Validate and convert data types to appropriate formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with validated data types
    """
    df_clean = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['age', 'experience_years', 'salary_usd', 'bonus_usd', 
                   'work_hours_per_week', 'performance_score', 'joining_year']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert categorical columns
    categorical_cols = ['gender', 'country', 'city', 'education', 'job_title', 
                       'department', 'remote_work', 'contract_type']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    print("✓ Validated and converted data types")
    return df_clean


def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, default=None
        List of numeric columns to check for outliers. If None, checks all numeric columns.
    factor : float, default=1.5
        IQR factor for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_rows = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            if outliers > 0:
                print(f"✓ Removed {outliers} outliers from {col}")
    
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"✓ Total: Removed {removed} rows with outliers")
    else:
        print("✓ No outliers detected")
    
    return df_clean


def clean_dataset(df):
    """
    Clean the employee salary dataset with sensible defaults.
    
    This function performs comprehensive data cleaning:
    - Drops rows where salary is missing
    - Fills numeric NaNs with median
    - Fills categorical NaNs with mode
    - Removes duplicate rows
    - Filters out salary <= 0
    - Converts categorical columns to lowercase
    - Converts experience, age, work_hours to numeric
    - Prints cleaning summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("=" * 60)
    print("DATASET CLEANING SUMMARY")
    print("=" * 60)
    
    initial_shape = df.shape
    df_clean = df.copy()
    
    # Step 1: Clean column names (convert to lowercase with underscores)
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    print(f"\nInitial dataset shape: {initial_shape[0]} rows × {initial_shape[1]} columns")
    
    # Step 2: Drop rows where salary is missing
    salary_col = None
    for col in ['salary_usd', 'salary']:
        if col in df_clean.columns:
            salary_col = col
            break
    
    if salary_col:
        rows_before = len(df_clean)
        df_clean = df_clean.dropna(subset=[salary_col])
        rows_dropped = rows_before - len(df_clean)
        if rows_dropped > 0:
            print(f"✓ Dropped {rows_dropped} rows with missing salary")
        else:
            print("✓ No rows with missing salary found")
    else:
        print("⚠ Warning: Salary column not found, skipping salary-based filtering")
    
    # Step 3: Filter out salary <= 0
    if salary_col:
        rows_before = len(df_clean)
        df_clean = df_clean[df_clean[salary_col] > 0]
        rows_filtered = rows_before - len(df_clean)
        if rows_filtered > 0:
            print(f"✓ Filtered out {rows_filtered} rows with salary <= 0")
        else:
            print("✓ No rows with salary <= 0 found")
    
    # Step 4: Convert experience, age, work_hours to numeric
    numeric_cols_to_convert = {
        'experience_years': ['experience_years', 'experience'],
        'age': ['age'],
        'work_hours_per_week': ['work_hours_per_week', 'work_hours', 'hours_per_week']
    }
    
    for target_col, possible_names in numeric_cols_to_convert.items():
        for col_name in possible_names:
            if col_name in df_clean.columns:
                df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')
                if col_name != target_col and target_col not in df_clean.columns:
                    df_clean = df_clean.rename(columns={col_name: target_col})
                print(f"✓ Converted {col_name} to numeric")
                break
    
    # Step 5: Remove duplicate rows
    rows_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = rows_before - len(df_clean)
    if duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed} duplicate row(s)")
    else:
        print("✓ No duplicate rows found")
    
    # Step 6: Handle missing values
    # Fill numeric NaNs with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            if pd.notna(median_val):
                missing_count = df_clean[col].isnull().sum()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"✓ Filled {missing_count} missing values in {col} with median ({median_val:.2f})")
    
    # Fill categorical NaNs with mode
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                missing_count = df_clean[col].isnull().sum()
                df_clean[col].fillna(mode_val[0], inplace=True)
                print(f"✓ Filled {missing_count} missing values in {col} with mode ('{mode_val[0]}')")
            else:
                # If no mode, fill with 'Unknown'
                missing_count = df_clean[col].isnull().sum()
                df_clean[col].fillna('Unknown', inplace=True)
                print(f"✓ Filled {missing_count} missing values in {col} with 'Unknown'")
    
    # Step 7: Convert categorical columns to lowercase
    # Get categorical columns again after all transformations
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.lower()
            print(f"✓ Converted {col} values to lowercase")
        elif df_clean[col].dtype.name == 'category':
            # Convert category to string, lowercase, then back to category
            df_clean[col] = df_clean[col].astype(str).str.lower().astype('category')
            print(f"✓ Converted {col} values to lowercase")
    
    # Final summary
    final_shape = df_clean.shape
    rows_removed = initial_shape[0] - final_shape[0]
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final dataset shape: {final_shape[0]} rows × {final_shape[1]} columns")
    print(f"Total rows removed: {rows_removed}")
    print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
    print("=" * 60 + "\n")
    
    return df_clean


def clean_data_pipeline(df, remove_dups=True, handle_missing='drop', 
                       clean_names=True, validate_types=True, 
                       remove_outliers=False, outlier_columns=None):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    remove_dups : bool, default=True
        Whether to remove duplicates
    handle_missing : str, default='drop'
        Strategy for handling missing values
    clean_names : bool, default=True
        Whether to clean column names
    validate_types : bool, default=True
        Whether to validate data types
    remove_outliers : bool, default=False
        Whether to remove outliers
    outlier_columns : list, default=None
        Columns to check for outliers
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("=" * 50)
    print("Starting Data Cleaning Pipeline")
    print("=" * 50)
    
    df_clean = df.copy()
    
    if clean_names:
        df_clean = clean_column_names(df_clean)
    
    if validate_types:
        df_clean = validate_data_types(df_clean)
    
    if remove_dups:
        df_clean = remove_duplicates(df_clean)
    
    if handle_missing:
        df_clean = handle_missing_values(df_clean, strategy=handle_missing)
    
    if remove_outliers:
        df_clean = remove_outliers_iqr(df_clean, columns=outlier_columns)
    
    print("=" * 50)
    print(f"Cleaning complete! Final shape: {df_clean.shape}")
    print("=" * 50)
    
    return df_clean

