"""
Module for feature engineering on employee salary data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def create_total_compensation(df):
    """
    Create total compensation feature (salary + bonus).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with total_compensation column added
    """
    df = df.copy()
    df['total_compensation'] = df['salary_usd'] + df['bonus_usd']
    print("✓ Created total_compensation feature")
    return df


def create_salary_per_hour(df):
    """
    Create salary per hour feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with salary_per_hour column added
    """
    df = df.copy()
    df['salary_per_hour'] = df['salary_usd'] / (df['work_hours_per_week'] * 52)
    print("✓ Created salary_per_hour feature")
    return df


def create_experience_level(df):
    """
    Create experience level categories based on experience years.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with experience_level column added
    """
    df = df.copy()
    
    def categorize_experience(years):
        if years < 2:
            return 'Entry Level'
        elif years < 5:
            return 'Junior'
        elif years < 10:
            return 'Mid Level'
        elif years < 15:
            return 'Senior'
        else:
            return 'Expert'
    
    df['experience_level'] = df['experience_years'].apply(categorize_experience)
    df['experience_level'] = df['experience_level'].astype('category')
    print("✓ Created experience_level feature")
    return df


def create_age_group(df):
    """
    Create age group categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with age_group column added
    """
    df = df.copy()
    
    def categorize_age(age):
        if age < 25:
            return 'Young (18-24)'
        elif age < 35:
            return 'Early Career (25-34)'
        elif age < 45:
            return 'Mid Career (35-44)'
        elif age < 55:
            return 'Senior (45-54)'
        else:
            return 'Veteran (55+)'
    
    df['age_group'] = df['age'].apply(categorize_age)
    df['age_group'] = df['age_group'].astype('category')
    print("✓ Created age_group feature")
    return df


def create_years_at_company(df):
    """
    Create years at company feature based on joining year.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with years_at_company column added
    """
    df = df.copy()
    current_year = datetime.now().year
    df['years_at_company'] = current_year - df['joining_year']
    print("✓ Created years_at_company feature")
    return df


def create_education_rank(df):
    """
    Create education rank feature (numeric encoding of education level).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with education_rank column added
    """
    df = df.copy()
    
    education_mapping = {
        'High School': 1,
        'Bachelor': 2,
        'Master': 3,
        'PhD': 4
    }
    
    df['education_rank'] = df['education'].map(education_mapping)
    print("✓ Created education_rank feature")
    return df


def create_performance_category(df):
    """
    Create performance category based on performance score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with performance_category column added
    """
    df = df.copy()
    
    def categorize_performance(score):
        if score <= 3:
            return 'Low'
        elif score <= 6:
            return 'Medium'
        else:
            return 'High'
    
    df['performance_category'] = df['performance_score'].apply(categorize_performance)
    df['performance_category'] = df['performance_category'].astype('category')
    print("✓ Created performance_category feature")
    return df


def create_bonus_percentage(df):
    """
    Create bonus percentage of salary feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with bonus_percentage column added
    """
    df = df.copy()
    df['bonus_percentage'] = (df['bonus_usd'] / df['salary_usd']) * 100
    print("✓ Created bonus_percentage feature")
    return df


def encode_categorical_features(df, columns=None, method='onehot'):
    """
    Encode categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, default=None
        List of columns to encode. If None, encodes all categorical columns.
    method : str, default='onehot'
        Encoding method: 'onehot' or 'label'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded features
    """
    df = df.copy()
    
    if columns is None:
        # Select categorical columns (excluding target and ID columns)
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        exclude_cols = ['employee_id', 'name', 'salary_usd']  # Exclude ID and target
        columns = [col for col in categorical_cols if col not in exclude_cols]
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=True)
        print(f"✓ One-hot encoded {len(columns)} categorical columns")
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))
        df_encoded = df
        print(f"✓ Label encoded {len(columns)} categorical columns")
    else:
        raise ValueError("Method must be 'onehot' or 'label'")
    
    return df_encoded


def add_salary_per_hour(df):
    """
    Compute salary per hour from salary and work hours.
    
    ⚠️ WARNING: This feature should NOT be used when predicting salary because it leaks 
    target information (salary_per_hour = salary / work_hours). This function is kept 
    only for exploratory data analysis purposes.
    
    For modeling, use only: add_seniority_level() and encode_categoricals()
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with salary_per_hour column added
    """
    df = df.copy()
    
    # Find salary and work_hours columns (handle different naming conventions)
    salary_col = None
    work_hours_col = None
    
    for col in ['salary_usd', 'salary']:
        if col in df.columns:
            salary_col = col
            break
    
    for col in ['work_hours_per_week', 'work_hours', 'hours_per_week']:
        if col in df.columns:
            work_hours_col = col
            break
    
    if salary_col and work_hours_col:
        # Compute salary per hour: salary / work_hours
        df['salary_per_hour'] = df[salary_col] / df[work_hours_col]
        print(f"✓ Created salary_per_hour feature (salary / work_hours)")
    else:
        missing_cols = []
        if not salary_col:
            missing_cols.append('salary')
        if not work_hours_col:
            missing_cols.append('work_hours')
        raise ValueError(f"Required columns not found: {missing_cols}")
    
    return df


def add_seniority_level(df):
    """
    Create seniority level based on years of experience.
    
    Categories:
    - < 3 years → "junior"
    - 3-7 years → "mid"
    - > 7 years → "senior"
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with seniority_level column added
    """
    df = df.copy()
    
    # Find experience column (handle different naming conventions)
    exp_col = None
    for col in ['experience_years', 'years_of_experience', 'experience']:
        if col in df.columns:
            exp_col = col
            break
    
    if exp_col:
        def categorize_seniority(years):
            if pd.isna(years):
                return 'unknown'
            elif years < 3:
                return 'junior'
            elif years <= 7:
                return 'mid'
            else:
                return 'senior'
        
        df['seniority_level'] = df[exp_col].apply(categorize_seniority)
        df['seniority_level'] = df['seniority_level'].astype('category')
        print(f"✓ Created seniority_level feature based on {exp_col}")
    else:
        raise ValueError("Experience column not found. Expected: 'experience_years', 'years_of_experience', or 'experience'")
    
    return df


def encode_categoricals(df):
    """
    One-hot encode specific categorical columns: job_title, education_level, contract_type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one-hot encoded categorical columns
    """
    df = df.copy()
    
    # Find the columns to encode (handle different naming conventions)
    columns_to_encode = []
    
    # Map possible column names to standard names
    job_title_variants = ['job_title', 'position']
    education_variants = ['education', 'education_level']
    contract_variants = ['contract_type', 'contract', 'employment_type']
    
    # Find job_title
    for variant in job_title_variants:
        if variant in df.columns:
            columns_to_encode.append(variant)
            break
    
    # Find education
    for variant in education_variants:
        if variant in df.columns:
            columns_to_encode.append(variant)
            break
    
    # Find contract_type
    for variant in contract_variants:
        if variant in df.columns:
            columns_to_encode.append(variant)
            break
    
    if not columns_to_encode:
        print("⚠ Warning: No categorical columns found to encode (job_title, education_level, contract_type)")
        return df
    
    # One-hot encode the columns
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode, drop_first=True)
    
    encoded_count = len(df_encoded.columns) - len(df.columns)
    print(f"✓ One-hot encoded {len(columns_to_encode)} categorical columns: {', '.join(columns_to_encode)}")
    print(f"  Added {encoded_count} new columns")
    
    return df_encoded


def feature_engineering_pipeline(df, create_all=True, encode_categorical=False, 
                                 encoding_method='onehot'):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    create_all : bool, default=True
        Whether to create all engineered features
    encode_categorical : bool, default=False
        Whether to encode categorical features
    encoding_method : str, default='onehot'
        Method for encoding categorical features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    print("=" * 50)
    print("Starting Feature Engineering Pipeline")
    print("=" * 50)
    
    df_engineered = df.copy()
    
    if create_all:
        df_engineered = create_total_compensation(df_engineered)
        df_engineered = create_salary_per_hour(df_engineered)
        df_engineered = create_experience_level(df_engineered)
        df_engineered = create_age_group(df_engineered)
        df_engineered = create_years_at_company(df_engineered)
        df_engineered = create_education_rank(df_engineered)
        df_engineered = create_performance_category(df_engineered)
        df_engineered = create_bonus_percentage(df_engineered)
    
    if encode_categorical:
        df_engineered = encode_categorical_features(df_engineered, method=encoding_method)
    
    print("=" * 50)
    print(f"Feature engineering complete! Final shape: {df_engineered.shape}")
    print("=" * 50)
    
    return df_engineered

