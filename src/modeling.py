"""
Module for machine learning modeling on employee salary data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df, target_col='salary_usd', exclude_cols=None):
    """
    Prepare features and target for modeling.
    Selects numeric columns and encoded categorical features (one-hot encoded).
    
    Automatically excludes salary-derived features (e.g., salary_per_hour) to avoid target leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, default='salary_usd'
        Name of the target column
    exclude_cols : list, default=None
        List of columns to exclude from features
        
    Returns:
    --------
    tuple
        X (features DataFrame) and y (target Series)
    """
    if exclude_cols is None:
        exclude_cols = ['employee_id', 'name', target_col]
    
    # Always exclude salary-derived features to prevent target leakage
    salary_derived_features = ['salary_per_hour', 'total_compensation']
    for col in df.columns:
        if 'salary' in col.lower() and col != target_col:
            if col not in exclude_cols:
                exclude_cols.append(col)
    
    # Also explicitly exclude known salary-derived features
    for feature in salary_derived_features:
        if feature in df.columns and feature not in exclude_cols:
            exclude_cols.append(feature)
    
    # Select numeric columns
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    # Select encoded categorical features (columns that look like one-hot encoded)
    # These are typically binary (0/1) columns that start with categorical column names
    categorical_encoded = []
    for col in df.columns:
        if col not in exclude_cols and col not in numeric_cols:
            # Check if it's a binary column (likely one-hot encoded)
            unique_vals = df[col].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                categorical_encoded.append(col)
            # Also include if it's numeric but not in the main numeric list
            elif df[col].dtype in ['int64', 'float64']:
                categorical_encoded.append(col)
    
    # Combine all feature columns
    feature_cols = numeric_cols + categorical_encoded
    
    if not feature_cols:
        raise ValueError("No features found. Make sure the DataFrame has numeric or encoded categorical columns.")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    print(f"✓ Prepared {len(feature_cols)} features for modeling")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Encoded categorical features: {len(categorical_encoded)}")
    
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Split data: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    X_test : pd.DataFrame or np.array
        Testing features
        
    Returns:
    --------
    tuple
        Scaled X_train, X_test, and the scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Scaled features using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name='Model'):
    """
    Evaluate a regression model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    y_train : np.array
        Training target
    y_test : np.array
        Testing target
    model_name : str, default='Model'
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Train RMSE: ${train_rmse:,.2f}")
    print(f"Test RMSE:  ${test_rmse:,.2f}")
    print(f"Train MAE:  ${train_mae:,.2f}")
    print(f"Test MAE:   ${test_mae:,.2f}")
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    print(f"{'='*50}\n")
    
    return metrics


def train_linear_regression(X_train, X_test, y_train, y_test, scaled=True):
    """
    Train and evaluate a Linear Regression model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    y_train : np.array
        Training target
    y_test : np.array
        Testing target
    scaled : bool, default=True
        Whether features are scaled
        
    Returns:
    --------
    tuple
        Trained model and evaluation metrics
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, 
                            'Linear Regression')
    
    return model, metrics


def train_ridge_regression(X_train, X_test, y_train, y_test, alpha=1.0):
    """
    Train and evaluate a Ridge Regression model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    y_train : np.array
        Training target
    y_test : np.array
        Testing target
    alpha : float, default=1.0
        Regularization strength
        
    Returns:
    --------
    tuple
        Trained model and evaluation metrics
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, 
                            f'Ridge Regression (α={alpha})')
    
    return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, 
                        n_estimators=100, max_depth=None, random_state=42):
    """
    Train and evaluate a Random Forest model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    y_train : np.array
        Training target
    y_test : np.array
        Testing target
    n_estimators : int, default=100
        Number of trees
    max_depth : int, default=None
        Maximum depth of trees
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    tuple
        Trained model and evaluation metrics
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, 
                            'Random Forest')
    
    return model, metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test,
                            n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Train and evaluate a Gradient Boosting model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    y_train : np.array
        Training target
    y_test : np.array
        Testing target
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Learning rate
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    tuple
        Trained model and evaluation metrics
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, 
                            'Gradient Boosting')
    
    return model, metrics


def get_feature_importance(model, feature_names, top_n=15):
    """
    Get feature importance from a tree-based model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=15
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def compare_models(models_dict):
    """
    Compare multiple models and return a summary DataFrame.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and metrics dictionaries as values
        
    Returns:
    --------
    pd.DataFrame
        Comparison DataFrame
    """
    comparison_df = pd.DataFrame(models_dict).T
    
    # Reorder columns
    col_order = ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 
                 'train_r2', 'test_r2']
    comparison_df = comparison_df[col_order]
    
    print("\n" + "="*70)
    print("Model Comparison Summary")
    print("="*70)
    print(comparison_df.round(2))
    print("="*70 + "\n")
    
    return comparison_df


def train_models(X_train, y_train):
    """
    Train LinearRegression and RandomForestRegressor models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
        
    Returns:
    --------
    tuple
        (linear_model, random_forest_model)
    """
    print("=" * 60)
    print("Training Models")
    print("=" * 60)
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    print("✓ Linear Regression trained")
    
    # Train Random Forest
    print("\nTraining Random Forest Regressor...")
    random_forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    random_forest_model.fit(X_train, y_train)
    print("✓ Random Forest Regressor trained")
    
    print("=" * 60 + "\n")
    
    return linear_model, random_forest_model


def evaluate(model, X_test, y_test):
    """
    Evaluate a model and compute MAE, RMSE, and R2 metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Create results dictionary
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    # Print metrics
    print("=" * 60)
    print("Model Evaluation Metrics")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):  ${mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print("=" * 60 + "\n")
    
    return results


def plot_feature_importance(model, feature_names):
    """
    Plot and save feature importance as a horizontal bar chart.
    Only works for models that have feature_importances_ attribute.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (must have feature_importances_ attribute)
    feature_names : list or array-like
        List of feature names corresponding to the model's features
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠ Model does not have feature_importances_ attribute. Skipping plot.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(importance_df) * 0.3)))
    
    # Horizontal bar chart
    bars = ax.barh(range(len(importance_df)), importance_df['importance'].values,
                   color=sns.color_palette("viridis", len(importance_df)))
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to reports/plots/
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'reports' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'feature_importance.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved feature importance plot to {output_path}")

