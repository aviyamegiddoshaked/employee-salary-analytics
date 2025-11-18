"""
Module for creating visualizations for employee salary data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_reports_plots_dir():
    """Get the reports/plots directory, creating it if it doesn't exist."""
    project_root = get_project_root()
    reports_dir = project_root / 'reports' / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def save_figure(fig, filename, dpi=300):
    """
    Save figure to a file.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str
        Name of the file to save
    dpi : int, default=300
        Resolution for saved figure
    """
    project_root = get_project_root()
    output_dir = project_root / 'notebooks' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")


def plot_distribution(df, column, title=None, bins=30, save=False):
    """
    Plot distribution of a numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    title : str, default=None
        Plot title
    bins : int, default=30
        Number of bins for histogram
    save : bool, default=False
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column.replace('_', ' ').title())
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {column.replace("_", " ").title()}')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[column].dropna(), vert=True)
    axes[1].set_ylabel(column.replace('_', ' ').title())
    axes[1].set_title(f'Box Plot of {column.replace("_", " ").title()}')
    axes[1].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, f'distribution_{column}.png')
    
    plt.show()


def plot_categorical_counts(df, column, title=None, top_n=None, save=False):
    """
    Plot count of categorical values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    title : str, default=None
        Plot title
    top_n : int, default=None
        Number of top categories to show
    save : bool, default=False
        Whether to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    value_counts = df[column].value_counts()
    
    if top_n:
        value_counts = value_counts.head(top_n)
    
    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                  color=sns.color_palette("husl", len(value_counts)))
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_xlabel(column.replace('_', ' ').title())
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Count by {column.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, f'counts_{column}.png')
    
    plt.show()


def plot_correlation_heatmap(df, numeric_only=True, save=False):
    """
    Plot correlation heatmap for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    numeric_only : bool, default=True
        Whether to include only numeric columns
    save : bool, default=False
        Whether to save the figure
    """
    if numeric_only:
        df_numeric = df.select_dtypes(include=[np.number])
    else:
        df_numeric = df
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation_matrix = df_numeric.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'correlation_heatmap.png')
    
    plt.show()


def plot_salary_by_category(df, category_col, title=None, save=False):
    """
    Plot salary distribution by category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    category_col : str
        Categorical column to group by
    title : str, default=None
        Plot title
    save : bool, default=False
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    categories = df[category_col].unique()
    data_to_plot = [df[df[category_col] == cat]['salary_usd'].values 
                    for cat in categories]
    
    axes[0].boxplot(data_to_plot, labels=categories)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].set_ylabel('Salary (USD)')
    axes[0].set_title(f'Salary Distribution by {category_col.replace("_", " ").title()}')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot of mean salaries
    mean_salaries = df.groupby(category_col)['salary_usd'].mean().sort_values(ascending=False)
    bars = axes[1].bar(range(len(mean_salaries)), mean_salaries.values,
                      color=sns.color_palette("viridis", len(mean_salaries)))
    axes[1].set_xticks(range(len(mean_salaries)))
    axes[1].set_xticklabels(mean_salaries.index, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Salary (USD)')
    axes[1].set_title(f'Mean Salary by {category_col.replace("_", " ").title()}')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'${int(height):,}',
                    ha='center', va='bottom', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, f'salary_by_{category_col}.png')
    
    plt.show()


def plot_scatter_with_regression(df, x_col, y_col, title=None, save=False):
    """
    Plot scatter plot with regression line.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        X-axis column
    y_col : str
        Y-axis column
    title : str, default=None
        Plot title
    save : bool, default=False
        Whether to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df[x_col], df[y_col], alpha=0.5, s=50)
    
    # Add regression line
    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2, 
            label=f'Linear Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}',
                    fontsize=12, fontweight='bold')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, f'scatter_{x_col}_vs_{y_col}.png')
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, title='Feature Importance', save=False):
    """
    Plot feature importance from a DataFrame.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=15
        Number of top features to show
    title : str, default='Feature Importance'
        Plot title
    save : bool, default=False
        Whether to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color=sns.color_palette("viridis", len(top_features)))
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'feature_importance.png')
    
    plt.show()


def plot_salary_distribution(df):
    """
    Plot salary distribution (histogram and box plot).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with salary column
    """
    # Find salary column (handle both raw and cleaned data)
    salary_col = None
    for col in df.columns:
        if col.lower() in ['salary_usd', 'salary']:
            salary_col = col
            break
    
    if not salary_col:
        raise ValueError("Salary column not found. Expected 'salary_usd', 'Salary_USD', or 'salary'")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(df[salary_col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Salary (USD)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Salary Distribution (Histogram)')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[salary_col].dropna(), vert=True)
    axes[1].set_ylabel('Salary (USD)')
    axes[1].set_title('Salary Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Salary Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to reports/plots/
    output_dir = get_reports_plots_dir()
    output_path = output_dir / 'salary_distribution.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def plot_salary_by_education(df):
    """
    Plot salary by education level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with salary and education columns
    """
    # Find columns (handle both raw and cleaned data)
    salary_col = None
    education_col = None
    
    for col in df.columns:
        if col.lower() in ['salary_usd', 'salary']:
            salary_col = col
            break
    
    for col in df.columns:
        if col.lower() in ['education', 'education_level']:
            education_col = col
            break
    
    if not salary_col:
        raise ValueError("Salary column not found. Expected 'salary_usd', 'Salary_USD', or 'salary'")
    if not education_col:
        raise ValueError("Education column not found. Expected 'education', 'Education', or 'education_level'")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    education_levels = df[education_col].unique()
    data_to_plot = [df[df[education_col] == level][salary_col].values 
                    for level in education_levels]
    
    axes[0].boxplot(data_to_plot, labels=education_levels)
    axes[0].set_xticklabels(education_levels, rotation=45, ha='right')
    axes[0].set_ylabel('Salary (USD)')
    axes[0].set_title('Salary Distribution by Education Level')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot of mean salaries
    mean_salaries = df.groupby(education_col)[salary_col].mean().sort_values(ascending=False)
    bars = axes[1].bar(range(len(mean_salaries)), mean_salaries.values,
                      color=sns.color_palette("viridis", len(mean_salaries)))
    axes[1].set_xticks(range(len(mean_salaries)))
    axes[1].set_xticklabels(mean_salaries.index, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Salary (USD)')
    axes[1].set_title('Mean Salary by Education Level')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'${int(height):,}',
                    ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Salary Analysis by Education Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to reports/plots/
    output_dir = get_reports_plots_dir()
    output_path = output_dir / 'salary_by_education.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def plot_salary_by_job_title(df, top_n=10):
    """
    Plot salary by job title (top N job titles by count).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with salary and job_title columns
    top_n : int, default=10
        Number of top job titles to show
    """
    # Find columns (handle both raw and cleaned data)
    salary_col = None
    job_title_col = None
    
    for col in df.columns:
        if col.lower() in ['salary_usd', 'salary']:
            salary_col = col
            break
    
    for col in df.columns:
        if col.lower() in ['job_title', 'position']:
            job_title_col = col
            break
    
    if not salary_col:
        raise ValueError("Salary column not found. Expected 'salary_usd', 'Salary_USD', or 'salary'")
    if not job_title_col:
        raise ValueError("Job title column not found. Expected 'job_title', 'Job_Title', or 'position'")
    
    # Get top N job titles by count
    top_job_titles = df[job_title_col].value_counts().head(top_n).index
    
    # Filter data to top N job titles
    df_filtered = df[df[job_title_col].isin(top_job_titles)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    data_to_plot = [df_filtered[df_filtered[job_title_col] == title][salary_col].values 
                    for title in top_job_titles]
    
    axes[0].boxplot(data_to_plot, labels=top_job_titles)
    axes[0].set_xticklabels(top_job_titles, rotation=45, ha='right')
    axes[0].set_ylabel('Salary (USD)')
    axes[0].set_title(f'Salary Distribution by Job Title (Top {top_n})')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot of mean salaries
    mean_salaries = df_filtered.groupby(job_title_col)[salary_col].mean().reindex(top_job_titles)
    bars = axes[1].bar(range(len(mean_salaries)), mean_salaries.values,
                      color=sns.color_palette("viridis", len(mean_salaries)))
    axes[1].set_xticks(range(len(mean_salaries)))
    axes[1].set_xticklabels(mean_salaries.index, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Salary (USD)')
    axes[1].set_title(f'Mean Salary by Job Title (Top {top_n})')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'${int(height):,}',
                    ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'Salary Analysis by Job Title (Top {top_n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to reports/plots/
    output_dir = get_reports_plots_dir()
    output_path = output_dir / 'salary_by_job_title.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        raise ValueError("No numeric columns found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation_matrix = df_numeric.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save to reports/plots/
    output_dir = get_reports_plots_dir()
    output_path = output_dir / 'correlation_heatmap.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def plot_salary_vs_experience(df):
    """
    Plot salary vs experience with regression line.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with salary and experience columns
    """
    # Find columns (handle both raw and cleaned data)
    salary_col = None
    experience_col = None
    
    for col in df.columns:
        if col.lower() in ['salary_usd', 'salary']:
            salary_col = col
            break
    
    for col in df.columns:
        if col.lower() in ['experience_years', 'years_of_experience', 'experience']:
            experience_col = col
            break
    
    if not salary_col:
        raise ValueError("Salary column not found. Expected 'salary_usd', 'Salary_USD', or 'salary'")
    if not experience_col:
        raise ValueError("Experience column not found. Expected 'experience_years', 'Experience_Years', 'years_of_experience', or 'experience'")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(df[experience_col], df[salary_col], alpha=0.5, s=50, color='steelblue')
    
    # Add regression line
    mask = df[[experience_col, salary_col]].notna().all(axis=1)
    if mask.sum() > 1:
        z = np.polyfit(df.loc[mask, experience_col], df.loc[mask, salary_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[experience_col].min(), df[experience_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                label=f'Linear Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()
    
    ax.set_xlabel('Experience (Years)')
    ax.set_ylabel('Salary (USD)')
    ax.set_title('Salary vs Experience', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to reports/plots/
    output_dir = get_reports_plots_dir()
    output_path = output_dir / 'salary_vs_experience.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")

