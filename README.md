# 📊 Employee Salary Analytics

## 📝 Executive Summary

This project analyzes a dataset of 1,200 employees with 17 features to understand salary patterns, identify key drivers of compensation, and build machine learning models to predict salary.

It includes a full end-to-end data pipeline: EDA, cleaning, feature engineering, modeling, visualization, and leakage correction.

Initial models appeared extremely accurate (R² ≈ 0.99), but this was due to target leakage from salary-derived features. After removing leakage, model performance dropped dramatically—revealing that the available features (age, education, job title, experience, contract type, bonus, performance score, etc.) do not fully explain salary variation.

This mirrors real-world HR challenges, where compensation is influenced by hidden organizational factors not represented in typical datasets (e.g., negotiation, company-level pay scales, market rates, seniority bands).

The final model highlights genuine salary drivers such as experience, age, work hours, bonus, and job-related features. All results follow best practices for ethical and leakage-free modeling.

## 📦 Project Structure

```
employee-salary-analytics/
│
├── data/
│   ├── raw/
│   │   └── employee_salaries.csv
│   └── processed/
│       └── salaries_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_salary_modeling.ipynb
│
├── reports/
│   └── plots/
│       ├── salary_distribution.png
│       ├── salary_vs_experience.png
│       ├── correlation_heatmap.png
│       └── feature_importance.png
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   └── modeling.py
│
└── README.md
```

## 📁 Dataset Overview

- **Rows**: 1,200
- **Columns**: 17

**Includes:**
- Age
- Gender
- Country, City
- Education (High School, Bachelor, Master, PhD)
- Job Title
- Department
- Experience (Years)
- Salary (USD)
- Bonus (USD)
- Work Hours per Week
- Contract Type
- Performance Score
- Joining Year

## 🔍 Exploratory Data Analysis (EDA)

### Key findings:

### 1. Salary Distribution
- Salary ranges approximately $30,000–$150,000.
- The distribution is fairly uniform, suggesting synthetic or standardized generation.
- Boxplot indicates a wide middle 50% with no extreme outliers.

### 2. Salary vs Experience
- Extremely weak relationship.
- Trendline shows almost no upward slope.
- Indicates salary is not strongly determined by experience alone in this dataset.

### 3. Correlations
- Salary has very low correlation with:
  - age
  - experience
  - work_hours_per_week
  - bonus
  - performance_score
- This supports the low model performance after leakage removal.

## 🛠 Feature Engineering

### Applied:

**Cleaning:**
- Dropped invalid salary rows
- Filled missing numeric values with median
- Filled categorical values with mode
- Removed duplicates
- Converted categorical features to lowercase

**Feature Engineering:**
- `seniority_level` derived from experience
- One-hot encoding of:
  - job title
  - education
  - contract type
- **Removed `salary_per_hour` due to target leakage**

**Processed dataset saved to:**
- `data/processed/salaries_clean.csv`

## 🤖 Modeling (After Leakage Fix)

Two models were trained:

### Random Forest (leakage-free)
- **R² = -0.0051**
- **MAE = $29,730**
- **RMSE = $34,573**

### Linear Regression (leakage-free)
- **R² = -0.0047**
- **MAE = $29,887**
- **RMSE = $34,567**

### Interpretation
The negative R² indicates:
- Salary cannot be accurately predicted from the available features.
- Key determinants of real compensation (company pay structure, negotiation, market rate, seniority bands) are missing.
- This is a realistic finding for HR salary datasets.

## 🚨 Data Leakage Discovery and Fix

Initially, a feature `salary_per_hour = salary / work_hours_per_week` was included.

This created severe target leakage because salary was indirectly passed into the model through a derived feature.

**Result before fix:**
- R² = 0.9971
- MAE ≈ $1,400
- This performance was too perfect, indicating leakage.

**After removing leakage:**
- R² ≈ -0.005
- Model now reflects true predictive capacity.

This correction demonstrates awareness of:
- proper feature design
- leakage detection
- ethical ML practices
- realistic modeling constraints

## ⭐ Feature Importance (Leakage-Free)

**Top predictors:**
1. `bonus_usd` (0.209)
2. `age` (0.143)
3. `experience_years` (0.141)
4. `work_hours_per_week` (0.131)
5. `joining_year` (0.128)
6. `performance_score` (0.089)

These patterns make sense for real compensation analyses.

## 📊 Visualizations

Saved in:
- `reports/plots/`

**Includes:**
- Salary Distribution
- Salary vs Experience
- Correlation Heatmap
- Feature Importance

These graphs provide a clear visual summary of salary structure and model behavior.

## ▶️ How to Run This Project

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 2. Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Run Feature Engineering

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 4. Run Modeling

```bash
jupyter notebook notebooks/03_salary_modeling.ipynb
```

All outputs, cleaned data, and plots will be saved automatically.

## 🚀 Future Improvements

- Incorporate more realistic features:
  - industry
  - company size
  - city cost-of-living
  - skills
  - certifications
  - promotions / seniority levels
- Train more advanced models (XGBoost, CatBoost)
- Build a Streamlit dashboard for interactive exploration
- Add SHAP interpretability plots
- Cluster employees by compensation patterns

## 🎯 Conclusion

This project demonstrates a full data science workflow with:

✔ Data exploration  
✔ Cleaning  
✔ Feature engineering  
✔ Leakage detection  
✔ Machine learning  
✔ Interpretation and reporting  
✔ Professional plots and structure  

It showcases strong analytical thinking, awareness of ML pitfalls, and the ability to build a clean, reproducible analytics pipeline.
