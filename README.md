# **HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES**

This project was developed as part of the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal was to predict house sale prices using a dataset with 79 explanatory variables describing nearly every aspect of residential homes in Ames, Iowa.
<br>
<br>

## **PROJECT OBJECTIVE**

Build a robust regression model capable of accurately predicting house prices using advanced data preprocessing, feature engineering, and ensemble modeling techniques.
<br>
<br>

## **LANGUAGES AND TOOLS**

- **Programming Language**: Python;
- **Data Manipulation**: `pandas`, `numpy`;
- **Visualization**: `matplotlib`, `seaborn`;
- **Modeling**: `XGBoost`, `LightGBM`, `CatBoost`, `GradientBoostingRegressor`;
- **Ensemble Modeling**: `StackingRegressor`;
- **Hyperparameter Tuning**: `Optuna`;
- **Preprocessing**: `MaxAbsScaler`, `K-Fold Target Encoding`, `VIF`.
<br>

## **PROJECT WORKFLOW**

### 1. Missing Data Handling
- Missing values were treated **case-by-case**, with contextual understanding of each feature;
- No blind imputation or mass dropping of data to preserve useful information.

### 2. Outlier Analysis
- Outliers were also evaluated **individually** to avoid discarding valuable data;
- Visual and statistical analysis were used to detect and decide on their treatment.

### 3. Exploratory Data Analysis (EDA)
- Target variable distribution analysis and normalization;
- **K-Fold Target Encoding** for categorical features, followed by correlation analysis with the target;
- Feature importance and redundancy evaluation using **correlation matrix** and **Variance Inflation Factor (VIF)**;
- **Feature Engineering** to uncover new patterns and relationships between features and the target.

### 4. Data Normalization
- Used **MaxAbsScaler** to normalize the data and maintain sparsity of the features.

### 5. Modeling and Optimization
- Trained four regression models: **LightGBM**, **XGBoost**, **CatBoost**, and **GradientBoosting**;
- Used **Optuna** for hyperparameter tuning to optimize model performance;
- Developed a final stacking ensemble, using **XGBoost Regressor as meta-model**, achieving an **RMSE of 0.10**.
<br>

## **MODELING**

Four base models were trained and tuned:

- **LightGBM Regressor**;
- **XGBoost Regressor**;
- **CatBoost Regressor**;
- **GradientBoosting Regressor**.

All models underwent **hyperparameter tuning using Optuna**, a powerful optimization framework.

### Final Model - Stacking Regressor
- A **Stacking Ensemble** was built using the four models above, with **XGBoost Regressor as the meta-model**;
- Achieved a final score of **RMSE = 0.10** on the test set.
<br>

## **PROJECT STRUCTURE**

```bash
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter Notebooks (EDA, modeling, etc.)
├── models/                 # Saved model files and evaluation results
├── visuals/                # Plots and visualizations
├── README.md               # Project documentation
└── requirements.txt        # Required Python packages
```
<br>

## **RESULTS**

- Final RMSE: **0.10**;
- Strong performance thanks to:
  - Feature engineering and encoding;
  - Individualized outlier/missing data treatment;
  - Model stacking and hyperparameter tuning.
<br>

## **WHAT I LEARNED**

- Importance of **understanding data context** before applying preprocessing steps;
- **Advanced encoding** techniques like K-Fold Target Encoding;
- Combining multiple models through **stacking** for better performance;
- Effectiveness of **Optuna** for hyperparameter optimization.
<br>

## **FUTURE IMPROVEMENTS**

- Add cross-validation visual analysis for each model;
- Deploy the model using a simple API;
- Automate the preprocessing pipeline with custom transformers.
<br>

## **AUTHOR**

**Fábio Galdino**