# eda-bank-customer-churn-prediction
# Bank Customer Churn – Exploratory Data Analysis & Preprocessing Pipeline

This repository contains a comprehensive **Exploratory Data Analysis (EDA)** and data preprocessing pipeline focused on analyzing and preparing a retail banking dataset for customer churn prediction. 

The primary objective of this project is to uncover underlying statistical patterns, resolve realistic data quality constraints, execute robust feature selection, and construct a production-ready clean dataset.

## 🛠️ Advanced Technical Implementations

- **Model-Based Imputation:** Used **K-Nearest Neighbors (KNN) Imputer ($k=5$)** as imputation method. This advanced approach preserves multidimensional feature relationships and variance, eliminating artificial spikes in skewed distributions.
- **Outlier Engineering:** Identified and engineered logical domain constraints, such as handling extreme anomalies within the customer tenure features (`YearsWithBank` outliers transformed and properly imputed).
- **Object-Oriented Design (OOP):** Transformed sequential scripting into modular, production-grade functions (`clean_and_impute_data`, `handle_missing_data_knn`) ensuring complete code reusability.
- **Feature Selection Pipeline:** Deployed non-linear statistical evaluation metrics—including **Mutual Information (Information Gain)** and **Variance Thresholding**—alongside traditional Pearson correlation matrices.

## 📊 Key Analytical Insights

Despite industry expectations where parameters like `Income` or `Savings` drive loyalty, statistical analysis demonstrated an overall weak signal within this specific dataset:
- **LoanStatus** emerged as the only categorical feature providing a distinct visual/statistical separation regarding customer behavior.
- **LoanAmount** and **CreditScore** yielded the highest comparative Mutual Information scores (**0.0103** and **0.0037** respectively), establishing them as key continuous predictors.
- Multi-collinearity checks confirmed high feature independence, with all Pearson coefficients bound strictly between **-0.035** and **+0.035**.

## 📁 Repository Structure

- `Bank Customer Churn Prediction.ipynb` - The primary Jupyter Notebook containing the full documented python pipeline.
- `bank_customer_analytics.csv` - The source raw banking dataset.
- `final_cleaned_dataset.csv` - The optimized, preprocessed output dataset ready for Machine Learning modeling.

## 💻 Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Seaborn, Matplotlib
- **Machine Learning & Feature Selection:** Scikit-Learn (`KNNImputer`, `mutual_info_classif`, `VarianceThreshold`)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
