# Customer Churn Prediction — End-to-End ML Pipeline

A production-grade machine learning pipeline predicting customer churn for a Nigerian fintech/subscription business, using XGBoost with SHAP explainability and a full feature engineering workflow.

## Project Overview

This project demonstrates a complete ML workflow: data preprocessing, feature engineering, model training, hyperparameter tuning, SHAP-based explainability, and a deployable prediction API.

## Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.891 |
| Precision | 0.84 |
| Recall | 0.79 |
| F1 Score | 0.81 |

## Key Features
- Full preprocessing pipeline (imputation, encoding, scaling)
- Feature engineering: RFM scores, engagement decay, transaction velocity
- XGBoost classifier with Optuna hyperparameter tuning
- SHAP waterfall and summary plots for explainability
- Cross-validation with stratified k-fold
- Exportable model artifact (.pkl)

## Tools & Technologies
- Python: scikit-learn, XGBoost, SHAP, Optuna
- Jupyter Notebook for EDA and model development
- pandas, numpy, matplotlib, seaborn
- FastAPI (model serving endpoint)

## Project Structure
```
customer-churn-prediction/
├── data/                     # Raw and processed datasets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── models/                   # Saved model artifacts
├── src/
│   ├── features.py           # Feature engineering
│   ├── train.py              # Training pipeline
│   └── predict.py            # Inference
└── reports/                  # SHAP plots and model report
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
python src/train.py
```
