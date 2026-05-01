"""
XGBoost Churn Model Training Pipeline with SHAP Explainability
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from features import generate_sample_dataset

FEATURES = [
    "tenure_months", "monthly_charges", "total_transactions",
    "avg_transaction_value", "days_since_last_txn",
    "support_tickets", "product_count", "digital_engagement_score",
]
TARGET = "churned"
MODEL_PATH = "models/churn_model.pkl"

def train_model():
    print("Loading data...")
    df = generate_sample_dataset(n_customers=5000)
    
    X = df[FEATURES]
    y = df[TARGET]
    
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.1%}")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42,
        eval_metric="auc",
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    model = train_model()
