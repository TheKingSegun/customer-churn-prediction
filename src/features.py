"""
Feature Engineering for Customer Churn Prediction
Computes RFM scores, engagement metrics, and transaction velocity features.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def compute_rfm(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) scores.
    
    Args:
        df: DataFrame with columns [customer_id, transaction_date, amount]
        snapshot_date: Reference date (defaults to max date in data)
    
    Returns:
        DataFrame with RFM features per customer
    """
    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max()
    
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    
    rfm = df.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
        avg_transaction=("amount", "mean"),
        std_transaction=("amount", "std"),
    ).reset_index()
    
    # RFM quintile scoring (1=worst, 5=best)
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    
    return rfm

def compute_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement decay and activity trend features."""
    df = df.sort_values(["customer_id", "transaction_date"])
    
    # Days between consecutive transactions
    df["days_since_last"] = df.groupby("customer_id")["transaction_date"].diff().dt.days
    
    # Rolling 30-day transaction count
    df["rolling_30d_txns"] = (
        df.groupby("customer_id")
        .rolling("30D", on="transaction_date")["amount"]
        .count()
        .reset_index(drop=True)
    )
    
    return df

def build_churn_labels(df: pd.DataFrame, churn_window_days: int = 30) -> pd.DataFrame:
    """
    Label customers as churned if no activity in the last N days.
    
    Args:
        df: Customer-level dataframe with last_active_date
        churn_window_days: Inactivity threshold for churn definition
    
    Returns:
        DataFrame with binary churn label
    """
    snapshot = df["last_active_date"].max()
    df["days_inactive"] = (snapshot - df["last_active_date"]).dt.days
    df["churned"] = (df["days_inactive"] >= churn_window_days).astype(int)
    print(f"Churn rate: {df['churned'].mean():.1%}")
    return df

def generate_sample_dataset(n_customers: int = 2000) -> pd.DataFrame:
    """Generate a realistic customer churn dataset for demo."""
    np.random.seed(42)
    customer_ids = [f"CUST_{i:05d}" for i in range(n_customers)]
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": np.random.randint(1, 60, n_customers),
        "monthly_charges": np.random.uniform(2000, 25000, n_customers),
        "total_transactions": np.random.randint(1, 120, n_customers),
        "avg_transaction_value": np.random.uniform(500, 50000, n_customers),
        "days_since_last_txn": np.random.exponential(25, n_customers).astype(int),
        "support_tickets": np.random.poisson(1.2, n_customers),
        "product_count": np.random.randint(1, 5, n_customers),
        "digital_engagement_score": np.random.uniform(0, 100, n_customers),
    })
    
    # Simulate churn probability based on features
    churn_prob = (
        0.3 * (df["days_since_last_txn"] > 30).astype(float) +
        0.2 * (df["tenure_months"] < 6).astype(float) +
        0.2 * (df["support_tickets"] > 2).astype(float) +
        0.1 * (df["product_count"] == 1).astype(float) +
        np.random.uniform(0, 0.2, n_customers)
    )
    df["churned"] = (churn_prob > 0.4).astype(int)
    
    return df

if __name__ == "__main__":
    df = generate_sample_dataset()
    print(df.describe())
    print(f"\nChurn rate: {df['churned'].mean():.1%}")
