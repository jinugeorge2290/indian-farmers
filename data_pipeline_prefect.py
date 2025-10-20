# data_pipeline_prefect.py
from prefect import flow, task
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

@task
def ingest_data(path):
    logging.info("Loading data...")
    return pd.read_csv(path)

@task
def preprocess_data(data):
    logging.info("Preprocessing data...")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    scaler = MinMaxScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
    return data_encoded

@task
def perform_eda(data):
    logging.info("Performing EDA...")
    corr = data.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.savefig("correlation_heatmap.png")
    data.describe().to_csv("eda_summary_statistics.csv")
    logging.info("EDA done, files saved")
    return data

@task
def save_processed_data(data, path):
    data.to_csv(path, index=False)
    logging.info(f"Processed data saved to {path}")

@flow(name="Farmer Suicide Data Pipeline")
def data_pipeline_flow():
    data = ingest_data("farmer_suicide_large_realistic_train.csv")
    data = preprocess_data(data)
    data = perform_eda(data)
    save_processed_data(data, "farmer_suicide_processed.csv")

#if __name__ == "__main__":
#    data_pipeline_flow()
