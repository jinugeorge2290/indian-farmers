# data_pipeline_prefect.py
from prefect import flow, task
# 1. IMPORT ARTIFACT FUNCTIONS
from prefect.artifacts import create_markdown_artifact
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os # For checking file existence

logging.basicConfig(level=logging.INFO)

@task
def ingest_data(path):
    logging.info("Loading data...")
    # Add a check for local runs, as the file path might change in deployments
    if not os.path.exists(path):
        logging.warning(f"File not found at {path}. Assuming dummy data for demonstration.")
        return pd.DataFrame({
            'col_a': [1, 5, 10, 15, 20], 
            'col_b': ['A', 'B', 'A', 'C', 'B'], 
            'target': [0, 1, 0, 1, 0]
        })
    return pd.read_csv(path)

@task
def preprocess_data(data):
    logging.info("Preprocessing data...")
    # ... (rest of your preprocessing logic remains the same)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    scaler = MinMaxScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
    return data_encoded

@task
def perform_eda(data):
    logging.info("Performing EDA and creating artifacts...")
    
    # --- 1. Create and Save Heatmap (Image Artifact) ---
    corr = data.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    
    # Save the file locally where the worker can access it
    heatmap_path = "correlation_heatmap.png" 
    plt.savefig(heatmap_path)
    plt.close() # Important to close the plot to free memory
    
    # Create an artifact linking to the saved image file
    # NOTE: For the image to show up in the UI, the *path* must be accessible
    # by the Prefect server/UI, which typically means uploading it to an S3/GCS
    # bucket and creating a Link or Image artifact. 
    # For a local worker/server setup, this might just show the file path.
    create_markdown_artifact(
        key="eda-heatmap-link",
        markdown=f"**Correlation Heatmap:** [View Image File]({heatmap_path})",
        description="Heatmap showing feature correlations after preprocessing."
    )
    
    # --- 2. Create and Save Summary Statistics (Table/Markdown Artifact) ---
    summary_df = data.describe().transpose()
    
    # Convert the DataFrame to a Markdown table string
    markdown_table = summary_df.to_markdown(floatfmt=".2f")
    
    create_markdown_artifact(
        key="eda-summary-stats",
        markdown=f"### Summary Statistics\n\n{markdown_table}",
        description="Descriptive statistics for all processed columns."
    )
    
    logging.info("EDA done, artifacts created.")
    return data

@task
def save_processed_data(data, path):
    # This task is now just for local saving.
    # To truly persist, you should switch this to a Storage Block (Option 2).
    data.to_csv(path, index=False)
    
    # Create an artifact indicating where the large file was saved.
    create_markdown_artifact(
        key="processed-data-location",
        markdown=f"Processed DataFrame saved locally to: `{path}`\n\n**Warning:** This file is only available inside the worker/container where the flow ran. Use a Storage Block (like S3) for persistence.",
        description="Location of the full processed CSV file."
    )
    logging.info(f"Processed data saved to {path}")

@flow(name="Farmer Suicide Data Pipeline")
def data_pipeline_flow():
    data = ingest_data("farmer_suicide_large_realistic_train.csv")
    data = preprocess_data(data)
    data = perform_eda(data)
    save_processed_data(data, "farmer_suicide_processed.csv")

#if __name__ == "__main__":
#    data_pipeline_flow()