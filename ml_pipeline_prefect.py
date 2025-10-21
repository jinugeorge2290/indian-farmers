# ml_pipeline_prefect.py
from prefect import flow, task, get_run_logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

@task
def load_data(path):
    logger = get_run_logger()
    logger.info(f"Loading processed data from {path}...")
    data = pd.read_csv(path)
    logger.info(f"Data loaded with shape: {data.shape}")
    return data

@task
def train_models(data):
    logger = get_run_logger()
    logger.info("Training models...")
    
    X = data.drop(['Total_Suicides'], axis=1)
    y_reg = data['Total_Suicides']
    y_clf = (y_reg > y_reg.median()).astype(int)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.3, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    reg_model.fit(X_train_r, y_train_r)
    
    clf_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    clf_model.fit(X_train_c, y_train_c)
    
    logger.info("Models trained successfully.")
    return reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c

@task
def evaluate_models(reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c):
    logger = get_run_logger()
    logger.info("Evaluating models...")
    
    # Regression metrics
    y_pred_r = reg_model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    logger.info(f"Regression RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # Classification metrics
    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    precision = precision_score(y_test_c, y_pred_c)
    recall = recall_score(y_test_c, y_pred_c)
    f1 = f1_score(y_test_c, y_pred_c)
    logger.info(f"Classification metrics - Acc: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

@task
def save_models(reg_model, clf_model):
    logger = get_run_logger()
    
    # Create output directory
    os.makedirs("model_outputs", exist_ok=True)
    
    reg_path = os.path.join("model_outputs", "reg_model.pkl")
    clf_path = os.path.join("model_outputs", "clf_model.pkl")
    
    with open(reg_path, "wb") as f:
        pickle.dump(reg_model, f)
    with open(clf_path, "wb") as f:
        pickle.dump(clf_model, f)
    
    logger.info("Models saved successfully.")
    
    # Log models as Prefect artifacts
    logger.add_artifact(reg_path)
    logger.add_artifact(clf_path)

@flow(name="Farmer Suicide ML Pipeline")
def ml_pipeline_flow():
    # Load processed CSV from data pipeline outputs
    data = load_data("processed_outputs/farmer_suicide_processed.csv")
    
    # Train models
    reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c = train_models(data)
    
    # Evaluate
    evaluate_models(reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c)
    
    # Save models and register as Prefect artifacts
    save_models(reg_model, clf_model)

#if __name__ == "__main__":
#    ml_pipeline_flow()
