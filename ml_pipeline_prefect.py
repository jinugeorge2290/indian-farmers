# ml_pipeline_prefect.py
from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

@task
def load_data(path):
    logging.info("Loading processed data...")
    return pd.read_csv(path)

@task
def train_models(data):
    logging.info("Training models...")
    X = data.drop(['Total_Suicides'], axis=1)
    y_reg = data['Total_Suicides']
    y_clf = (y_reg > y_reg.median()).astype(int)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.3, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    reg_model.fit(X_train_r, y_train_r)
    
    clf_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    clf_model.fit(X_train_c, y_train_c)
    
    return reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c

@task
def evaluate_models(reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c):
    logging.info("Evaluating models...")
    y_pred_r = reg_model.predict(X_test_r)
    #logging.info(f"Regression RMSE: {mean_squared_error(y_test_r, y_pred_r, squared=False):.2f}, R2: {r2_score(y_test_r, y_pred_r):.2f}")
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))  # safer than squared=False
    r2 = r2_score(y_test_r, y_pred_r)
    logging.info(f"Regression RMSE: {rmse:.2f}, R2: {r2:.2f}")

    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    precision = precision_score(y_test_c, y_pred_c)
    recall = recall_score(y_test_c, y_pred_c)
    f1 = f1_score(y_test_c, y_pred_c)
    logging.info(f"Classification metrics - Acc: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
@task
def save_models(reg_model, clf_model):
    with open("reg_model.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open("clf_model.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    logging.info("Models saved successfully")

@flow(name="Farmer Suicide ML Pipeline")
def ml_pipeline_flow():
    data = load_data("farmer_suicide_processed.csv")
    reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c = train_models(data)
    evaluate_models(reg_model, clf_model, X_test_r, y_test_r, X_test_c, y_test_c)
    save_models(reg_model, clf_model)

if __name__ == "__main__":
    ml_pipeline_flow()
