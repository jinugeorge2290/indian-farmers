import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
import logging
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting ML Pipeline")

    # -----------------------------
    # 1. Load Processed Data
    # -----------------------------
    data = pd.read_csv("farmer_suicide_processed.csv")
    logging.info(f"Processed data loaded with shape {data.shape}")

    # -----------------------------
    # 2. Features and Targets
    # -----------------------------
    if 'Total_Suicides' not in data.columns:
        raise ValueError("Total_Suicides column not found in processed data")
        
    X = data.drop(['Total_Suicides'], axis=1)
    y_reg = data['Total_Suicides']
    y_clf = (y_reg > y_reg.median()).astype(int)  # High-risk classification

    # -----------------------------
    # 3. Train-Test Split
    # -----------------------------
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.3, random_state=42)

    # -----------------------------
    # 4. Train Models
    # -----------------------------
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    reg_model.fit(X_train_r, y_train_r)
    logging.info("Random Forest Regressor trained")

    clf_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    clf_model.fit(X_train_c, y_train_c)
    logging.info("Random Forest Classifier trained")

    # -----------------------------
    # 5. Evaluate Models
    # -----------------------------
    # Regression
    y_pred_r = reg_model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))  # safer than squared=False
    r2 = r2_score(y_test_r, y_pred_r)
    logging.info(f"Regression RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Classification
    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    precision = precision_score(y_test_c, y_pred_c, zero_division=0)
    recall = recall_score(y_test_c, y_pred_c, zero_division=0)
    f1 = f1_score(y_test_c, y_pred_c, zero_division=0)

    logging.info(f"Classification Accuracy: {acc:.3f}")
    logging.info(f"Classification Precision: {precision:.3f}")
    logging.info(f"Classification Recall: {recall:.3f}")
    logging.info(f"Classification F1 Score: {f1:.3f}")

    logging.info("\n" + classification_report(y_test_c, y_pred_c, zero_division=0))

    # -----------------------------
    # 6. Save Trained Models
    # -----------------------------
    with open("reg_model.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open("clf_model.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    logging.info("Trained models saved as 'reg_model.pkl' and 'clf_model.pkl'")

    logging.info("ML Pipeline completed successfully!")

if __name__ == "__main__":
    main()
