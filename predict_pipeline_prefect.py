# predict_pipeline_prefect.py
from prefect import flow, task, get_run_logger
import pandas as pd
import pickle
import os

@task
def load_models(reg_path="reg_model.pkl", clf_path="clf_model.pkl"):
    logger = get_run_logger()
    logger.info(f"Loading trained models from {reg_path} and {clf_path}")
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

@task
def preprocess_new_data(new_data_path="farmer_suicide_large_realistic.csv",
                        processed_train_path="farmer_suicide_processed.csv"):
    logger = get_run_logger()
    logger.info(f"Loading new data from {new_data_path}")
    new_data = pd.read_csv(new_data_path)

    # Encode categorical columns (must match training preprocessing)
    categorical_cols = ['State','District','Crop_Type']
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

    # Add any engineered features if present
    if 'Loan_Amount' in new_data_encoded.columns and 'Farmer_Income' in new_data_encoded.columns:
        new_data_encoded['Debt_Income_Ratio'] = new_data_encoded['Loan_Amount'] / new_data_encoded['Farmer_Income']

    # Align columns with training data
    train_columns = pd.read_csv(processed_train_path).drop(['Total_Suicides'], axis=1).columns
    new_data_encoded = new_data_encoded.reindex(columns=train_columns, fill_value=0)

    logger.info("New data preprocessed and aligned with training features")
    return new_data, new_data_encoded

@task
def make_predictions(new_data, new_data_encoded, reg_model, clf_model):
    logger = get_run_logger()
    logger.info("Making predictions...")
    regression_preds = reg_model.predict(new_data_encoded)
    classification_preds = clf_model.predict(new_data_encoded)

    new_data['Predicted_Suicides'] = regression_preds
    new_data['High_Risk'] = classification_preds

    os.makedirs("prediction_outputs", exist_ok=True)
    output_path = os.path.join("prediction_outputs", "predictions.csv")
    new_data.to_csv(output_path, index=False)
    logger.info(f"Predictions saved at {output_path}")
    return output_path

@flow(name="Farmer Suicide Prediction Flow")
def prediction_flow(new_data_path="farmer_suicide_large_realistic.csv"):
    reg_model, clf_model = load_models()
    new_data, new_data_encoded = preprocess_new_data(new_data_path=new_data_path)
    output_path = make_predictions(new_data, new_data_encoded, reg_model, clf_model)
    return output_path

if __name__ == "__main__":
    prediction_flow()
