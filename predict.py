# use_trained_model.py
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # -----------------------------
    # 1. Load Trained Models
    # -----------------------------
    with open("reg_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("clf_model.pkl", "rb") as f:
        clf_model = pickle.load(f)
    logging.info("Trained models loaded successfully")

    # -----------------------------
    # 2. Load New Data
    # -----------------------------
    # The new data should have the same features as processed training data
    new_data = pd.read_csv("farmer_suicide_large_realistic.csv")  # replace with your new CSV

    # -----------------------------
    # 3. Preprocess New Data
    # -----------------------------
    # Ensure same preprocessing as training
    # 3a. Encode categorical variables
    new_data_encoded = pd.get_dummies(new_data, columns=['State','District','Crop_Type'], drop_first=True)

    # 3b. Add feature if needed
    if 'Loan_Amount' in new_data_encoded.columns and 'Farmer_Income' in new_data_encoded.columns:
        new_data_encoded['Debt_Income_Ratio'] = new_data_encoded['Loan_Amount'] / new_data_encoded['Farmer_Income']

    # 3c. Align columns with training data
    # Load processed training data columns
    train_columns = pd.read_csv("farmer_suicide_processed.csv").drop(['Total_Suicides'], axis=1).columns
    new_data_encoded = new_data_encoded.reindex(columns=train_columns, fill_value=0)

    # -----------------------------
    # 4. Make Predictions
    # -----------------------------
    regression_preds = reg_model.predict(new_data_encoded)
    classification_preds = clf_model.predict(new_data_encoded)

    # Add predictions to dataframe
    new_data['Predicted_Suicides'] = regression_preds
    new_data['High_Risk'] = classification_preds

    # -----------------------------
    # 5. Save Predictions
    # -----------------------------
    new_data.to_csv("predictions.csv", index=False)
    logging.info("Predictions saved to 'predictions.csv'")
    logging.info(new_data.head())

if __name__ == "__main__":
    main()
