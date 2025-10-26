
# Indian-Farmers suicide Prediction

`Indian-Farmers`
The project addresses the issue of farmer suicides in India by analyzing socioeconomic and agricultural factors that contribute to distress.
Objective: build a data pipeline to automate data ingestion, preprocessing, and EDA — forming the foundation for predictive modeling to support preventive action.


## Introduction

This project analyses the dataset `farmer_suicide_large_realistic.csv` on Indian farmers (and a train split), constructs data-pipelines (data preparation, feature engineering), builds ML models (classification/regression) and uses Prefect to orchestrate these workflows.  
The key scripts include:

- `data_pipeline.py` / `data_pipeline_prefect.py` — data ingestion & prep
- `ml_pipeline.py` / `ml_pipeline_prefect.py` — model training
- `predict.py` / `predict_pipeline_prefect.py` — making predictions
- `flow_creation.py` — defines workflows
- `master_pipeline.py` — master orchestration script
- `lambda_data_pipeline.py` — AWS Lambda version of the pipeline
- `requirements.txt`, `prefect.yaml` – configuration files
- `clf_model.pkl`, `reg_model.pkl` – pre-trained model pickles
- `correlation_heatmap.png` – visual output


## Setting up Prefect Cloud

1. Go to the Prefect Cloud website and **sign up / log in**. [Prefect+2Prefect+2](https://www.prefect.io/cloud?utm_source=chatgpt.com)

2. In your local terminal, install Prefect:
    `pip install -U prefect`   
    
3. Connect your local environment to Prefect Cloud:
    `prefect cloud login`
    This will prompt you to either login via browser or paste an API key. 

Create a API Key here to authenticate

https://app.prefect.cloud/settings/api-keys
```
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/[ACCOUNT-ID]/workspaces/[WORKSPACE-ID]"

export PREFECT_API_KEY="[YOUR_API_KEY]"
```

4. Clone this repository:
    
    `git clone https://github.com/jinugeorge2290/indian-farmers.git cd indian-farmers`
    
5. Create & activate a virtual environment:
    
    ```
    python3 -m venv venv 
    source venv/bin/activate
    ```
6. Install requirements:
    
    ```
    pip3 install -r requirements.txt`
    ```
7. Confirm Prefect is installed & working:
    
    ```
    prefect version
    ```
    ```
    Version:              3.4.24
    API version:          0.8.4
    Python version:       3.9.6
    Git commit:           0472bb5a
    Built:                Thu, Oct 16, 2025 06:36 PM
    OS/Arch:              darwin/x86_64
    Profile:              ephemeral
    Server type:          cloud
    Pydantic version:     2.11.9
    Server:
    Database:           sqlite
    SQLite version:     3.43.2
    ```
    
5. Update any config (if needed) – e.g., in `prefect.yaml` you may set your workspace, queue names, or storage paths.

    ```
    prefect deploy
    ```

---


# Manually Run Each Task 
- **Generate or load dataset** — `generate_data.py` or the included CSVs (`farmer_suicide_large_realistic*.csv`) provide synthetic / prepared data.
    
- **Data preprocessing / ETL** — `data_pipeline.py` (and `data_pipeline_prefect.py`) transform/clean data and produce train/test CSVs or features.
    
- **Model training** — `ml_pipeline.py` (and `ml_pipeline_prefect.py`) train ML models and save trained artifacts like `reg_model.pkl` and `clf_model.pkl`.
    
- **Prediction / inference** — `predict.py` uses the saved models to produce predictions; `predict_pipeline_prefect.py` is the Prefect-flavored flow for inference.
    
- **Orchestration** — `master_pipeline.py` likely calls multiple pipelines in sequence (generate → data → train → predict). Prefect files and `prefect.yaml` let you run flows with Prefect (or build deployments).
    
- **Extras** — `flow_creation.py`, `lambda_data_pipeline.py` suggest alternate execution options (e.g., AWS Lambda or programmatic flow creation). `correlation_heatmap.png` is a result/visualization already saved.

## Pre-Requisite - How to run it

```bash
# clone (if you haven't)
git clone https://github.com/jinugeorge2290/indian-farmers.git
cd indian-farmers

# create a python virtualenv and activate it (macOS / linux)
python3 -m venv .venv
source .venv/bin/activate

# or on macOS with zsh: source .venv/bin/activate
```

## Step 1 — install dependencies

```
# inspect requirements first
less requirements.txt

# install
pip install -r requirements.txt
```

## Step 2 — create / inspect data

```bash
# generate synthetic / prepared dataset
python generate_data.py
```
Output:
```bash
CSV file 'farmer_suicide_large_realistic_train.csv' generated successfully!
```
What this does: generate_data.py writes CSV(s) 
(e.g., farmer_suicide_large_realistic.csv) to the repo directory. After it runs, list CSVs:
```
ls -lah *.csv
head -n 5 farmer_suicide_large_realistic.csv
```

## Step 3 — run the data pipeline

```bash
python data_pipeline.py
```
This runs the ETL: 
clean, transform, split into train/test and write files ( *_train.csv, feature CSVs).

Output
```bash
python3 data_pipeline.py                                                                      ✔  prefect-env   at 02:52:26 PM  
2025-10-25 15:00:10,050 - Starting Data Pipeline
2025-10-25 15:00:10,440 - Data loaded with shape (300000, 11)
2025-10-25 15:00:10,502 - Filled missing values for numeric columns: ['Year', 'Total_Suicides', 'Loan_Amount', 'Rainfall', 'Temp', 'Irrigated_Land', 'Farmer_Income', 'Poverty_Rate']
2025-10-25 15:00:10,624 - Encoded categorical columns: ['State', 'District', 'Crop_Type']
2025-10-25 15:00:10,689 - Normalized numeric columns using MinMaxScaler
2025-10-25 15:00:10,692 - Added Debt_Income_Ratio feature
2025-10-25 15:00:16,539 - Correlation heatmap saved as 'correlation_heatmap.png'
2025-10-25 15:00:16,695 - Summary statistics saved as 'eda_summary_statistics.csv'
2025-10-25 15:00:25,089 - Processed data saved as 'farmer_suicide_processed.csv'
2025-10-25 15:00:25,089 - Data Pipeline completed successfully!
```


## Step 4 — train ML models

Plain training:
```bash
python ml_pipeline.py
```
Output
```bash
python3 ml_pipeline.py                                                       2 ✘  prefect-env   at 03:17:55 PM  
2025-10-25 15:18:05,229 - Starting ML Pipeline
2025-10-25 15:18:06,910 - Processed data loaded with shape (300000, 73)
2025-10-25 15:19:44,455 - Random Forest Regressor trained
2025-10-25 15:19:56,940 - Random Forest Classifier trained
2025-10-25 15:19:57,313 - Regression RMSE: 0.07, R2: 0.78
2025-10-25 15:19:57,856 - Classification Accuracy: 0.871
2025-10-25 15:19:57,856 - Classification Precision: 0.858
2025-10-25 15:19:57,856 - Classification Recall: 0.883
2025-10-25 15:19:57,856 - Classification F1 Score: 0.870
2025-10-25 15:19:57,879 - 
              precision    recall  f1-score   support

           0       0.88      0.86      0.87     46077
           1       0.86      0.88      0.87     43923

    accuracy                           0.87     90000
   macro avg       0.87      0.87      0.87     90000
weighted avg       0.87      0.87      0.87     90000

2025-10-25 15:19:57,900 - Trained models saved as 'reg_model.pkl' and 'clf_model.pkl'
2025-10-25 15:19:57,900 - ML Pipeline completed successfully!
```


## Step 5 — test / predict (single-step)
```bash
python predict.py
```
predict.py reads the saved model(s) and a test CSV and writes predictions (or prints them). After running, check for a predictions CSV or console output.

Output

```bash
python3 predict.py                                                  ✔  prefect-env   at 03:43:13 PM  

2025-10-25 15:43:27,365 - Trained models loaded successfully
2025-10-25 15:43:32,378 - Predictions saved to 'predictions.csv'
2025-10-25 15:43:32,378 -            State     District  Year  Total_Suicides Crop_Type  ...  Irrigated_Land  Farmer_Income  Poverty_Rate  Predicted_Suicides  High_Risk
0      Rajasthan      Udaipur  2018              82   Soybean  ...             138          61352     31.709943            0.426281          1
1     Tamil Nadu      Chennai  2011              73     Wheat  ...             411          67020     35.959660            0.414231          1
2          Bihar  Muzaffarpur  2013              80    Cotton  ...             274         208251     35.347767            0.419609          1
3  Uttar Pradesh     Varanasi  2024              86     Wheat  ...             427         110038     22.089162            0.425107          1
4      Rajasthan      Udaipur  2015              50   Soybean  ...             490         277934     25.279857            0.204970          0

[5 rows x 13 columns]
```

## Step 6 — run the full orchestrator (master script)

If you want to run everything in one go (but you said one-step at a time — this is optional):
```
python master_pipeline.py
```

It should call generate → data pipeline → train → predict in sequence. Use this only after verifying each previous step succeeded.
