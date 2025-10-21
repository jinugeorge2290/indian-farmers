from prefect import flow
from data_pipeline_prefect import data_pipeline_flow
from ml_pipeline_prefect import ml_pipeline_flow
from predict_pipeline_prefect import prediction_flow  # import your prediction flow

@flow(name="Master Farmer Pipeline")
def master_pipeline_flow(new_data_path: str = "farmer_suicide_large_realistic.csv"):
    # Step 1: Run data ingestion + preprocessing
    data_pipeline_flow()

    # Step 2: Run ML training + evaluation
    ml_pipeline_flow()

    # Step 3: Run predictions on new data
    prediction_flow(new_data_path=new_data_path)

#if __name__ == "__main__":
#    master_pipeline_flow()
