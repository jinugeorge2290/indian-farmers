from prefect import flow
from data_pipeline_prefect import data_pipeline_flow
from ml_pipeline_prefect import ml_pipeline_flow

@flow(name="Master Farmer Pipeline")
def master_pipeline_flow():
    data_pipeline_flow()   # Run data ingestion + preprocessing
    ml_pipeline_flow()     # Then run ML training + evaluation

#if __name__ == "__main__":
#    master_pipeline_flow()
