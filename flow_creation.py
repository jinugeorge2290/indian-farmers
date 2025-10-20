from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta
from data_pipeline_prefect import data_pipeline_flow
from ml_pipeline_prefect import ml_pipeline_flow

# -----------------------------
# Data Pipeline Deployment
# -----------------------------
data_deploy = Deployment.build_from_flow(
    flow=data_pipeline_flow,
    name="Data Pipeline Cloud",
    schedule=IntervalSchedule(interval=timedelta(hours=2))  # every 2 hours
)
data_deploy.apply()  # registers with Prefect Cloud

# -----------------------------
# ML Pipeline Deployment
# -----------------------------
ml_deploy = Deployment.build_from_flow(
    flow=ml_pipeline_flow,
    name="ML Pipeline Cloud",
    schedule=IntervalSchedule(interval=timedelta(hours=6))  # every 6 hours
)
ml_deploy.apply()  # registers with Prefect Cloud
