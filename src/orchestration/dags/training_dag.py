# Airflow DAG: scheduled retraining pipeline
# runs every Sunday at midnight, retrains the LSTM autoencoder
# on the latest data in the feature store
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# default args applied to every task in the DAG
default_args = {
    "owner":            "nabeegh",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# paths inside the Airflow container
# these match the volume mounts in docker-compose.yml
PROJECT_ROOT = "/opt/airflow"
DATA_DIR     = f"{PROJECT_ROOT}/data/processed"
SRC_DIR      = f"{PROJECT_ROOT}/src"


def log_training_start():
    """Log that the training run is starting — useful for audit trail."""
    print(f"Training DAG started at {datetime.utcnow().isoformat()}")
    print(f"Reading features from: {DATA_DIR}/features.parquet")


def log_training_complete():
    """Log completion and print key file paths for downstream tasks."""
    print(f"Training DAG completed at {datetime.utcnow().isoformat()}")
    print(f"Model saved to:  {DATA_DIR}/model/lstm_autoencoder.pt")
    print(f"Config saved to: {DATA_DIR}/model/model_config.json")


with DAG(
    dag_id="lstm_retraining_pipeline",
    description="Weekly retraining of the LSTM autoencoder on the NAB feature store",
    default_args=default_args,
    schedule_interval="0 0 * * 0",   # every Sunday at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=["training", "lstm", "anomaly-detection"],
) as dag:

    task_log_start = PythonOperator(
        task_id="log_training_start",
        python_callable=log_training_start,
    )

    task_validate_features = BashOperator(
        task_id="validate_feature_store",
        bash_command=(
            f"python -c \""
            f"import pandas as pd; "
            f"df = pd.read_parquet('{DATA_DIR}/features.parquet'); "
            f"assert len(df) > 0, 'Feature store is empty'; "
            f"print(f'Feature store OK: {{len(df):,}} rows'); "
            f"\""
        ),
    )

    task_retrain = BashOperator(
        task_id="retrain_lstm_autoencoder",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH={PROJECT_ROOT} "
            f"python {SRC_DIR}/training/train.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    task_run_drift = BashOperator(
        task_id="run_drift_check_after_training",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH={PROJECT_ROOT} "
            f"python {SRC_DIR}/monitoring/drift_report.py"
        ),
    )

    task_log_complete = PythonOperator(
        task_id="log_training_complete",
        python_callable=log_training_complete,
    )

    # task dependency chain
    (
        task_log_start
        >> task_validate_features
        >> task_retrain
        >> task_run_drift
        >> task_log_complete
    )