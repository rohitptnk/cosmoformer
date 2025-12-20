import mlflow
from pathlib import Path

def setup_mlflow(
    experiment_name: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name=experiment_name)

    return mlflow.start_run(run_name=run_name)

def log_params_flat(params: dict, prefix: str | None = None):
    for k,v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        mlflow.log_param(key, v)

def log_artifacts(path: str | Path, artifact_path: str | None = None):
    mlflow.log_artifact(str(path), artifact_path=artifact_path)