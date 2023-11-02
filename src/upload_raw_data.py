import pandas as pd
import wandb

def upload_raw_data(data_path, project_name):
    with wandb.init(project=project_name, job_type='upload_raw_data') as run:
        df = pd.read_csv(data_path)
        table = wandb.Table(data=df)
        table_artifact = wandb.Artifact('bank_customer_churn_prediction', type='raw_data')
        table_artifact.add(table, 'bank_customer_churn_prediction')
        run.log_artifact(table_artifact)