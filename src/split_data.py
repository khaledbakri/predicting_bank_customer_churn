import wandb
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(test_size, project_name):
    with wandb.init(project=project_name) as run:
        table = run.use_artifact('khaleddbakri/predicting_bank_customer_churn/bank_customer_churn_prediction:latest')
    
    df = table.get('bank_customer_churn_prediction').get_dataframe()
    df = df.drop('customer_id', axis=1)
    target = df['churn']

    num_rows = df.shape[0]
    indices = np.arange(num_rows)

    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=target)

    df.loc[train_idx, 'stage'] = 'train'
    df.loc[test_idx, 'stage'] = 'test'

    with wandb.init(project=project_name, job_type='upload_splitted_data') as run:
        table = wandb.Table(data=df)
        table_artifact = wandb.Artifact('bank_customer_churn_prediction_splitted', type='splitted_data')
        table_artifact.add(table, 'bank_customer_churn_prediction_splitted')
        run.log_artifact(table_artifact)

