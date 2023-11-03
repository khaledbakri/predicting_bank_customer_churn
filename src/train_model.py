from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
import wandb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold

def train_model(model_config, project_name):
    transformers = [("onehot1",  OneHotEncoder(), ['country']),
                    ("onehot2",  OneHotEncoder(), ['gender'])]
    ct = ColumnTransformer(transformers, remainder='passthrough')
    clf = RandomForestClassifier(n_estimators=model_config['n_estimators'],
                                max_depth=model_config['max_depth'],
                                min_samples_split=model_config['min_samples_split'],
                                min_samples_leaf=model_config['min_samples_leaf'],)

    steps = [('columns_transformer', ct),
             ('classifier', clf)]
    pipe = Pipeline(steps)
    with wandb.init(project=project_name) as run:
        table = run.use_artifact('khaleddbakri/predicting_bank_customer_churn/bank_customer_churn_prediction_splitted:latest')
    
    df = table.get('bank_customer_churn_prediction_splitted').get_dataframe()

    train_dataset = df[df['stage']=='train']
    X_train = train_dataset.drop(['churn', 'stage'], axis = 1)
    y_train = train_dataset['churn']

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['roc_auc'])
    
    roc_auc_avg = np.mean(cv_results['test_roc_auc'])
    
    with wandb.init(project=project_name, job_type='train_model') as run:
        wandb.summary["roc_auc_val"] = roc_auc_avg

    pipe.fit(X_train, y_train)

    return pipe

    