import hydra
from omegaconf import DictConfig
from upload_raw_data import upload_raw_data
from split_data import split_data
from train_model import train_model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_pipeline(cfg : DictConfig) -> None:
    for step in cfg['pipeline_steps']:
        if step == 'upload_raw_data':
            upload_raw_data(cfg['data_path'], 
                            cfg['wandb']['project_name'])

        if step == 'split_data':
            split_data(cfg['test_size'], 
                       cfg['wandb']['project_name'])

        if step == 'train_model':
            pipe = train_model(cfg['model'],
                        cfg['wandb']['project_name'])

if __name__ == "__main__":
    run_pipeline()