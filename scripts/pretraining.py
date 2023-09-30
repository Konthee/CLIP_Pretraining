import hydra
import os
from dataset_preparation import *
from openthaigpt_model_test.pl_trainer import clip_pretraining


@hydra.main(
    version_base=None,
    config_path="./pretraining_configuration",
    config_name="default_config",
)
def pretraining(cfg):
    clip_pretraining(
        model_config=cfg.model,
        trainer_config=cfg.trainer,
        dataset_config=cfg.dataset,
        optimizer_config=cfg.optimizer,
        scheduler_config =cfg.scheduler,
        checkpointing_config=cfg.checkpointing,
        logging_config=cfg.logging,
    )

if __name__ == "__main__":
    print(os.get_exec_path)
    os.system("python scripts/dataset_preparation.py")
    # pretraining()