import hydra
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
    )


if __name__ == "__main__":
    pretraining()