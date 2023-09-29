import hydra
from functools import partial
from datasets import load_dataset
from transformers import AutoProcessor
from openthaigpt_model_test.data_preprocessing import dataset_tokenization
import os
import shutil

@hydra.main(
    version_base=None,
    config_path="./dataset_preparation_configuration",
    config_name="default_config",
)
def prepare_dataset(cfg):
    dataset = load_dataset(cfg.dataset)
    processor = AutoProcessor.from_pretrained(cfg.processor)
    tokenized_dataset = dataset.map(
        partial(
            dataset_tokenization,
            processor=processor,
        ),
        batched=True,
        batch_size=cfg.tokenize_batch_size,
        remove_columns=dataset['train'].column_names,
        num_proc=cfg.num_proc,
    )
    tokenized_dataset.save_to_disk(cfg.save_dir)


if __name__ == "__main__":
    try : shutil.rmtree("Dataset")
    except : pass
    os.mkdir("Dataset")
    prepare_dataset()
