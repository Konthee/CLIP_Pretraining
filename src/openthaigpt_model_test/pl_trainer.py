import torch

import pytorch_lightning as pl

from datasets import load_from_disk

from transformers import CLIPModel

from .constants import (
    INPUT_IDS_KEY,
    ATTENTION_MASK_KEY,
    PIXEL_VALUES_KEY,
    TORCH_FORMAT,
    TRAIN_LOSS_MONITOR,
    VAL_LOSS_MONITOR,
)


class CLIPModelPL(pl.LightningModule):
    def __init__(
        self,
        model_config=None,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(
            model_config.clip_pretrained_path
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
    ):
        return self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=4e-4,
            weight_decay=0.2,
            betas=[0.9, 0.98],
            eps=1e-6,
        )
        optimizers_config = {
            "optimizer": optimizer,
            "monitor": VAL_LOSS_MONITOR,
        }
        return optimizers_config

    def training_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_KEY]
        attention_mask = batch[ATTENTION_MASK_KEY]
        pixel_values = batch[PIXEL_VALUES_KEY]

        output = self.forward(
            input_ids,
            attention_mask,
            pixel_values,
        )
        loss = output.loss

        self.log(TRAIN_LOSS_MONITOR, loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_KEY]
        attention_mask = batch[ATTENTION_MASK_KEY]
        pixel_values = batch[PIXEL_VALUES_KEY]

        output = self.forward(
            input_ids,
            attention_mask,
            pixel_values,
        )
        loss = output.loss

        self.log(VAL_LOSS_MONITOR, loss, prog_bar=True, sync_dist=True)


def clip_pretraining(
    model_config,
    trainer_config,
    dataset_config,
):
    """
    Args:
        model_config: Model configuration
        trainer_config: Trainer configuration
        dataset_config: Dataset configuration
    """

    hf_train_dataset = load_from_disk(
        dataset_config.training_dataset
    ).with_format(TORCH_FORMAT)
    train_dataloaders = torch.utils.data.DataLoader(
        hf_train_dataset['train'],
        batch_size=trainer_config.batch_size,
        num_workers=dataset_config.num_workers,
    )
    
    hf_validation_dataset = load_from_disk(
        dataset_config.validation_dataset
    ).with_format(TORCH_FORMAT)
    validation_dataloaders = torch.utils.data.DataLoader(
        hf_validation_dataset['train'],
        batch_size=trainer_config.batch_size,
        num_workers=dataset_config.num_workers,
    )
    model = CLIPModelPL(
        model_config=model_config,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=validation_dataloaders,
    )
