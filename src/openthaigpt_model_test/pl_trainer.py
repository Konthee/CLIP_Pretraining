import torch

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger

from torchsummary import summary

from optimum.bettertransformer import BetterTransformer

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
        optimizer_config='Adam',
        scheduler_config='None',
        trainer_config=None,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(
            model_config.clip_pretrained_path
        )
        self.save_hyperparameters(trainer_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        # Freeze the image encoder layer weights
        if trainer_config.freeze_image_encoder: 
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        if trainer_config.better_transformer :
            self.clip_model = BetterTransformer.transform(self.clip_model,keep_original_model=False)
        #sumary model
        summary(self.clip_model)

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
        # optimizer
        if self.optimizer_config.type == 'Adam' :
            optimizer_cls = torch.optim.Adam 
        elif self.optimizer_config.type == 'AdamW' :
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
            eps=self.optimizer_config.eps,
        )

        # scheduler
        if self.scheduler_config.type == 'None' :
            return {"optimizer": optimizer,"monitor": VAL_LOSS_MONITOR,}
        elif self.scheduler_config.type == 'cosine' :
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.T_max,
                eta_min=self.scheduler_config.eta_min
            )
            schedulers={
                'scheduler': cosine_scheduler,
                'interval': 'epoch',
                'monitor': VAL_LOSS_MONITOR
                }
        elif self.scheduler_config.type == 'steplr' :
            step_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.step_size,
                gamma=self.scheduler_config.gamma
            )
            schedulers={
                'scheduler': step_scheduler,
                'interval': 'epoch',
                'monitor': VAL_LOSS_MONITOR
                }
        
        optimizers_config = {
            "optimizer": optimizer,
            'lr_scheduler': schedulers,
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
    optimizer_config,
    scheduler_config,
    checkpointing_config,
    logging_config,
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
        hf_validation_dataset['val'],
        batch_size=trainer_config.batch_size,
        num_workers=dataset_config.num_workers,
    )
    model = CLIPModelPL(
        model_config=model_config,
        optimizer_config=optimizer_config, 
        scheduler_config=scheduler_config,
        trainer_config=trainer_config,
    )
    
    #callback
    callbacks =[]
    #Checkpoint
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpointing_config.dirpath,
            filename='{epoch}-{val_loss:.2f}',
            every_n_epochs = checkpointing_config.every_n_epochs,
            save_top_k=checkpointing_config.save_top_k,
            save_last=checkpointing_config.save_last,
            verbose=checkpointing_config.verbose,
            save_weights_only= checkpointing_config.save_weights_only,
            monitor=VAL_LOSS_MONITOR,
            mode=checkpointing_config.mode,  
        )
    )
    if trainer_config.early_stopping.use:
        callbacks.append(
            EarlyStopping(
                monitor=VAL_LOSS_MONITOR, 
                patience=trainer_config.early_stopping.patience,
                min_delta =  trainer_config.early_stopping.min_delta, 
                verbose=trainer_config.early_stopping.verbose,
                mode=trainer_config.early_stopping.mode,
            )
        )
    #logger
    loggers = []
    if logging_config.wandb.use: 
        loggers.append(
            WandbLogger(
                project=logging_config.wandb.project, 
                log_model=logging_config.wandb.log_model
            )
        )
    if logging_config.tensorboard.use:
        loggers.append(
            TensorBoardLogger(
                save_dir=logging_config.tensorboard.save_dir, 
                name=logging_config.tensorboard.name
                )
        )
    # resume_from_checkpoint
    resume_from_checkpoint=None
    if trainer_config.resume_from_checkpoint.use :
        resume_from_checkpoint=trainer_config.resume_from_checkpoint.checkpoint_path  

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        max_epochs=trainer_config.max_epochs,
        log_every_n_steps=trainer_config.max_epochs,
        precision=trainer_config.precision,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        callbacks=callbacks,
        logger = loggers if loggers else None ,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=validation_dataloaders,
        ckpt_path=resume_from_checkpoint,
    )
