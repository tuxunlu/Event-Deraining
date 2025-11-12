import importlib
from dataclasses import asdict

import torch
import lightning.pytorch as pl
import torchvision
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)


class ModelInterface(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        training_cfg: TrainingConfig,
        data_cfg: DataConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg

        self.save_hyperparameters(
            {
                "model": asdict(self.model_cfg),
                "optimizer": asdict(self.optimizer_cfg),
                "scheduler": asdict(self.scheduler_cfg),
                "training": asdict(self.training_cfg),
                "data": asdict(self.data_cfg),
            }
        )

        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()

    def forward(self, x):
        return self.model(x)

    # For all these hook functions like on_XXX_<epoch|batch>_<end|start>(),
    # check document: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
    # Epoch level training logging
    def on_train_epoch_end(self):
        # Visualize some train_derained images at the end of each epoch
        train_batch = next(iter(self.trainer.datamodule.train_dataloader()))

    # Caution: self.model.train() is invoked
    # For logging, check document: https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging
    # Important clarification for new users:
    # 1. If on_step=True, a _step suffix will be concatenated to metric name. Same for on_epoch, but epoch-level metrics will be automatically averaged using batch_size as weight.
    # 2. If enable_graph=True, .detach() will not be invoked on the value of metric. Could introduce potential error.
    # 3. If sync_dist=True, logger will average metrics across devices. This introduces additional communication overhead, and not suggested for large metric tensors.
    # We can also define customized metrics aggregator for incremental step-level aggregation(to be merged into epoch-level metrics).
    def training_step(self, batch, batch_idx):
        train_gt_fft, train_merge = batch['raw'], batch['merge']
        train_derained = self(train_merge)
        train_loss = self.loss_function(train_derained, train_gt_fft, 'train')

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=train_merge.shape[0])

        train_step_output = {
            'loss': train_loss,
        }

        return train_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        val_gt_fft, val_merge = batch['raw'], batch['merge']
        val_derained = self(val_merge)
        val_loss = self.loss_function(val_derained, val_gt_fft, 'val')

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=val_merge.shape[0])

        # Log val_derained images for visualization
        if batch_idx == 0:
            imgs = val_derained.detach().cpu()
            imgs = imgs.squeeze(0).unsqueeze(1)  
            grid = torchvision.utils.make_grid(imgs, nrow=5, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_derained_images', grid, self.current_epoch)

            # Log val_gt_fft images for visualization
            gt_imgs = torch.fft.ifft2(val_gt_fft).real
            gt_imgs = gt_imgs.detach().cpu()
            gt_imgs = gt_imgs.squeeze(0).unsqueeze(1)  
            gt_grid = torchvision.utils.make_grid(gt_imgs, nrow=5, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_gt_images', gt_grid, self.current_epoch)

        val_step_output = {
            'loss': val_loss,
        }

        return val_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        test_gt_fft, test_merge = batch['raw'], batch['merge']
        test_derained = self(test_merge)
        test_loss = self.loss_function(test_derained, test_gt_fft, 'test')

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=test_merge.shape[0])

        test_step_output = {
            'loss': test_loss,
        }

        return test_step_output

    def configure_optimizers(self):
        # https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_cfg.name)
        except AttributeError as exc:
            raise ValueError(f"Invalid optimizer: OPTIMIZER.{self.optimizer_cfg.name}") from exc

        optimizer_arguments = dict(self.optimizer_cfg.arguments or {})
        optimizer_instance = optimizer_class(params=self.model.parameters(), **optimizer_arguments)

        learning_rate_scheduler_cfg = self.scheduler_cfg.learning_rate
        if not learning_rate_scheduler_cfg.enabled:
            return [optimizer_instance]

        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_cfg.name)
        except AttributeError as exc:
            raise ValueError(
                f"Invalid learning rate scheduler: SCHEDULER.learning_rate.{learning_rate_scheduler_cfg.name}."
            ) from exc

        scheduler_arguments = dict(learning_rate_scheduler_cfg.arguments or {})
        scheduler_instance = scheduler_class(optimizer=optimizer_instance, **scheduler_arguments)

        return [optimizer_instance], [scheduler_instance]

    def __configure_loss(self):
        def loss_func(derained, gt_fft, stage: str):
            # L1 loss between derained and gt
            gt = torch.fft.ifft2(gt_fft).real
            spatial_L1_loss = torch.nn.functional.l1_loss(derained, gt)
            self.log(f'{stage}_spatial_L1_loss', spatial_L1_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # L1 loss between amplitude spectrum of derained and gt
            derained_fft = torch.fft.fft2(derained)
            derained_fft_amp = torch.abs(derained_fft)
            gt_fft_amp = torch.abs(gt_fft)
            FFT_amp_L1_loss = 0.01*torch.nn.functional.l1_loss(derained_fft_amp, gt_fft_amp)
            self.log(f'{stage}_FFT_amp_L1_loss', FFT_amp_L1_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # L1 loss between phase spectrum of derained and gt
            derained_fft_phase = torch.angle(derained_fft)
            gt_fft_phase = torch.angle(gt_fft)
            FFT_phase_L1_loss = 10*torch.nn.functional.l1_loss(derained_fft_phase, gt_fft_phase)
            self.log(f'{stage}_FFT_phase_L1_loss', FFT_phase_L1_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            final_loss = spatial_L1_loss + FFT_amp_L1_loss + FFT_phase_L1_loss

            return final_loss

        return loss_func

    def __load_model(self):
        file_name = self.model_cfg.file_name
        class_name = self.model_cfg.class_name
        if class_name is None:
            raise ValueError("MODEL.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("MODEL.file_name must be specified in the configuration.")
        try:
            model_class = getattr(importlib.import_module('model.' + file_name, package=__package__), class_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {file_name}.{class_name}!')

        model_init_kwargs = asdict(self.model_cfg)
        model_init_kwargs.pop("class_name", None)
        model_init_kwargs.pop("file_name", None)

        model = model_class(**model_init_kwargs)
        if self.training_cfg.use_compile:
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                raise RuntimeError("torch.compile requested but not available in this torch build")
            model = compile_fn(model)
        return model
