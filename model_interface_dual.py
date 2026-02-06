import importlib
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchvision
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)

# --- Helper Functions & Classes ---

def phase_loss_complex(pred_fft: torch.Tensor,
                       gt_fft: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    # Normalize to unit magnitude
    pred_unit = pred_fft / (pred_fft.abs() + eps)
    gt_unit   = gt_fft   / (gt_fft.abs() + eps)

    # cos of phase difference: Re(pred * conj(gt))
    cos_dphi = (pred_unit * gt_unit.conj()).real
    cos_dphi = cos_dphi.clamp(-1.0 + eps, 1.0 - eps)

    # 1 - cos(Δθ) ∈ [0,2]
    return (1.0 - cos_dphi).mean()

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Robust L1)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        return torch.mean(loss)

# --- Main Interface ---

class ModelInterfaceDual(pl.LightningModule):
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

    def unpack_model_output(self, outputs):
        """
        Safely unpacks model output to handle torch.compile flattening.
        """
        # Case 1: Standard Tuple (bg, rain, [masks]) -> Length 3
        if isinstance(outputs, tuple) and len(outputs) == 3:
            if isinstance(outputs[2], list):
                return outputs[0], outputs[1], outputs[2]
            
        # Case 2: Compiled/Flattened Tuple -> Length >= 3
        # Assuming structure: bg, rain, mask1, mask2, ...
        if isinstance(outputs, (list, tuple)):
            if len(outputs) >= 2:
                pred_bg = outputs[0]
                pred_rain = outputs[1]
                # Collect all remaining items as masks
                masks = list(outputs[2:]) if len(outputs) > 2 else []
                return pred_bg, pred_rain, masks
            
        # Case 3: Inference / Single Output
        if isinstance(outputs, torch.Tensor):
            return outputs, None, []
            
        # Fallback
        return outputs[0], None, []

    def training_step(self, batch, batch_idx):
        train_gt, train_merge = batch['raw'], batch['merge']
        
        raw_outputs = self.model(train_merge)
        pred_bg, pred_rain, masks = self.unpack_model_output(raw_outputs)
        
        train_loss = self.loss_function(pred_bg, pred_rain, masks, train_gt, train_merge, 'train')

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=train_merge.shape[0])

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        val_gt, val_merge = batch['raw'], batch['merge']
        
        raw_outputs = self.model(val_merge)
        pred_bg, pred_rain, masks = self.unpack_model_output(raw_outputs)
        
        val_loss = self.loss_function(pred_bg, pred_rain, masks, val_gt, val_merge, 'val')

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=val_merge.shape[0])

        # Visualization
        if batch_idx == 0:
            # 1. Background Prediction (Derained)
            imgs_bg = pred_bg.detach().cpu()
            grid_bg = torchvision.utils.make_grid(imgs_bg, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_pred_bg', grid_bg, self.current_epoch)

            # 2. Ground Truth
            imgs_gt = val_gt.detach().cpu()
            grid_gt = torchvision.utils.make_grid(imgs_gt, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_gt', grid_gt, self.current_epoch)
            
            # 3. Input (Rainy)
            imgs_in = val_merge.detach().cpu()
            grid_in = torchvision.utils.make_grid(imgs_in, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_input', grid_in, self.current_epoch)

            # 4. Rain Stream (if available)
            if pred_rain is not None:
                imgs_rain = pred_rain.detach().cpu()
                # make_grid handles single channel automatically
                grid_rain = torchvision.utils.make_grid(imgs_rain, nrow=4, normalize=True, scale_each=True)
                self.logger.experiment.add_image('val_pred_rain', grid_rain, self.current_epoch)

        return {'loss': val_loss}

    def test_step(self, batch, batch_idx):
        test_gt, test_merge = batch['raw'], batch['merge']
        
        raw_outputs = self.model(test_merge)
        pred_bg, pred_rain, masks = self.unpack_model_output(raw_outputs)
        
        test_loss = self.loss_function(pred_bg, pred_rain, masks, test_gt, test_merge, 'test')
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=test_merge.shape[0])
        
        return {'loss': test_loss}

    def configure_optimizers(self):
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
        self.w_spatial = 10.0
        self.w_amp = 5.0
        self.w_phase = 5.0
        self.w_mask_reg = 0.1
        self.w_rain_aux = 1.0 

        self.char_crit = CharbonnierLoss()

        def loss_func(pred_bg, pred_rain, masks, gt, rainy, stage: str):
            # 1. BG Losses
            bg_spatial_loss = self.w_spatial * F.l1_loss(pred_bg, gt)
            self.log(f'{stage}_bg_spatial_loss', bg_spatial_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            bg_pred_fft = torch.fft.rfft2(pred_bg, dim=(-2, -1))
            bg_gt_fft = torch.fft.rfft2(gt, dim=(-2, -1))

            bg_amp_pred = torch.abs(bg_pred_fft) + 1e-8
            bg_amp_gt   = torch.abs(bg_gt_fft) + 1e-8
            bg_amp_loss = self.w_amp * F.l1_loss(torch.log(bg_amp_pred), torch.log(bg_amp_gt))
            self.log(f'{stage}_bg_fft_amp_loss', bg_amp_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            bg_phase_l = self.w_phase * phase_loss_complex(bg_pred_fft, bg_gt_fft)
            self.log(f'{stage}_bg_fft_phase_loss', bg_phase_l, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            # 2. Rain Losses (Only if rain head is active)
            rain_loss_total = 0.0
            if pred_rain is not None:
                gt_rain = (rainy - gt).detach()

                rain_spatial_loss = self.w_spatial * F.l1_loss(pred_rain, gt_rain)
                self.log(f'{stage}_rain_spatial_loss', rain_spatial_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

                rain_pred_fft = torch.fft.rfft2(pred_rain, dim=(-2, -1))
                rain_gt_fft   = torch.fft.rfft2(gt_rain, dim=(-2, -1))

                rain_amp_pred = torch.abs(rain_pred_fft) + 1e-8
                rain_amp_gt   = torch.abs(rain_gt_fft) + 1e-8
                rain_amp_loss = self.w_amp * F.l1_loss(torch.log(rain_amp_pred), torch.log(rain_amp_gt))
                
                rain_phase_l = self.w_phase * phase_loss_complex(rain_pred_fft, rain_gt_fft)
                
                # Combined Rain Stream Loss
                rain_loss_total = rain_spatial_loss + rain_amp_loss + rain_phase_l

            # 3. Region/Identity Losses
            bg_region_mask = (rainy - gt).abs() < 0.1 
            bg_diff = (pred_bg - gt).abs()
            bg_identity_loss = (bg_region_mask * bg_diff).sum() / (bg_region_mask.sum() + 1e-8)
            self.log(f'{stage}_bg_identity_loss', bg_identity_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            rain_region_mask = ~bg_region_mask
            rain_region_loss = (rain_region_mask * bg_diff).sum() / (rain_region_mask.sum() + 1e-8)
            self.log(f'{stage}_rain_region_loss', rain_region_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            # 4. Mask Regularization
            loss_mask_reg = 0
            if masks is not None:
                for m in masks:
                    # Ensure m is a tensor before calculation
                    if isinstance(m, torch.Tensor):
                        loss_mask_reg += torch.mean(m * (1 - m))
                self.log(f'{stage}_mask_reg_loss', loss_mask_reg, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            final_loss = bg_spatial_loss + bg_amp_loss + bg_phase_l + \
                         rain_loss_total + \
                         bg_identity_loss + rain_region_loss
                        #  (self.w_mask_reg * loss_mask_reg)
            
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