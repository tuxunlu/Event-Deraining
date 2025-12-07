import importlib
from dataclasses import asdict

import torch
import torch.nn as nn
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

class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss to capture structural differences 
    that pixel-wise losses (L1/Charbonnier) might miss.
    """
    def __init__(self):
        super().__init__()
        # Load VGG19 features (pretrained)
        # We use .eval() to ensure layers like Dropout/BatchNorm behave deterministically
        try:
            weights = torchvision.models.VGG19_Weights.DEFAULT
            vgg = torchvision.models.vgg19(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            vgg = torchvision.models.vgg19(pretrained=True)
            
        # Use features up to ReLU 3_4 or 4_4. 
        # Slicing up to 35 captures mid-to-high level features.
        self.features = vgg.features[:35].eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()
        
        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Handle grayscale inputs (repeat to 3 channels for VGG)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
            
        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # Extract features
        x_feat = self.features(x)
        y_feat = self.features(y)
        
        return self.l1(x_feat, y_feat)

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

    def training_step(self, batch, batch_idx):
        train_gt, train_merge = batch['raw'], batch['merge']
        train_derained = self(train_merge)
        train_loss = self.loss_function(train_derained, train_gt, 'train')

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=train_merge.shape[0])

        train_step_output = {
            'loss': train_loss,
        }
        return train_step_output

    def validation_step(self, batch, batch_idx):
        val_gt, val_merge = batch['raw'], batch['merge']
        val_derained = self(val_merge)
        val_loss = self.loss_function(val_derained, val_gt, 'val')

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=val_merge.shape[0])

        # Log a random val_derained images for visualization
        if batch_idx == 0:
            imgs = val_derained.detach().cpu()
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_derained_images', grid, self.current_epoch)

            gt_imgs = val_gt.detach().cpu()
            gt_grid = torchvision.utils.make_grid(gt_imgs, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_gt_images', gt_grid, self.current_epoch)

        val_step_output = {
            'loss': val_loss,
        }
        return val_step_output

    def test_step(self, batch, batch_idx):
        test_gt, test_merge = batch['raw'], batch['merge']
        test_derained = self(test_merge)
        test_loss = self.loss_function(test_derained, test_gt, 'test')

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=test_merge.shape[0])

        test_step_output = {
            'loss': test_loss,
        }
        return test_step_output

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
        # Instantiate Losses as attributes of self so Lightning moves them to the correct device
        self.char_crit = CharbonnierLoss()
        
        def loss_func(derained, gt, stage: str):
            # 1. Spatial Loss: Charbonnier Loss
            spatial_loss = 10 * torch.nn.functional.l1_loss(derained, gt)
            self.log(f'{stage}_spatial_loss', spatial_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # 2. Frequency Losses
            derained_fft = torch.fft.rfft2(derained, dim=(-2, -1))
            gt_fft = torch.fft.rfft2(gt, dim=(-2, -1))
            
            # Log-Amplitude Loss
            amp_pred = torch.abs(derained_fft) + 1e-8
            amp_gt = torch.abs(gt_fft) + 1e-8
            amp_loss = 5 * torch.nn.functional.l1_loss(torch.log(amp_pred), torch.log(amp_gt))
            self.log(f'{stage}_fft_amp_loss', amp_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # L1 loss between phase spectrum of derained and gt
            # derained_fft_phase = torch.angle(derained_fft)
            # gt_fft_phase = torch.angle(gt_fft)
            # FFT_phase_L1_loss = 10*torch.nn.functional.l1_loss(derained_fft_phase, gt_fft_phase)
            phase_loss = 5 * phase_loss_complex(derained_fft, gt_fft)
            self.log(f'{stage}_FFT_phase_loss', phase_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            final_loss = spatial_loss + amp_loss + phase_loss

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
