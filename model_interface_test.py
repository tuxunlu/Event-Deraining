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
        try:
            weights = torchvision.models.VGG19_Weights.DEFAULT
            vgg = torchvision.models.vgg19(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            vgg = torchvision.models.vgg19(pretrained=True)
            
        # Use features up to ReLU 3_4 or 4_4. 
        self.features = vgg.features[:35].eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()
        
        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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

class ModelInterfaceTest(pl.LightningModule):
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

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, x):
        """
        For generic inference, return the CLEAN (derained) output only.
        The underlying model returns (pred_clean, pred_rain).
        """
        pred_clean, _ = self.model(x)
        return pred_clean

    # -------------------------------
    # Training / Validation / Test
    # -------------------------------
    def training_step(self, batch, batch_idx):
        train_gt, train_merge = batch['raw'], batch['merge']

        # New model: returns pred_clean, pred_rain
        pred_clean, pred_rain = self.model(train_merge)

        train_loss = self.loss_function(
            pred_clean, pred_rain,
            train_gt, train_merge,
            stage='train'
        )

        self.log('train_loss', train_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True,
                 batch_size=train_merge.shape[0])

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_gt, val_merge = batch['raw'], batch['merge']
        pred_clean, pred_rain = self.model(val_merge)

        val_loss = self.loss_function(
            pred_clean, pred_rain,
            val_gt, val_merge,
            stage='val'
        )

        self.log('val_loss', val_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True,
                 batch_size=val_merge.shape[0])

        # Log some val images
        if batch_idx == 0:
            imgs = pred_clean.detach().cpu()
            grid = torchvision.utils.make_grid(
                imgs, nrow=4, normalize=True, scale_each=True
            )
            self.logger.experiment.add_image(
                'val_derained_images', grid, self.current_epoch
            )

            gt_imgs = val_gt.detach().cpu()
            gt_grid = torchvision.utils.make_grid(
                gt_imgs, nrow=4, normalize=True, scale_each=True
            )
            self.logger.experiment.add_image(
                'val_gt_images', gt_grid, self.current_epoch
            )

        return {"loss": val_loss}

    def test_step(self, batch, batch_idx):
        test_gt, test_merge = batch['raw'], batch['merge']
        pred_clean, pred_rain = self.model(test_merge)

        test_loss = self.loss_function(
            pred_clean, pred_rain,
            test_gt, test_merge,
            stage='test'
        )

        self.log('test_loss', test_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True,
                 batch_size=test_merge.shape[0])

        return {"loss": test_loss}

    # -------------------------------
    # Optimizer / Scheduler
    # -------------------------------
    def configure_optimizers(self):
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_cfg.name)
        except AttributeError as exc:
            raise ValueError(f"Invalid optimizer: OPTIMIZER.{self.optimizer_cfg.name}") from exc

        optimizer_arguments = dict(self.optimizer_cfg.arguments or {})
        optimizer_instance = optimizer_class(
            params=self.model.parameters(), **optimizer_arguments
        )

        learning_rate_scheduler_cfg = self.scheduler_cfg.learning_rate
        if not learning_rate_scheduler_cfg.enabled:
            return [optimizer_instance]

        try:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, learning_rate_scheduler_cfg.name
            )
        except AttributeError as exc:
            raise ValueError(
                f"Invalid learning rate scheduler: "
                f"SCHEDULER.learning_rate.{learning_rate_scheduler_cfg.name}."
            ) from exc

        scheduler_arguments = dict(learning_rate_scheduler_cfg.arguments or {})
        scheduler_instance = scheduler_class(
            optimizer=optimizer_instance, **scheduler_arguments
        )

        return [optimizer_instance], [scheduler_instance]

    # -------------------------------
    # Loss Configuration
    # -------------------------------
    def __configure_loss(self):
        # Instantiate losses
        self.char_crit = CharbonnierLoss()
        # self.perc_crit = PerceptualLoss()  # optionally use later

        # Hyper-parameters (you can tune these)
        self.bg_alpha = 10.0   # if you revive the exp(-α·diff) weighting
        self.bg_lambda = 0.5
        self.lambda_rain_layer = 1.0
        self.lambda_consistency = 1.0

        def loss_func(pred_clean, pred_rain, gt, rainy, stage: str):
            """
            pred_clean: predicted background (derained), (B, C, H, W)
            pred_rain:  predicted rain layer,            (B, C, H, W)
            gt:         ground-truth clean image,       (B, C, H, W)
            rainy:      rainy input image,              (B, C, H, W)
            stage:      'train' | 'val' | 'test' (for logging)
            """

            # --------------------------------------------------
            # 1. Spatial (pixel-wise) loss on CLEAN
            # --------------------------------------------------
            spatial_loss = 10.0 * torch.nn.functional.l1_loss(pred_clean, gt)
            self.log(f'{stage}_spatial_loss', spatial_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # --------------------------------------------------
            # 2. Frequency-domain losses (Amplitude + Phase) on CLEAN
            # --------------------------------------------------
            clean_fft = torch.fft.rfft2(pred_clean, dim=(-2, -1))
            gt_fft    = torch.fft.rfft2(gt,         dim=(-2, -1))

            amp_pred = torch.abs(clean_fft) + 1e-8
            amp_gt   = torch.abs(gt_fft)    + 1e-8

            # Log-amplitude L1
            amp_loss = 5.0 * torch.nn.functional.l1_loss(
                torch.log(amp_pred), torch.log(amp_gt)
            )
            self.log(f'{stage}_fft_amp_loss', amp_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # Phase loss
            phase_l = 5.0 * phase_loss_complex(clean_fft, gt_fft)
            self.log(f'{stage}_fft_phase_loss', phase_l,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # --------------------------------------------------
            # 3. Background / Rain-region losses on CLEAN
            #    - bg_mask: where rainy ≈ gt (little rain)
            #    - rain_mask: where rainy differs from gt (likely rain)
            # --------------------------------------------------
            diff_inp_gt = (rainy - gt).abs()
            bg_mask = diff_inp_gt < 0.1
            rain_mask = ~bg_mask

            # Background identity: in bg regions, pred_clean should match gt
            bg_diff = (pred_clean - gt).abs()
            bg_loss = (bg_mask * bg_diff).sum() / (bg_mask.sum() + 1e-8)
            self.log(f'{stage}_bg_identity_loss', bg_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # Rain-region loss: in rain regions, also compare to gt
            rain_diff = (pred_clean - gt).abs()
            rain_loss = (rain_mask * rain_diff).sum() / (rain_mask.sum() + 1e-8)
            self.log(f'{stage}_rain_region_loss', rain_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # --------------------------------------------------
            # 4. Rain-layer supervision (multi-task)
            #    rain_gt ≈ rainy - gt
            # --------------------------------------------------
            rain_gt = rainy - gt
            rain_layer_loss = 10 * self.lambda_rain_layer * torch.nn.functional.l1_loss(
                pred_rain, rain_gt
            )
            self.log(f'{stage}_rain_layer_loss', rain_layer_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # --------------------------------------------------
            # 5. Decomposition consistency:
            #    pred_clean + pred_rain should reconstruct rainy
            # --------------------------------------------------
            recon = pred_clean + pred_rain
            cons_loss = 10 * self.lambda_consistency * torch.nn.functional.l1_loss(
                recon, rainy
            )
            self.log(f'{stage}_decomp_consistency_loss', cons_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # --------------------------------------------------
            # Final loss (tune weights as needed)
            # --------------------------------------------------
            final_loss = (
                spatial_loss
                + amp_loss
                + phase_l
                # + bg_loss
                # + rain_loss
                + rain_layer_loss
                + cons_loss
            )

            return final_loss

        return loss_func

    # -------------------------------
    # Model loading
    # -------------------------------
    def __load_model(self):
        file_name = self.model_cfg.file_name
        class_name = self.model_cfg.class_name
        if class_name is None:
            raise ValueError("MODEL.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("MODEL.file_name must be specified in the configuration.")
        try:
            # e.g. model_cfg.file_name = "fourier_mamba_2d"
            #      model_cfg.class_name = "FourierMamba2D"
            module = importlib.import_module('model.' + file_name, package=__package__)
            model_class = getattr(module, class_name)
        except Exception:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {file_name}.{class_name}!'
            )

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
