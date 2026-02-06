#!/bin/bash
python inference_dual.py \
--config_path /fs/nexus-scratch/tuxunlu/git/event-based-deraining/lightning_logs/20251210-02-17-23-DualStreamFourierMamba_bg+rain_losses/version_0/hparams.yaml \
--test_checkpoint /fs/nexus-scratch/tuxunlu/git/event-based-deraining/lightning_logs/20251210-02-17-23-DualStreamFourierMamba_bg+rain_losses/version_0/checkpoints/best-epoch=049-val_loss_epoch=5.71666.ckpt \
--no-save