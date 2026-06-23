#!/bin/bash
python inference.py \
  --config_path "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/lightning_logs/20260225-00-22-10-DynamicFourierFilterNetFDConv_2D/version_0/hparams.yaml" \
  --test_checkpoint "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/lightning_logs/20260225-00-22-10-DynamicFourierFilterNetFDConv_2D/version_0/checkpoints/best-epoch=049-val_loss_epoch=2.21635.ckpt" \
  # --no-save