#!/bin/bash
python inference.py \
  --config_path "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/configs/config_dynamicfourierfilternet.yaml" \
  --test_checkpoint "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/lightning_logs/20260223-12-46-21-DynamicFourierFilterNet_2D/version_0/checkpoints/best-epoch=049-val_loss_epoch=2.26851.ckpt" \
  --no-save