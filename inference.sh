#!/bin/bash
python inference.py \
--config_path /fs/nexus-scratch/tuxunlu/git/event-based-deraining/lightning_logs/20251206-03-03-40-FourierMamba2D/version_0/hparams.yaml \
--test_checkpoint /fs/nexus-scratch/tuxunlu/git/event-based-deraining/lightning_logs/20251206-03-03-40-FourierMamba2D/version_0/checkpoints/best-epoch=049-val_loss_epoch=1.77958.ckpt \
--no-save