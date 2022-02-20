#!/usr/bin/env bash

set -x

#python train_wavegan.py train ./train \
#  --wavegan_loss wgan \
python train.py train ./train \
  --data_prefetch_gpu_num -1 \
  --data_dir ./data/drums/train \
  --data_first_slice \
  --data_pad_end \
  --data_fast_wav

exit 0
