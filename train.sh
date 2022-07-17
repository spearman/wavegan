#!/usr/bin/env bash

set -x

#python3 train_wavegan.py train ./train \
#  --wavegan_loss wgan \
python3 train.py train ./train \
  --wavegan_genr_pp \
  --data_dir ./data/drums/train \
  --data_first_slice \
  --data_pad_end \
  --data_fast_wav
  #--data_prefetch_gpu_num -1 \   # use this option when running on cpu

exit 0
