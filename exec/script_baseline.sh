#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM -m b
#PJM -m e
#PJM --fs /work

source /work/gg45/g45004/.bashrc
conda activate loop_tf

export WANDB_CONFIG_DIR="/work/gg45/g45004/looped_transformer/tmp"

source /work/gg45/g45004/looped_transformer/exec/env_variables.sh

n_gpu=0

# Linear Regression
#python scripts/train.py --config configs/base.yaml \
#    --wandb.name "LR_baseline" \
#    --gpu.n_gpu $n_gpu

## Sparse LR
python scripts/train.py --config configs/sparse_LR/base.yaml \
    --wandb.name "SparseLR_baseline" \
    --gpu.n_gpu $n_gpu
#
## Decision Tree
python scripts/train.py --config configs/decision_tree/base.yaml \
    --wandb.name "DT_baseline" \
    --gpu.n_gpu $n_gpu

## ReLU 2NN
python scripts/train.py --config configs/relu_2nn_regression/base.yaml \
    --wandb.name "ReLU2NN_baseline" \
    --gpu.n_gpu $n_gpu
