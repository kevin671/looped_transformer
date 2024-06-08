
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

# Linear Regression  ###################################################################################################
b=30
T=15
#python scripts/train.py --config configs/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "LR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu


# Distillate Linear Regression  ############################################################################################
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0525163553-LR_loop_L1_ends{30}_T{15}-f0a2/state.pt" \
    --n_points 1000 \
    --teacher_n_loops $b \
    --training.n_loop_window $T \
    --wandb.name "Distillate_LR_loop_L1_ends{$b}_T{$T}" \
    --gpu.n_gpu $n_gpu