
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

# Decision Tree ########################################################################################################
b=64
T=15
#python scripts/train.py --config configs/decision_tree/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "DT_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu


# Distillate to 32 steps
b_teacher=64
b_student=32
python scripts/distillate.py --config configs/decision_tree/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/decision_tree_loop/0609110718-DT_loop_L1_ends{64}_T{15}-4c59/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu

v
# Distillate to 16 steps
b_teacher=32
b_student=16
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu

# Distillate to 8 steps
b_teacher=16
b_student=8
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu

# Distillate to 4 steps
b_teacher=8
b_student=4
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu

# Distillate to 2 steps
b_teacher=4
b_student=2
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu

# Distillate to 1 steps
b_teacher=2
b_student=1
python scripts/distillate.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Distillate_DT_loop_L1_ends{$b_student}" \
    --gpu.n_gpu $n_gpu
COMMENTOUT