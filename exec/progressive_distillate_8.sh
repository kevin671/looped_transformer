
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
b=32
T=15
#python scripts/train.py --config configs/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "LR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu\

b_teacher=32
b_student=8
python scripts/progressive_distillation.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --progressive_distillation.teacher_n_loops $b_teacher\
    --progressive_distillation.student_n_loops $b_student \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Progressive_Distillation_LR_loop_teacher{$b_teacher}_student{$b_student}" \
    --gpu.n_gpu $n_gpu


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

b_teacher=64
b_student=4
#python scripts/block_distillate.py --config configs/decision_tree/base_loop.yaml \
#    --model.n_layer 1 \
#    --progressive_distillation.teacher_n_loops $b_teacher\
#    --progressive_distillation.student_n_loops $b_student \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b_teacher \
#    --training.n_loop_window $T \
#    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/decision_tree_loop/0609110718-DT_loop_L1_ends{64}_T{15}-4c59/state.pt" \
#    --wandb.name "Distillation_DT_loop_teacher{$b_teacher}_student{$b_student}" \
#    --gpu.n_gpu $n_gpu
