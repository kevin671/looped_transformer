
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

b=32
T=15

b_teacher=32
b_student=32
python scripts/progressive_distillation.py --config configs/base_loop.yaml \
    --model.n_layer 1 \
    --progressive_distillation.teacher_n_loops $b_teacher\
    --progressive_distillation.student_n_loops $b_student \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b_teacher \
    --training.n_loop_window $T \
    --training.learning_rate 0.0001 \
    --model.pretrained_path "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop/0609110718-LR_loop_L1_ends{32}_T{15}-1338/state.pt" \
    --wandb.name "Debug_Progressive_Distillation_LR_loop_teacher{$b_teacher}_student{$b_student}" \
    --gpu.n_gpu $n_gpu
