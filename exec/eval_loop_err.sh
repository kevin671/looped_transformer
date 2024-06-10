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
mkdir -p /work/gg45/g45004/looped_transformer/result_folder/Figures/

export WANDB_CONFIG_DIR="/work/gg45/g45004/looped_transformer/tmp"

python jupyter_notebooks/eval_loop_err.py