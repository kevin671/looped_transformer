#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc
conda activate loop_tf
mkdir -p /work/gg45/g45004/looped_transformer/result_folder/Figures/

export WANDB_CONFIG_DIR="/work/gg45/g45004/looped_transformer/tmp"

python jupyter_notebooks/Figures_Distillated_LR.py

# python jupyter_notebooks/Figures_LR.py

# python jupyter_notebooks/Figures_Sparse_LR.py

# python jupyter_notebooks/Figures_DT.py

# python jupyter_notebooks/Figures_ReLU2NN.py