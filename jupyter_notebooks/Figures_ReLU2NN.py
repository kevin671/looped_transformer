import os
import json
import numpy as np
from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

import sys
sys.path.append('/work/gg45/g45004/looped_transformer/scripts')
from nano_gpt import GPT2Model, GPT2Config

from utils import aggregate_metrics, get_model, eval_unlooped_model, eval_looped_model


class Relu2nnRegression():
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, hidden_layer_size=100, non_sparse=100):
        super(Relu2nnRegression, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.b_size = batch_size
        self.n_points = n_points

        W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size, device=device)
        W2 = torch.randn(self.b_size, hidden_layer_size, 1, device=device)

        if non_sparse < hidden_layer_size:
            import random
            non_sparse_mask = torch.zeros(hidden_layer_size, device=device)
            non_sparse_indices = random.sample(range(hidden_layer_size), non_sparse)
            non_sparse_mask[non_sparse_indices] = 1
            self.W1 = W1 * non_sparse_mask[None, None, :]
            self.W2 = W2 * non_sparse_mask[None, :, None]
        else:
            self.W1 = W1
            self.W2 = W2

        self.xs = torch.randn(batch_size, n_points, n_dims, device=device)  # [B, n, d]
        self.xs[..., n_dims_truncated:] = 0

        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs_b):
        W1 = self.W1
        W2 = self.W2
        # Renormalize to Linear Regression Scale
        ys_b_nn = (F.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        # ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

if __name__ == "__main__":
    fig_hparam = {
        'figsize': (8, 5),
        'labelsize': 28,
        'ticksize': 20,
        'linewidth': 5,
        'fontsize': 15,
        'titlesize': 20,
        'markersize': 15
    }

    # font specification
    fontdict = {'family': 'serif',
            'size': fig_hparam['fontsize'],
            }

    device = torch.device('cuda:0')

    sample_size = 1280
    batch_size = 64
    n_points = 101
    n_dims_truncated = 20
    n_dims = 20


    torch.manual_seed(456)
    real_task = Relu2nnRegression(sample_size, n_points, n_dims, n_dims_truncated, device)
    xs, ys = real_task.xs, real_task.ys

    result_dir = '/work/gg45/g45004/looped_transformer/results2/relu_2nn_baseline'
    run_id = '0527041715-ReLU2NN_baseline-bfd6'

    from models import TransformerModel

    n_positions = 101
    n_embd = 256
    n_layer = 12
    n_head = 8

    model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head)
    step = -1
    model = get_model(model, result_dir, run_id, step)
    model = model.to(device)

    err, y_pred_total = eval_unlooped_model(model, xs, ys)

    result_errs = {}
    result_errs['Transformer'] = err

    from models import TransformerModelLooped

    result_dir = '/work/gg45/g45004/looped_transformer/results2/relu_2nn_loop'
    run_id = '0527161747-relu2nn_loop_L1_ends{12}_T{5}-b3f2'

    n_positions = 101
    n_embd = 256
    n_head = 8
    T = 500
    n_layer = 1

    model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)
    step = -1
    model = get_model(model, result_dir, run_id, step)
    model = model.to(device)
        
    err, loop_err = eval_looped_model(model, xs, ys, loop_max=T)

    result_errs['Looped Transformer'] = err


    from utils import get_relevant_baselines
    from utils import LeastSquaresModel, NNModel, AveragingModel, GDModel, NeuralNetwork
    # baselines = get_relevant_baselines("relu_2nn_regression")
    baselines = [
        (LeastSquaresModel, {}),
        (NNModel, {"n_neighbors": 3}),
        (AveragingModel, {}),
    ]
    gd_baselines = [
        (GDModel, {
            "model_class": NeuralNetwork,
            "model_class_args": {
                "in_size": 20,
                "hidden_size": 100,
                "out_size": 1,
            },
            "opt_alg": "adam",
            "batch_size": 10,
            "lr": 5e-3,
            "num_steps": 1000,
        },)
    ]

    baselines += gd_baselines          
    baseline_models = [model_cls(**kwargs) for model_cls, kwargs in baselines]
    # baseline_errs = {}
    for baseline_model in baseline_models:
        if "gd_model" in baseline_model.name:
            y_pred = baseline_model(xs, ys, device)
        else:
            y_pred = baseline_model(xs, ys)
        err = (y_pred.cpu() - ys.cpu()).square()
        result_errs[baseline_model.name] = err

    result_errs_agg = aggregate_metrics(result_errs, 20)
    print(result_errs.keys())

    import matplotlib.pyplot as plt
    import matplotlib

    fig, ax = plt.subplots(1, figsize=fig_hparam['figsize'])

    err_result_dict_agg = result_errs_agg

    cmap = matplotlib.cm.get_cmap("coolwarm")

    # result_name_list = ['Transformer', 'Looped Transformer']
    # result_name_list = ['Transformer', 'Least Squares', '3-Nearest Neighbors', 'Averaging', 'Looped Transformer']
    result_name_list = ['Transformer', 'Least Squares', '3-Nearest Neighbors', "gd_model_opt_alg=adam_lr=0.005_batch_size=10_num_steps=1000", 'Looped Transformer']
    colors = cmap(np.linspace(0, 1, len(result_name_list)))
    for idx, model_name in enumerate(result_name_list):
        err = err_result_dict_agg[model_name]["mean"]
        if "gd_model" in model_name:
            label_name = "2-layer NN, GD"
        else:
            label_name = model_name
        if "Looped" in model_name:
            ls = '-'
        else:
            ls = '-'
        ax.plot(err, color=colors[idx], lw=fig_hparam['linewidth'], label=label_name, ls=ls)
        low = err_result_dict_agg[model_name]["bootstrap_low"]
        high = err_result_dict_agg[model_name]["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3, color=colors[idx])

    ax.tick_params(axis='both', labelsize=fig_hparam['ticksize'])
    ax.axhline(1, color='k', ls='--', lw=fig_hparam['linewidth'])
    ax.set_ylim(-0.1, 1.25)
    # plt.xticks(np.arange(0, n_points))
    plt.rc('font', family='serif')
    ax.set_xlabel("in-context examples", fontsize=fig_hparam['labelsize'])
    y_label = ax.set_ylabel("squared error", fontsize=fig_hparam['labelsize'])
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fig_hparam['fontsize'])

    plt.savefig("/work/gg45/g45004/looped_transformer/result_folder/Figures/relu2nn_err.png", dpi=600, bbox_inches='tight')
    plt.savefig("/work/gg45/g45004/looped_transformer/result_folder/Figures/relu2nn_err.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.close()