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

import sys

sys.path.append("/work/gg45/g45004/looped_transformer/scripts")
from nano_gpt import GPT2Model, GPT2Config
from utils import eval_unlooped_model, aggregate_metrics, eval_looped_model


def get_model(model, result_dir, run_id, step, best=False):
    if best:
        model_path = os.path.join(result_dir, run_id, "model_best.pt")
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        best_err = torch.load(model_path, map_location="cpu")["loss"]
        print("saved model with loss:", best_err)
    if step == -1:
        model_path = os.path.join(result_dir, run_id, "state.pt")
        state_dict = torch.load(model_path, map_location="cpu")["model_state_dict"]
    else:
        model_path = os.path.join(result_dir, run_id, "model_{}.pt".format(step))
        state_dict = torch.load(model_path, map_location="cpu")["model"]

    #     return state_dict
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    return model


class LinearRegression:
    def __init__(
        self, batch_size, n_points, n_dims, n_dims_truncated, device, w_star=None
    ):
        super(LinearRegression, self).__init__()
        self.device = device
        self.xs = torch.randn(batch_size, n_points, n_dims).to(device)
        self.xs[..., n_dims_truncated:] = 0
        w_b = (
            torch.randn(batch_size, n_dims, 1) if w_star is None else w_star.to(device)
        )  # [B, d, 1]
        w_b[:, n_dims_truncated:] = 0
        self.w_b = w_b.to(device)
        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]


if __name__ == "__main__":
    fig_hparam = {
        "figsize": (8, 5),
        "labelsize": 28,
        "ticksize": 20,
        "linewidth": 5,
        "fontsize": 15,
        "titlesize": 20,
        "markersize": 15,
    }

    # font specification
    fontdict = {
        "family": "serif",
        "size": fig_hparam["fontsize"],
    }

    device = torch.device("cuda:0")
    sample_size = 1280
    batch_size = 128
    n_points = 41
    n_dims_truncated = 20
    n_dims = 20

    real_task = LinearRegression(
        sample_size, n_points, n_dims, n_dims_truncated, device
    )
    xs, ys, w_b = real_task.xs, real_task.ys, real_task.w_b

    result_errs = {}

    from models import TransformerModel

    n_positions = 101
    n_embd = 256
    # n_layer = 12
    n_head = 8

    T = 20

    tf_model_list = [
        # {
        #    "name": "Transformer_L{12}",
        #    "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_baseline",
        #    "run_id": "0525163557-LR_baseline-1697",
        # },
        {
            "name": "Transformer_L{8}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_baseline",
            "run_id": "0611175353-LR_baseline_L{8}-5c5a",
        },
        {
            "name": "Transformer_L{4}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_baseline",
            "run_id": "0611180712-LR_baseline_L{4}-0e31",
        },
    ]

    for model_dict in tf_model_list:
        n_layer = int(model_dict["name"].split("{")[1].split("}")[0])
        model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head)
        step = -1
        model = get_model(model, model_dict["result_dir"], model_dict["run_id"], step)
        model = model.to(device)

        err, y_pred_total = eval_unlooped_model(model, xs, ys)
        err = err[:, -1]
        err = err.unsqueeze(1).repeat(1, T)
        result_errs[model_dict["name"]] = err

    from models import TransformerModelLooped

    # result_dir = '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop'
    # run_id = '0609104043-Distillate_LR_loop_L1_ends{30}_T{15}-20ed'

    # eval multiple models

    """
    model_list = [
        {
            "name": "Looped_Transformer_b{32}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0609110718-LR_loop_L1_ends{32}_T{15}-1338',
        },
        {
            "name": "Looped_Transformer_b{16}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0610131812-Distillate_LR_loop_L1_ends{16}-2ecd',
        },
        {
            "name": "Looped_Transformer_b{8}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0610173845-Distillate_LR_loop_L1_ends{8}-4b8e',
        },
        {
            "name": "Looped_Transformer_b{4}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0610183139-Distillate_LR_loop_L1_ends{4}-ac31',
        },
        {
            "name": "Looped_Transformer_b{2}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0610212309-Distillate_LR_loop_L1_ends{2}-cf84',
        },
        {
            "name": "Looped_Transformer_b{1}",
            "result_dir": '/work/gg45/g45004/looped_transformer/results2/linear_regression_loop',
            "run_id": '0610214028-Distillate_LR_loop_L1_ends{1}-18ed',
        },
    ]
    """
    model_list = [
        {
            "name": "Looped_Pretrain_b{8}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
            "run_id": "0611192515-LR_loop_L1_ends{8}_T{8}-68b0",
        },
        {
            "name": "Looped_Pretrain_b{4}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
            "run_id": "0611192839-LR_loop_L1_ends{4}_T{4}-ed71",
        },
        # {
        #    "name": "Looped_Pretrain_b{32}",
        #    "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
        #    "run_id": "0609110718-LR_loop_L1_ends{32}_T{15}-1338",
        # },
        # {
        #    "name": "Looped_Progressive_b{16}",
        #    "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
        #    "run_id": "0610131812-Distillate_LR_loop_L1_ends{16}-2ecd",
        # },
        {
            "name": "Looped_Progressive_b{8}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
            "run_id": "0610173845-Distillate_LR_loop_L1_ends{8}-4b8e",
        },
        {
            "name": "Looped_Progressive_b{4}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
            "run_id": "0610183139-Distillate_LR_loop_L1_ends{4}-ac31",
        },
        {
            "name": "Looped_Direct_b{4}",
            "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
            "run_id": "0610222316-Progressive_Distillation_LR_loop_teacher{32}_student{4}-2175",
        },
        # {
        #    "name": "Looped_Transformer_b{2}",
        #    "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
        #    "run_id": "0610212309-Distillate_LR_loop_L1_ends{2}-cf84",
        # },
        # {
        #    "name": "Looped_Transformer_b{1}",
        #    "result_dir": "/work/gg45/g45004/looped_transformer/results2/linear_regression_loop",
        #    "run_id": "0610214028-Distillate_LR_loop_L1_ends{1}-18ed",
        # },
    ]

    n_positions = 101
    n_embd = 256
    n_head = 8
    # T = 20
    n_layer = 1

    model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)
    step = -1

    # result_errs = {}

    for model_dict in model_list:
        model = get_model(model, model_dict["result_dir"], model_dict["run_id"], step)
        model = model.to(device)

        _, loop_err = eval_looped_model(model, xs, ys, loop_max=T)
        # loop_err: [B, T]
        result_errs[model_dict["name"]] = loop_err

    """
    from utils import get_relevant_baselines

    baselines = get_relevant_baselines("linear_regression")
    # baseline_errs = {}
    for baseline_model in baselines:
        y_pred = baseline_model(xs, ys)
        err = (y_pred.cpu() - ys.cpu()).square()
        result_errs[baseline_model.name] = err

    result_errs_agg = aggregate_metrics(result_errs, n_dims_truncated)

    print(result_errs_agg.keys())
    """

    result_errs_agg = result_errs

    import matplotlib.pyplot as plt
    import matplotlib

    fig, ax = plt.subplots(1, figsize=fig_hparam["figsize"])

    err_result_dict_agg = result_errs_agg

    cmap = matplotlib.cm.get_cmap("coolwarm")
    # result_name_list = ['Least Squares', '3-Nearest Neighbors', 'Averaging', 'Looped Transformer']  # ,
    baseline_name_list = []
    result_name_list = (
        [model_dict["name"] for model_dict in tf_model_list]
        + [model_dict["name"] for model_dict in model_list]
        + baseline_name_list
    )

    colors = cmap(np.linspace(0, 1, len(result_name_list)))
    for idx, model_name in enumerate(result_name_list):
        # plot loop_err of each i th loop
        err = err_result_dict_agg[model_name]  # [B, T]
        err_mean = err.mean(0)  # [T]
        # err_std = err.std(0)
        ax.plot(
            err_mean,
            color=colors[idx],
            lw=fig_hparam["linewidth"],
            label=model_name.capitalize(),
        )
        # if looped, ax.axvline(loop, color="k", ls="--", lw=fig_hparam["linewidth"])
        if "Looped" in model_name:
            b = int(model_name.split("{")[-1].split("}")[0])
            ax.axvline(b, color=colors[idx], ls="--", lw=fig_hparam["linewidth"])

        # ax.fill_between(range(len(err_mean)), err_mean - err_std, err_mean + err_std, alpha=0.3, color=colors[idx])

    ax.tick_params(axis="both", labelsize=fig_hparam["ticksize"])
    # ax.axhline(1, color="k", ls="--", lw=fig_hparam["linewidth"])

    # ax.set_ylim(-0.1, 1.25)
    # y軸を対数スケールに設定
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e0)
    from matplotlib.ticker import LogLocator, LogFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0))

    # plt.xticks(np.arange(0, n_points))
    plt.rc("font", family="serif")
    ax.set_xlabel("loop iterations", fontsize=fig_hparam["labelsize"])
    y_label = ax.set_ylabel("squared error", fontsize=fig_hparam["labelsize"])
    legend = plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fig_hparam["fontsize"]
    )

    # Save the figure instead of showing it
    plt.savefig(
        "/work/gg45/g45004/looped_transformer/result_folder/Figures/Distill_LR_loop_err_kkokdefwiofei.png",
        dpi=600,
        bbox_inches="tight",
    )
    # plt.savefig("/work/gg45/g45004/looped_transformer/result_folder/Figures/Distill_LR_loop_err.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
