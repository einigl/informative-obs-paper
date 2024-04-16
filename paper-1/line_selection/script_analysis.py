import os
import sys
import itertools as itt
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(1, os.path.join(".."))

from infovar import DiscreteHandler
from infovar.stats.ranking import prob_higher

from infobs.plots import Plotter
from pdr_util import latex_line, latex_param, Settings

Settings.only_rotational = True

plt.rc("text", usetex=True)

#

handler = DiscreteHandler()

project = 'main' # You have to change the name
only_mi = True
errorbars = True
orionb_compatible = True

print("PROJECT:", project)

dirname = f"results_{project}"

handler.set_paths(
    save_path=os.path.join("..", "data", "discrete", dirname),
)

if orionb_compatible:
    dirname += "_orionb"

#

plotter = Plotter(
    line_formatter=latex_line,
    param_formatter=latex_param,
)

latex_comb_lines = lambda ls: plotter.lines_comb_formatter(ls, short=False)
latex_comb_params = lambda ps: plotter.params_comb_formatter(ps)

#

def param_str(params: Union[str, List[str]]):
    if isinstance(params, str):
        return params
    return "_".join(params)

def regime_str(params: Union[str, Tuple[str, str]], reg: Dict[str, str]):
    if isinstance(params, str):
        return param_str(params) + "_" + reg[params]
    return "_".join([param_str(param) + "_" + reg[param] for param in params])

#

env_name_list = ["all", "pdr1", "pdr2", "pdr3"]
params_list = ["Avmax"] #, "P", "radm"]
if project == "kappa":
    params_list.append("kappa")
n_lines_list = [[1]]#, [1, 2]]#, [1, 2, 3]]
zoom_in_list = [True]

#

fir_lines = [
    "c_el3p_j1__el3p_j0",
    "c_el3p_j2__el3p_j1",
    "cp_el2p_j3_2__el2p_j1_2"
]

if not orionb_compatible:

    emir_lines = [
        "hcn_j1_f2__j0_f1",
        "hcop_j1__j0",
        "hnc_j1__j0",
        "cs_j2__j1",
        "c_18o_j1__j0",
        "13c_o_j1__j0",
        "cn_n1_j0d5__n0_j0d5",
        "cn_n1_j1d5__n0_j0d5",
        "co_v0_j1__v0_j0",
        "cs_j3__j2",
        "hcn_j2_f3__j1_f2",
        "hcop_j2__j1",
        "c_18o_j2__j1",
        "13c_o_j2__j1",
        "cn_n2_j1d5__n1_j0d5",
        "cn_n2_j2d5__n1_j1d5",
        "co_v0_j2__v0_j1",
        "cs_j5__j4",
        "c2h_n3d0_j3d5_f4d0__n2d0_j2d5_f3d0",
        "c2h_n3d0_j2d5_f3d0__n2d0_j1d5_f2d0",
        "hcn_j3_f3__j2_f3",
        "hcop_j3__j2",
        "hnc_j3__j2",
        "cs_j6__j5",
        "13c_o_j3__j2",
        "cn_n3_j3d5__n2_j2d5",
        "cs_j7__j6",
        "co_v0_j3__v0_j2",
        "c2h_n4d0_j4d5_f5d0__n3d0_j3d5_f4d0",
        "hcop_j4__j3"
    ]

else:

    emir_lines = [
        "hcn_j1_f2__j0_f1",
        "hcop_j1__j0",
        "hnc_j1__j0",
        "cs_j2__j1",
        "c_18o_j1__j0",
        "13c_o_j1__j0",
        # "cn_n1_j0d5__n0_j0d5", # TODO verify which one to keep
        "cn_n1_j1d5__n0_j0d5",
        "co_v0_j1__v0_j0",
    ]

#

for env_name, params, n_lines in itt.product(
    env_name_list, params_list, n_lines_list
):

    fig_path = os.path.join(dirname, env_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    #

    if orionb_compatible:
        iterable = itt.chain(
            *[itt.combinations(emir_lines, k) for k in n_lines]
        )
    else:
        iterable = itt.chain(
            *[itt.combinations(emir_lines, k) for k in n_lines],
            *[itt.combinations(fir_lines, k) for k in n_lines]
        )
    iterable = list(iterable)

    items = handler.read(
        iterable,
        params,
        env_name,
        default=None,
        iterable_x=True
    )

    #

    mis = np.array([it["mi"]["value"] for it in items])
    mi_sigmas = np.array([it["mi"]["std"] for it in items])

    if not only_mi:
        lininfos = np.array([it["linearinfo"]["value"] for it in items])
        lininfo_sigmas = np.array([it["linearinfo"]["std"] for it in items])
        lininfologs = np.array([it["linearinfolog10"]["value"] for it in items])
        lininfolog_sigmas = np.array([it["linearinfolog10"]["std"] for it in items])
        lininfogauss = np.array([it["linearinfogauss"]["value"] for it in items])
        lininfogauss_sigmas = np.array([it["linearinfogauss"]["std"] for it in items])

    order_mi = np.argsort(mis)[::-1]

    if not only_mi:
        order_lininfo = np.argsort(lininfos)[::-1]
        order_lininfolog = np.argsort(lininfologs)[::-1]
        order_lininfogauss = np.argsort(lininfogauss)[::-1]

    #

    colors = [None] * len(iterable)
    for i, combs in enumerate(iterable):
        colors[i] = "tab:green" if combs[0] in fir_lines else "tab:blue"

    #

    first_n = 20
    figsize = (0.1*first_n * 6.4, 4.8)
    dpi = 150
    titlesize = 22
    titlepad = 10

    #

    for zoom_in in zoom_in_list:

        plt.figure(figsize=figsize, dpi=dpi)
        _ = plotter.plot_mi_bar(
            [iterable[i] for i in order_mi[:first_n]],
            mis[order_mi[:first_n]],
            errs=mi_sigmas[order_mi[:first_n]] if errorbars else None,
            colors=[colors[i] for i in order_mi[:first_n]],
            sort=False, short_names=False, zoom_in=zoom_in
        )
        plt.title(f"Mutual information between ${latex_comb_params(params)}$ and lines intensity", fontsize=titlesize, pad=titlepad)
        plt.savefig(os.path.join(fig_path, f"{param_str(params)}_mi_{'_'.join([str(n) for n in n_lines])}_{'zoom' if zoom_in else ''}"), bbox_inches="tight")
        plt.close()
        
        #

        if only_mi:
            continue

        #

        plt.figure(figsize=figsize, dpi=dpi)
        _ = plotter.plot_mi_bar(
            [iterable[i] for i in order_lininfo[:first_n]],
            lininfos[order_lininfo[:first_n]],
            errs=lininfo_sigmas[order_lininfo[:first_n]] if errorbars else None,
            colors=[colors[i] for i in order_lininfo[:first_n]],
            sort=False, short_names=False, zoom_in=zoom_in
        )
        plt.title(f"Mutual information (Gaussian asumption) between ${latex_comb_params(params)}$ and lines intensity", fontsize=titlesize, pad=titlepad)
        plt.savefig(os.path.join(fig_path, f"{param_str(params)}_cc_{'_'.join([str(n) for n in n_lines])}_{'zoom' if zoom_in else ''}"), bbox_inches="tight")
        plt.close()
        
        #

        plt.figure(figsize=figsize, dpi=dpi)
        _ = plotter.plot_mi_bar(
            [iterable[i] for i in order_lininfolog[:first_n]],
            lininfologs[order_lininfolog[:first_n]],
            errs=lininfolog_sigmas[order_lininfolog[:first_n]] if errorbars else None,
            colors=[colors[i] for i in order_lininfolog[:first_n]],
            sort=False, short_names=False, zoom_in=zoom_in
        )
        plt.title(f"Mutual information (Gaussian asumption, log) between ${latex_comb_params(params)}$ and lines intensity", fontsize=titlesize, pad=titlepad)
        plt.savefig(os.path.join(fig_path, f"{param_str(params)}_cclog_{'_'.join([str(n) for n in n_lines])}_{'zoom' if zoom_in else ''}"), bbox_inches="tight")
        plt.close()

        #

        plt.figure(figsize=figsize, dpi=dpi)
        _ = plotter.plot_mi_bar(
            [iterable[i] for i in order_lininfogauss[:first_n]],
            lininfogauss[order_lininfogauss[:first_n]],
            errs=lininfogauss_sigmas[order_lininfogauss[:first_n]] if errorbars else None,
            colors=[colors[i] for i in order_lininfogauss[:first_n]],
            sort=False, short_names=False, zoom_in=zoom_in
        )
        plt.title(f"Mutual information (Gaussian asumption, reparam.) between ${latex_comb_params(params)}$ and lines intensity", fontsize=titlesize, pad=titlepad)
        plt.savefig(os.path.join(fig_path, f"{param_str(params)}_ccgauss_{'_'.join([str(n) for n in n_lines])}_{'zoom' if zoom_in else ''}"), bbox_inches="tight")
        plt.close()

        # Comparison

        plt.figure(figsize=figsize, dpi=dpi)
        _ = plotter.plot_mi_bar(
            [iterable[i] for i in order_mi[:first_n]],
            mis[order_mi[:first_n]],
            errs=mi_sigmas[order_mi[:first_n]] if errorbars else None,
            colors=[colors[i] for i in order_mi[:first_n]],
            sort=False, short_names=False, zoom_in=False
        )

        plt.scatter(
            np.arange(first_n), lininfos[order_mi[:first_n]], s=50,
            color="tab:purple", label="Gaussian asumption"
        )
        plt.scatter(
            np.arange(first_n), lininfologs[order_mi[:first_n]], s=50,
            color="tab:olive", label="Gaussian asumption (log)"
        )
        plt.scatter(
            np.arange(first_n), lininfogauss[order_mi[:first_n]], s=50,
            color="tab:orange", label="Gaussian asumption (reparam.)"
        )
        plt.legend(fontsize=18)

        plt.title(f"Mutual information between ${latex_comb_params(params)}$ and lines intensity", fontsize=titlesize, pad=titlepad)
        plt.savefig(os.path.join(fig_path, f"{param_str(params)}_comparison_{'_'.join([str(n) for n in n_lines])}"), bbox_inches="tight")
        plt.close()
