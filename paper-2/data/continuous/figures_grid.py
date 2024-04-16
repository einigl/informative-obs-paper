import os
import sys
import yaml
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import infovar

sys.path.insert(0, os.path.join("..", "..", ".."))
from infobs.plots import Plotter

sys.path.insert(1, os.path.join("..", ".."))
from orion_util import latex_line, latex_param

results_path = os.path.join("results") # "old"
figures_path = os.path.join("grids")

plt.rc("text", usetex=True)


handler = infovar.ContinuousHandler()

handler.set_paths(
    save_path=results_path,
)

plotter = Plotter(
    line_formatter=latex_line,
    param_formatter=latex_param
)

latex_comb_lines = lambda ls: plotter.lines_comb_formatter(ls, short=True)
latex_comb_params = lambda ps: plotter.params_comb_formatter(ps)


lines_list = [
    ["12co10"],
    ["13co10"],
    ["c18o10"],
    ["hcop10"],
    ["hcn10"],
    ["hnc10"],
    ["cch10"],
    ["12cn10"],
    ["12cs21"],
    ["n2hp10"],
    ["h2co"],
    ["32so21"],
]

for params, share_cbar in zip(
    [["av"], ["av"], ["g0"], ["g0"]], [True, False, True, False]
):

    wins_features = ["av", "g0"]
    stat = "mi"

    ticksfontsize = 18
    labelfontsize = 22
    titlefontsize = 24

    rows, cols = 4, 3

    # Look for vmax
    vmax = 0
    for lines in lines_list:

        try:
            d = handler.read(
                lines, params, wins_features
            )
        except:
            continue

        data = d["stats"][stat]["data"]
        vmax = max(vmax, np.nanmax(data))

        wins_features = d["features"]


    fig, axs = plt.subplots(rows, cols, figsize=(cols*6.4, rows*4.8), dpi=200)
    axs = axs.flatten()

    for i, lines in enumerate(lines_list):

        r, c = np.unravel_index(i, (rows, cols))

        try:
            d = handler.read(
                lines, params, wins_features
            )
        except:
            continue

        data = d["stats"][stat]["data"].T
        samples = d["stats"][stat]["samples"]
        yticks, xticks = d["stats"][stat]["coords"]
        wins_features = d["features"]

        paramx, paramy = wins_features

        #

        X, Y = np.meshgrid(yticks, xticks)

        # print(yticks, xticks) TODO
        # print(data.shape)
        # exit()

        im = axs[i].pcolor(X, Y, data, cmap='jet', vmin=0, vmax=vmax if share_cbar else None)
        # ax.set_xlim(lims[params_regime[0]])
        # ax.set_ylim(lims[params_regime[1]])

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')

        axs[i].tick_params(axis='both', which='major', labelsize=ticksfontsize)

        if r == rows-1:
            axs[i].set_xlabel(f"${plotter.param_formatter(paramx)}$", fontsize=labelfontsize)
        if c == 0:
            axs[i].set_ylabel(f"${plotter.param_formatter(paramy)}$", fontsize=labelfontsize)
        axs[i].set_title(f"${plotter.lines_comb_formatter(lines)}$", fontsize=titlefontsize, pad=10)

        cbar = fig.colorbar(im)
        # cbar.set_label("Amount of information (bits)", labelpad=10)
        cbar.ax.tick_params(labelsize=ticksfontsize) 

    # plt.suptitle(f"Amount of information (bits) between ${plotter.params_comb_formatter(params)}$ and lines")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "_".join(params) + f"_grid{'_same' if share_cbar else ''}.png"))
