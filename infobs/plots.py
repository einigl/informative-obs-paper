import itertools as itt
from typing import List, Dict, Tuple, Callable, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .util import truncate_colormap, expformat

__all__ = [
    "Plotter"
]

class Plotter():

    line_formatter: Callable
    param_formatter: Callable

    def __init__(
        self,
        line_formatter: Callable,
        param_formatter: Callable
    ):
        self.line_formatter = line_formatter
        self.param_formatter = param_formatter


    # MI plots (discrete)

    def plot_mi_bar(
        self,
        lines: List[str],
        mis: List[float],
        sorted:bool=False,
        errs: Optional[List[Union[float, Tuple[float, float]]]]=None,
    ) -> Figure:
        ###
        dpi = 200
        width = 0.6
        xscale = 1.2
        yscale = 1.0
        capsize = 6
        ###

        fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, yscale*4.8), dpi=dpi)

        if sorted:
            indices = np.array(mis).argsort()[::-1]
            mis = [mis[i] for i in indices]
            lines = [lines[i] for i in indices]

        ax.bar(np.arange(len(mis)), mis, width=width, color='tab:blue')
        ax.errorbar(np.arange(len(mis)), mis, yerr=errs, fmt='none', capsize=capsize, color='tab:red')

        ax.set_xticks(np.arange(len(mis)))
        ax.set_xticklabels([self.line_formatter(l, short=True, equation_mode=True) for l in lines], rotation=45, fontsize = 12)

        ax.set_xlabel('Integrated molecular lines', labelpad=20)
        ax.set_ylabel('Mutual information (bits)', labelpad=20)

        return fig   

    def plot_mi_matrix(
        self,
        lines: List[str],
        mis: List[List[float]],
        show_diag: bool=True,
    ) -> Figure:
        ###
        dpi = 200
        cmap = 'OrRd'
        xscale = 1.0
        yscale = 1.0
        ###

        fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, yscale*4.8), dpi=dpi)

        mis = np.array(mis)
        mask = np.where(
            np.tril(np.ones_like(mis), k=-1 if show_diag else 0),
            float('nan'), 1.
        )
        im = ax.imshow(mask * mis, origin='lower', cmap=cmap)

        cbar = fig.colorbar(im)
        cbar.set_label('Mutual information (bits)', labelpad=30, rotation=270)

        ax.set_xticks(np.arange(mis.shape[0]))
        ax.set_xticklabels([self.line_formatter(l, short=True, equation_mode=True) for l in lines], rotation=45, fontsize=12)
        ax.set_yticks(np.arange(mis.shape[0]))
        ax.set_yticklabels([self.line_formatter(l, short=True, equation_mode=True) for l in lines], rotation=45, fontsize=12);

        return fig


    # MI plots (continuous)

    def plot_mi_profile(
        self
    ):
        pass

    def plot_mi_map(
        self
    ):
        pass


    # Summaries

    def plot_summary_1d(
        self,
        parameters: Tuple[str, ...],
        regimes: Dict[str, Dict[str, Tuple]],
        best_lines: List[Tuple[str, ...]],
        confidences: List[float],
    ) -> Figure:
        """
        Plot the summary of the most informative lines. The constraint is on a single parameter.
        `parameter` is the set of physical parameter to estimate
        Format (example): ('dust-g0',)
        `regimes` contains the bounds for all subregimes
        Format (example): {'dust-av': {'1': [1, 2], '2': [2, None]}}
        `best_lines` contains a
        Format (example): [('13co10', 'c18o10'), ('n2hp10')]
        `confidence` contains the probabilities for the lines in `best_lines` to be the best.
        Format (example): [(line1, line2), (line3)]
        """
        ###
        xscale = 1.2
        yscale = 1.0
        dpi = 200
        ###

        fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, 0.5*yscale*4.8), dpi=dpi)

        # Checking
        if isinstance(parameters, str):
            parameters = (parameters,)

        # Plot grid
        param_regime = list(regimes.keys())[0]
        x = []
        for val in regimes[param_regime].values():
            if val is None or val[0] is None:
                continue
            ax.axvline(len(x)+1, color='black')
            
            if param_regime in ["dust-g0"]:
                x.append(f"${expformat(val[0])}$")
            else:
                x.append(f"${val[0]}$")
            if val[1] is None:
                x.append("$+\\infty$")
        
        # Static settings
        fontsizes = {
            1: 13,
            2: 10,
            3: 10,
            4: 10
        }

        # Plot names and confidences
        cmap = plt.get_cmap("gist_rainbow")
        subcmap = truncate_colormap(cmap, 0.0, 0.35)
        for i, (l, c) in enumerate(zip(best_lines, confidences), 1):
            if l is not None:
                if isinstance(l, str):
                    l = (l,)
                    c = (c,)
                l = list(l)
                c = list(c)
                sign = [None] * len(l)

                ax.add_patch(
                    Rectangle((i, 0), 1, 1, color=subcmap(c[0]), alpha=0.4)
                )
                for k, _ in enumerate(c):
                    _c = 100 * c[k]
                    if _c > 99.9:
                        _c, _sign = 99.9, ">"
                    elif _c < 0.1:
                        _c, _sign = 0.1, "<"
                    else:
                        _sign = "="
                    c[k], sign[k] = _c, _sign
                ax.text(
                    i+0.5, 0.5,
                    '\n\n'.join([
                        f"${'(' if len(_l) > 1 else ''}{','.join([self.line_formatter(_ll, short=True) for _ll in _l])}{')' if len(_l) > 1 else ''}$\n$p {_sign} {_c:.1f}\%$"\
                            for _l, _c, _sign in zip(l, c, sign)
                    ]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=fontsizes[len(l)]
                )
            else:
                ax.add_patch(
                    Rectangle((i, 0), 1, 1, color="gray", alpha=0.6)
                )
                ax.add_patch(
                    Rectangle((i, 0), 1, 1, fill=False, hatch="//")
                )

        # Settings
        ax.set_xticks(np.arange(1, len(x)+1))
        ax.set_yticks([])
        ax.set_xticklabels(x)
        ax.set_xlabel(self.param_formatter_formatter(param_regime, equation_mode=True), labelpad=10)
        ax.set_xlim([1, len(x)])
        ax.set_ylim([0, 1])

        return fig

    def plot_summary_2d(
        self,
        parameters: Tuple[str, ...],
        regimes: Dict[str, Dict[str, Tuple]],
        best_lines: List[List[Tuple[str, ...]]],
        confidences: List[List[float]],
    ):
        ###
        xscale = 1.2
        yscale = 1.0
        dpi = 200
        ###

        fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, yscale*4.8), dpi=dpi)

        # Checking
        if isinstance(parameters, str):
            parameters = (parameters,)

        # Plot grid
        param_regime_1, param_regime_2 = list(regimes.keys())[0:2]
        x, y = [], []
        for val in regimes[param_regime_1].values():
            if val is None or val[0] is None:
                continue
            ax.axvline(len(x)+1, color='black')
            x.append(f"${val[0]}$")
            if val[1] is None:
                x.append("$+\\infty$")
        for val in regimes[param_regime_2].values():
            if val is None or val[0] is None:
                continue
            ax.axhline(len(y)+1, color='black')
            y.append(f"${expformat(val[0])}$")
            if val[1] is None:
                y.append("$+\\infty$")

        # Static settings
        coords = {
            1: [(0.5, 0.5)],
            2: [(0.5, 0.7), (0.5, 0.3)],
            3: [(0.5, 0.7), (0.25, 0.3), (0.75, 0.3)],
            4: [(0.25, 0.7), (0.75, 0.7), (0.25, 0.3), (0.75, 0.3)]
        }
        fontsizes = {
            1: 13,
            2: 10,
            3: 8,
            4: 8
        }
        
        # Plot names and confidences
        cmap = plt.get_cmap("gist_rainbow")
        subcmap = truncate_colormap(cmap, 0.0, 0.35)
        for i, j in itt.product(range(len(best_lines)), range(len(best_lines[0]))):
            l = best_lines[i][j]
            c = confidences[i][j]

            if l is not None:
                if isinstance(l, str):
                    l = (l,)
                    c = (c,)

                ax.add_patch(
                    Rectangle((i+1, j+1), 1, 1, color=subcmap(c[0]), alpha=0.4)
                )
                for k, _ in enumerate(l):
                    _c = 100 * c[k]
                    if _c > 99.9:
                        _c, _sign = 99.9, ">"
                    elif _c < 0.1:
                        _c, _sign = 0.1, "<"
                    else:
                        _sign = "="
                    _l = l[k]
                
                    i0, j0 = coords[len(l)][k]
                    ax.text(
                        i+1+i0, j+1+j0,
                        f"${'(' if len(_l) > 1 else ''}{','.join([self.line_formatter(_ll, short=True) for _ll in _l])}{')' if len(_l) > 1 else ''}$\n$p {_sign} {_c:.1f}\%$",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=fontsizes[len(l)]
                    )
            else:
                ax.add_patch(
                    Rectangle((i+1, j+1), 1, 1, color="gray", alpha=0.6)
                )
                ax.add_patch(
                    Rectangle((i+1, j+1), 1, 1, fill=False, hatch="//")
                )

        # Settings
        ax.set_xticks(np.arange(1, len(x)+1))
        ax.set_yticks(np.arange(1, len(y)+1))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)
        ax.set_xlabel(self.param_formatter(param_regime_1, equation_mode=True), labelpad=10)
        ax.set_ylabel(self.param_formatter(param_regime_2, equation_mode=True), labelpad=10)
        ax.set_xlim([1, len(x)])
        ax.set_ylim([1, len(y)])

        return fig

    # Helpers

    def regime_formatter(
        self,
        param_name: str,
        reg: Optional[Tuple[Optional[float], Optional[float]]],
        lower_bound: Optional[float]=0,
        upper_bound: Optional[float]=None
    ) -> str:
        lb = "-\infty" if lower_bound is None else expformat(lower_bound)
        ub = "+\infty" if upper_bound is None else expformat(upper_bound)

        if reg is None or reg[0] is None and reg[1] is None:
            return f"${lb} < {self.param_formatter(param_name)} < {ub}$"
        if reg[0] is None:
            return f"${lb} < {self.param_formatter(param_name)} < {expformat(reg[1])}$"
        if reg[1] is None:
            return f"${expformat(reg[0])} < {self.param_formatter(param_name)} < {ub}$"
        return f"${expformat(reg[0])} < {self.param_formatter(param_name)} < {expformat(reg[1])}$"
