import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
                               
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import helpers
from pdr_util import latex_line, remove_hyperfine, Settings

Settings.only_rotational = True

plt.rc("text", usetex=True)

figures_dir = "figures"


### SELECTED LINES LOADING ###

df = pd.read_csv("emir_table_filtered.csv")

lines = df["line_id"].to_list()
freqs = df["freq"].to_numpy()

lines = [remove_hyperfine(l) for l in lines]


### NOISE PROFILE LOADING ###

observing_time = helpers.orionb_obstime # In minutes

data = np.load("rms_profile.npz")
f_array, n_array = data["freq"], data["rms"]

n_array = np.sqrt(observing_time) * n_array

### GLOBAL DISPLAY ###

rms_min = 0 # mK
rms_max = 800 # mK
rms_delta = rms_max - rms_min

for show_noise in [True, False]:

    plt.figure(figsize=(6.4, 0.5*4.8), dpi=200)

    for f in freqs:
        plt.axvline(f, 0.05, 0.95, linewidth=0.8)
        
    bands = helpers.emir_bands()

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    lowest, highest = -float("inf"), float("inf")
    for name, (low, upp) in bands.items():
        lowest, highest = min(lowest, low), max(highest, upp)
        plt.plot([low, upp], 2*[rms_min+0.9*rms_delta], color="red")
        plt.text(
            (low+upp)/2, rms_min+0.80*rms_delta, name,
            horizontalalignment='center',
            verticalalignment='center',
            color="red", weight='bold', bbox=props
        )

    if show_noise:
        mask = (lowest <= f_array) & (f_array <= highest)
        plt.plot(f_array[mask], 1e3*n_array[mask], color="tab:gray")
        plt.ylabel("Noise RMS (mK)", labelpad=10)

    plt.xlabel("$f$ (GHz)")
    plt.ylim([rms_min, rms_max])
    if show_noise:
        pass
    else:
        plt.yticks([])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))

    suffix = '_noise' if show_noise else ''
    plt.savefig(os.path.join(figures_dir, f"selected_lines{suffix}.png"), bbox_inches="tight")


### BAND-WISE DISPLAY ###

min_space = 0.022 # [0, 1]
global_offset_pos = -0.01

percentile = 90

for show_noise in [True, False]:
    for mode in ("short", "full"):

        for band, (low, upp) in helpers.emir_bands().items():

            plt.figure(figsize=(6.4, 0.5*4.8), dpi=200)

            mask = (low <= f_array) & (f_array <= upp)

            rms_min = 1e3*np.min(n_array[mask]) # mK
            rms_max = min(1e3*np.percentile(n_array[mask], percentile), 900) # mK
            rms_min = max(rms_min - 0.1 * (rms_max - rms_min), 0)
            rms_delta = rms_max - rms_min               

            plt.plot([low, upp], 2*[rms_min+0.9*rms_delta], color="red", label=f"{band} band")

            if show_noise:
                plt.plot(f_array[mask], 1e3*n_array[mask], color="tab:gray")
                plt.ylabel("Noise RMS (mK)", labelpad=10)

            _lines = []
            _freqs = []
            for line, f in zip(lines, freqs):
                if not low <= f <= upp:
                    continue
                plt.axvline(f, 0.05, 0.95, linewidth=0.8)
                _lines.append(line)
                _freqs.append(f)

            xlim = plt.gca().get_xlim()
            factor = xlim[1] - xlim[0]
            min_df = min_space * factor

            _positions = []
            group_f, group_p = [], []

            i = 0
            while i < len(_freqs):
                f = _freqs[i]
                if len(group_f) == 0:
                    group_f.append(f)
                    group_p.append(f)
                    i += 1
                elif f - group_f[-1] < min_df:
                    group_f.append(f)
                    group_p.append(group_p[-1] + min_df)
                    i += 1
                else:
                    group_offset = np.mean(group_p) - np.mean(group_f)
                    _positions.extend([p - group_offset for p in group_p])
                    group_f, group_p = [], []
                    # No increment

            group_offset = np.mean(group_p) - np.mean(group_f)
            _positions.extend([p - group_offset for p in group_p])

            for line, f, pos in zip(_lines, _freqs, _positions):
                plt.annotate(
                    "$" + latex_line(line, short=mode=="short") + "$",
                    xy=(pos + factor * global_offset_pos, rms_max + 0.02*rms_delta),
                    rotation=60, ha='left', va='bottom', annotation_clip=False,
                    fontsize=7
                )

            plt.xlabel("$f$ (GHz)")
            plt.ylim([rms_min, rms_max])
            if show_noise:
                pass
            else:
                plt.yticks([])
            plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
            plt.legend(loc="lower left", handlelength=1, borderpad=0.5, fontsize=9)

            suffix = '_noise' if show_noise else ''
            plt.savefig(os.path.join(figures_dir, f"{band}_full{suffix}.png" if mode == "full" else f"{band}{suffix}.png"), bbox_inches="tight")
