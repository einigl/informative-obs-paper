import os
import sys
import yaml
from typing import List, Dict, Optional, Tuple
from random import shuffle

import numpy as np
from tqdm import tqdm

import infovar

scriptpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptpath, ".."))
sys.path.insert(1, os.path.join(scriptpath, "..", ".."))

from orion_util import get_lines, get_params
from helpers import prepare_data

# Load input dict

with open(os.path.join(scriptpath, 'inputs_continuous.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Load and preprocess data
    
all_line_names, all_param_names = get_lines(), get_params()
lines, params = prepare_data(all_line_names, all_param_names)

# lines, params = np.log10(lines), np.log10(params)
lines, params = lines, params


class Getter:

    line_names = all_line_names
    param_names = all_param_names
    lines = lines
    params = params

    @classmethod
    def getter(
        cls,
        x_features: List[str], y_features: List[str],
        restrictions: Dict[str, Tuple[float]],
        max_samples: Optional[int]=None
    ):
        lines_idx = [cls.line_names.index(v) for v in x_features]
        _lines = np.column_stack([
            cls.lines[:, i] for i in lines_idx
        ])

        params_idx = [cls.param_names.index(v) for v in y_features]
        _params = np.column_stack([
            cls.params[:, i] for i in params_idx
        ])
        
        filt = np.ones(_params.shape[0], dtype="bool")

        # Remove non-finite pixels in variables
        filt &= np.isfinite(_lines).all(axis=1)

        # Remove pixels out of the targets ranges (including NaNs)
        for key, (low, upp) in restrictions.items():
            if key in cls.line_names:
                i = cls.line_names.index(key)
                filt &= np.logical_and(
                    cls.lines[:, i] >= low,
                    cls.lines[:, i] <= upp
                )
            elif key in cls.param_names:
                i = cls.param_names.index(key)
                filt &= np.logical_and(
                    cls.params[:, i] >= low,
                    cls.params[:, i] <= upp
                )

        _lines = _lines[filt]
        _params = _params[filt]
        
        if max_samples is None or _lines.shape[0] <= max_samples:
            return _lines, _params
        
        idx = list(range(_lines.shape[0]))
        shuffle(idx)
        idx = idx[:max_samples]

        return _lines[idx, :], _params[idx, :]


# Instantiate handler

save_path = os.path.join(scriptpath, "results")

handler = infovar.ContinuousHandler()

handler.set_paths(
    save_path=save_path
)

handler.set_getter(
    Getter.getter
)


# Launch the procedure

# LINES
# 13co10
# 12cs21
# 12co10
# hcn10
# n2hp10
# hcop10
# cch10
# 32so21
# hnc10
# c18o10
# h2co
# 12cn10

# PARAMETERS
# av
# g0
# [av, g0]

line_names = [
    # AV
    # ["12co10"],
    # ["13co10"],
    # ["c18o10"],
    # ["hcop10"],
    # ["n2hp10"],
    # ["12co10", "13co10", "c18o10", "hcop10", "n2hp10"],
    # G0
    ["12co10"],
    ["13co10"],
    ["cch10"],
    ["hcop10"],
    ["hcn10"],
    ["12co10", "13co10", "cch10", "hcop10", "hcn10"]
]

param_names = [
    # ["av"],
    ["g0"]
]

for params in param_names:
    print(str(params).replace("'", ""))

    pbar = tqdm(
        line_names,
        desc=str(params).replace("'", "")
    )
    for lines in pbar:
        pbar.set_postfix({'lines': str(lines).replace("'", "")})

        handler.update(
            lines,
            params,
            inputs_dict
        )
