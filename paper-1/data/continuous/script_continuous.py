import os
import sys
import yaml
from typing import List, Dict, Tuple, Optional
from random import shuffle

import numpy as np
from tqdm import tqdm

import infovar

scriptpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptpath, "..", ".."))

from pdr_util import get_physical_env, simulate


# Load input dict

with open(os.path.join(scriptpath, 'inputs_continuous.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Load and preprocess data

n_samples = 500_000
lower, upper = get_physical_env("all")

df = simulate(
    n_samples,
    lower, upper,
    obs_time=0.75,
    seed=0
)

# Conversion to linear scale

df["P"] = df["P"].apply(lambda t: 10**t)
df["radm"] = df["radm"].apply(lambda t: 10**t)
df["Avmax"] = df["Avmax"].apply(lambda t: 10**t)


class Getter:

    df = df

    @classmethod
    def getter(
        cls,
        x_features: List[str], y_features: List[str],
        restrictions: Dict[str, Tuple[float]],
        max_samples: Optional[int]=None
    ):
        features = list(cls.df.columns)
        features_restrict = [key for key in restrictions]

        if not set(x_features) < set(features):
            raise ValueError("TODO")
        if not set(y_features) < set(features):
            raise ValueError("TODO")
        if not set(features_restrict) < set(features):
            raise ValueError("Some restriction or not valid features")

        filt = ~cls.df["P"].isnull() # TODO pas très propre
        for key, (low, upp) in restrictions.items():
            filt &= cls.df[key].between(low, upp)

        x, y = cls.df.loc[filt, x_features].values, cls.df.loc[filt, y_features].values

        if max_samples is None or x.shape[0] <= max_samples:
            return x, y
                
        idx = list(range(x.shape[0]))
        shuffle(idx)
        idx = idx[:max_samples]
        return x[idx, :], y[idx, :]


# Instantiate handler

save_path = os.path.join(scriptpath, "results")

handler = infovar.ContinuousHandler()

handler.set_paths(
    save_path=save_path,
)

handler.set_getter(
    Getter.getter
)


# Launch the procedure

line_names = [
    # Simple lines
    ["co_v0_j1__v0_j0"],
    ["co_v0_j2__v0_j1"],
    ["co_v0_j3__v0_j2"],
    ["13c_o_j1__j0"],
    ["13c_o_j2__j1"],
    ["13c_o_j3__j2"],
    ["c_18o_j1__j0"],
    ["c_18o_j2__j1"],
    ["c_18o_j3__j2"],
    ["hcop_j1__j0"],
    ["hcop_j2__j1"],
    ["hcop_j3__j2"],
    ["hcop_j4__j3"],
    ["hnc_j1__j0"],
    ["hnc_j3__j2"],
    ["hcn_j1_f2__j0_f1"],
    ["hcn_j2_f3__j1_f2"],
    ["hcn_j3_f3__j2_f3"],
    ["cs_j2__j1"],
    ["cs_j3__j2"],
    ["cs_j5__j4"],
    ["cs_j6__j5"],
    ["cs_j7__j6"],
    # CN lines
    ["cn_n1_j0d5__n0_j0d5"],
    ["cn_n1_j1d5__n0_j0d5"],
    ["cn_n2_j1d5__n1_j0d5"],
    ["cn_n2_j2d5__n1_j1d5"],
    ["cn_n3_j3d5__n2_j2d5"],
    # C2H lines
    ["c2h_n1d0_j1d5_f2d0__n0d0_j0d5_f1d0"],
    ["c2h_n2d0_j2d5_f3d0__n1d0_j1d5_f2d0"],
    ["c2h_n3d0_j3d5_f4d0__n2d0_j2d5_f3d0"],
    ["c2h_n3d0_j2d5_f3d0__n2d0_j1d5_f2d0"],   
    ["c2h_n4d0_j4d5_f5d0__n3d0_j3d5_f4d0"],
    # Carbon lines
    ["c_el3p_j1__el3p_j0"],
    ["c_el3p_j2__el3p_j1"],
    ["cp_el2p_j3_2__el2p_j1_2"],
    # Some interesting combinations
    ["co_v0_j1__v0_j0", "co_v0_j2__v0_j1"],
    ["co_v0_j1__v0_j0", "co_v0_j2__v0_j1", "co_v0_j3__v0_j2"],
    #
    ["13c_o_j1__j0", "13c_o_j2__j1"],
    ["13c_o_j1__j0", "13c_o_j2__j1", "13c_o_j3__j2"],
    #
    ["c_18o_j1__j0", "c_18o_j2__j1"],
    ["c_18o_j1__j0", "c_18o_j2__j1", "c_18o_j3__j2"],
    #
    ["co_v0_j1__v0_j0", "13c_o_j1__j0"],
    ["co_v0_j1__v0_j0", "c_18o_j1__j0"],
    ["13c_o_j1__j0", "c_18o_j1__j0"],
    ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"],
    #
    ["c_el3p_j1__el3p_j0", "c_el3p_j2__el3p_j1"],
    ["c_el3p_j1__el3p_j0", "cp_el2p_j3_2__el2p_j1_2"],
    ["c_el3p_j2__el3p_j1", "cp_el2p_j3_2__el2p_j1_2"],
    #
    ["hcn_j1_f2__j0_f1", "hcn_j2_f3__j1_f2"],
    ["hcn_j1_f2__j0_f1", "hcn_j2_f3__j1_f2", "hcn_j3_f3__j2_f3"],
    #
    ["hcn_j1_f2__j0_f1", "hnc_j1__j0"],
    #
    ["cs_j2__j1", "cs_j3__j2"],
]

param_names = [
    ["Avmax"],
    ["radm"],
    ["P"]
]


# Variables loop

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
