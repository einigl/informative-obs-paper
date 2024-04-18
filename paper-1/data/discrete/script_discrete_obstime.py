import os
import sys
import yaml
import itertools as itt
from typing import List, Dict, Tuple, Optional
from random import shuffle
from math import comb

import numpy as np
from tqdm import tqdm

import infovar

scriptpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptpath, "..", ".."))

from pdr_util import PDRGetter

# Increasing factor in observing time

factor = 10

# Load input dict

with open(os.path.join(scriptpath, 'inputs_discrete.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Instantiate handler

save_path = os.path.join(scriptpath, f"results_obstime_{factor}")

handler = infovar.DiscreteHandler()

handler.set_paths(
    save_path=save_path,
)


# Getter

n_samples = 1_000_000
env = "horsehead"
obs_time = 4 * 0.1 # 10s
seed = 0

getter = PDRGetter(
    n_samples,
    env,
    obs_time,
    seed=seed
)

handler.set_getter(
    getter.get
)

with open('envs.yaml', 'r') as f:
    envs_dict = yaml.safe_load(f)

handler.set_restrictions(
    envs_dict
)


# Launch the procedure

line_names = getter.x_names
param_names = ["Avmax", "radm"] # "P"

n_lines = [1, 2]

emir_line_names = line_names[:-3]
fir_line_names = line_names[-3:]


# Variables loop (FIR lines)
        
for params in param_names:
    iterable = itt.chain(*[itt.combinations(fir_line_names, k) for k in n_lines])
    total = sum([comb(len(fir_line_names), k) for k in n_lines])

    handler.update(
        iterable, 
        params,
        inputs_dict,
        iterable_x=True,
        save_every=1,
        total_iter=total
    )

# Variables loop (EMIR lines)

for params in param_names:
    iterable = itt.chain(*[itt.combinations(emir_line_names, k) for k in n_lines])
    total = sum([comb(len(emir_line_names), k) for k in n_lines])

    handler.update(
        iterable, 
        params,
        inputs_dict,
        iterable_x=True,
        save_every=100,
        total_iter=total
    )
