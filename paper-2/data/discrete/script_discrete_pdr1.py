import os
import sys
import yaml
import itertools as itt
from math import comb

import infovar

scriptpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptpath, "..", "..", ".."))
sys.path.insert(1, os.path.join(scriptpath, "..", ".."))
sys.path.insert(2, os.path.join(scriptpath, ".."))

from infobs.stats import LinearInfoLog10
from orion_util import get_lines, get_params
from helpers import prepare_data, select_envs

# Load input dict

with open(os.path.join(scriptpath, 'inputs_discrete_pdr1.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Load and preprocess data
    
line_names, param_names = get_lines(), ["av", "g0"]
lines, params = prepare_data(line_names, param_names)

lines, params = select_envs(lines, params, ["pdr1"])


# Instantiate handler

save_path = os.path.join(scriptpath, "results_pdr1")

handler = infovar.DiscreteHandler()

handler.set_paths(
    save_path=save_path,
)


# Getter

getter = infovar.StandardGetter(
    line_names, param_names,
    lines, params
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

n_lines = [1, 2] #[1, 2, 3]


# Variables loop

for params in param_names:
    iterable = itt.chain(*[itt.combinations(line_names, k) for k in n_lines])
    total = sum([comb(len(line_names), k) for k in n_lines])

    handler.update(
        iterable, 
        params,
        inputs_dict,
        iterable_x=True,
        save_every=10,
        total_iter=total
    )

