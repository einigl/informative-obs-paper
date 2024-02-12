import os
import sys
import yaml

import numpy as np

import infovar

scriptpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptpath, ".."))

from orion_util import get_lines, get_params
from helpers import prepare_data

# Load input dict

with open(os.path.join(scriptpath, 'inputs_continuous.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Load and preprocess data
    
line_names, param_names = get_lines(), get_params()
lines, params = prepare_data(line_names, param_names)

lines, params = np.log10(lines), np.log10(params)

# Instantiate handler

save_path = os.path.join(scriptpath, "continuous")
ref_path = os.path.join(scriptpath, "reference.yaml")

handler = infovar.ContinuousHandler(
    save_path=save_path,
    ref_path=ref_path,
)

handler.set_data(
    variables=lines,
    variable_names=line_names,
    targets=params,
    target_names=param_names,
)

handler.set_fn(
    fn_bounds=[np.log10, np.log10],
    inv_fn_bounds=[lambda t: 10**t, lambda t: 10**t]
)

# Launch the procedure

handler.overwrite(
    inputs_dict
)
