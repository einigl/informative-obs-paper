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

with open(os.path.join(scriptpath, 'inputs_discrete.yaml'), 'r') as f:
    inputs_dict = yaml.safe_load(f)

# Load and preprocess data
    
line_names, param_names = get_lines(), get_params()
lines, params = prepare_data(line_names, param_names)

lines, params = np.log10(lines), np.log10(params)

# Instantiate handler

save_path = os.path.join(scriptpath, "discrete")
ref_path = os.path.join(scriptpath, "reference.yaml")

handler = infovar.DiscreteHandler(
    variables=lines,
    variable_names=line_names,
    targets=params,
    target_names=param_names,
    save_path=save_path,
    ref_path=ref_path,
    fn_bounds=[np.log10, np.log10]
)

# Launch the procedure

handler.update(
    inputs_dict
)
