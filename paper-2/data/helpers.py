import os
import sys
from typing import List, Tuple

import numpy as np

abspath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(abspath, ".."))
sys.path.insert(0, os.path.join(abspath, "..", ".."))

import infobs
import orion_util


# Data processing helpers

def prepare_data(
    line_names: List[str],
    param_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    integs = np.column_stack([
        orion_util.load_integral(l)[0].flatten() for l in line_names
    ])
    noises = np.column_stack([
        orion_util.load_noise_map(l)[0].flatten() for l in line_names
    ])
    params = np.column_stack([
        orion_util.load_param(p)[0].flatten() for p in param_names
    ])

    integs = infobs.data.replace_negative(integs, noises)

    return integs, params
