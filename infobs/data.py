# Contient les fonctions de prétraitement des données communes aux deux approches

from typing import Union

import numpy as np

__all__ = [
    "replace_negative"
]

def replace_negative(
    data: np.ndarray,
    noise: Union[None, float, np.ndarray]=None
):
    """
    Replace negative (or zero) samples in data by non-informative ones.
    If `noise` is not None, then they are replaced by a gaussian noise (in absolute value to ensure positivity) of RMS indicated by `noise` (float, 1D or 2D ndarray).
    If `noise` is None, then the negative samples are replaced by zeros.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be an instance of ndarray, not {type(data)}")
    if noise is not None and not isinstance(noise, (float, np.ndarray)):
        raise TypeError(f"noise must be None, a float or an instance of ndarray, not {type(noise)}")

    if noise is None:
        a = 0.
    else:
        a = np.abs(np.random.normal(0, noise, size=data.shape))
    return np.where(data > 0, data, a)
