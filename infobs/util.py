from matplotlib import colors

import numpy as np

__all__ = [
    "truncate_colormap",
    "expformat"
]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ TODO """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def expformat(value: str) -> str:
    """ TODO """
    if value == 0:
        return "0"
    value = f"{value:.1e}"
    mant = value[:-4]
    if mant[-1] == '0':
        mant = mant[:-2]
    exp = value[-2:]
    newexp = int(exp)
    if newexp == 0:
        return f"{mant}"
    return f"{mant}\\cdot 10^{newexp}"
