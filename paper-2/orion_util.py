"""Contains function to load and display Orion data"""

import os
import pickle
from typing import List, Tuple, Optional, Union, Literal
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import colors
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(path, 'astro_cmap.pickle'), 'rb') as f :
    default_cmap = pickle.load(f)

plt.rcParams["image.origin"] = 'lower' # Origin of astro image in bottom left corner of the figure

__all__ = [
    "get_lines",
    "get_params",
    "load_integral",
    "load_noise_map",
    "load_param",
    "plot_map",
    "latex_line",
    "latex_param",
]

# Loading
    
_lines_dict = {
    '12cn10'    : '12cn10-6-s1-fts',
    '12co10'    : '12co10-s1-fts',
    '12cs21'    : '12cs21-s1-fts',
    '13co10'    : '13co10-s1-fts',
    '32so21'    : '32so21-s1-fts',
    'c3h2'      : 'c3h2-s2-fts',
    'c17o10'    : 'c17o10-s1-fts',
    'c18o10'    : 'c18o10-s1-fts',
    'cch10'     : 'cch10-2-s2-fts',
    'ch3oh21'   : 'ch3oh21-s1-fts',
    'h2co'      : 'h2co-s3-fts',
    'h13cop10'  : 'h13cop10-sa-fts',
    'hcn10'     : 'hcn10-sa-fts',
    'hcop10'    : 'hcop10-sa-fts',
    'hnc10'     : 'hnc10-sa-fts',
    'n2hp10'    : 'n2hp10-sa-fts',
}

_params_dict = {
    'av'        : 'Av',
    'g0'        : 'G0',
    'tkin'      : 'Tkin',
    'nh2'       : 'NH2',
    'density'   : 'density',
}

_envs_dict = {
    'full': 'full',
    'filaments': 'filaments',
    'horsehead': 'horsehead',
    'pdr1': 'pdr1'
}

def get_lines() -> List[str]:
    return list(_lines_dict.keys())

def get_params() -> List[str]:
    return list(_params_dict.keys())

def load_integral(line: str) -> Tuple[np.ndarray, fits.Header]:
    line = line.strip().lower()
    if line not in _lines_dict:
        raise ValueError(f"'{line}' is not an existing line.")
    hdu = fits.open(os.path.join(path, "data", "raw", "integrals", f"{_lines_dict[line]}.fits"))[0]
    return hdu.data, hdu.header

def load_noise_map(line: str) -> Tuple[np.ndarray, fits.Header]:
    line = line.strip().lower()
    if line not in _lines_dict:
        raise ValueError(f"'{line}' is not an existing line.")
    hdu = fits.open(os.path.join(path, "data", "raw", "noise-maps", f"{_lines_dict[line]}.fits"))[0]
    return hdu.data, hdu.header

def load_param(param: str) -> Tuple[np.ndarray, fits.Header]:
    param = param.strip().lower()
    if param not in _params_dict:
        raise ValueError(f"'{param}' is not an existing parameter.")
    hdu = fits.open(os.path.join(path, "data", "raw", "parameters", f"{_params_dict[param]}.fits"))[0]
    return hdu.data, hdu.header

def load_env(env: str) -> Tuple[np.ndarray, fits.Header]:
    env = env.strip().lower()
    if env not in _envs_dict:
        raise ValueError(f"'{env}' is not an existing environment.")
    hdu = fits.open(os.path.join(path, "data", "raw", "envs", f"{_envs_dict[env]}.fits"))[0]
    return hdu.data.astype(bool), hdu.header


# Display

def _colorbar(mappable : img.AxesImage, label : Optional[str] = None, **kwargs):
    """ TODO """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.10)
    cbar = fig.colorbar(mappable, cax = cax, label = label, **kwargs)
    plt.sca(last_axes)
    return cbar

def _check_validity(data: np.ndarray, header: fits.Header) -> Tuple[np.ndarray, fits.Header]:
    # Types
    if not isinstance(data, np.ndarray) :
        raise ValueError(f"data must be a ndarray, not {type(data)}")
    if not isinstance(header, fits.Header) :
        raise ValueError(f"header must be a Header, not {type(header)}")

    # Check data and header number of axis
    if data.ndim != 2 :
        raise ValueError(f"data must have 2 dimensions, not {data.ndim}")
    if header['NAXIS'] != 2 :
        raise ValueError(f"header must have 2 axes, not {header['NAXIS']}")
    
    # Check axes compatibility
    dims = (header['NAXIS2'], header['NAXIS1'])
    if data.shape == (header['NAXIS2'], header['NAXIS1']) :
        pass
    elif data.shape == (header['NAXIS1'], header['NAXIS2']) :
        warn(f'Axis of map data swapped to match shape {dims}')
        data = data.T
    else :
        raise ValueError(f'Shape of data {data.shape} cannot match {dims}, even by swapping axes')
    
    return data, header

# Coordinates
# Unit for index is numpy index (beginning at 0)
# Unit for velocity is km/s
# Unit for frequency is GHz
# Unit for angle is ' (minute of arc)

C_LIGHT = 299792458 # In m/s

def _indices_to_coordinates(header : fits.Header, i : Union[int, np.ndarray],
    j : Union[int, np.ndarray], absolute : bool = False) :
    """ Returns the absolute coordinates in degrees of the (i,j) point in pixels (beginning at 0) """
    if header['NAXIS'] not in [2, 3] : # If header of profile
        raise ValueError("header must be the header of a cube or a map")
    if (np.array(i) < 0).any() or (np.array(j) < 0).any() :
        raise ValueError('i and j must be non-negative indices')
    if (np.array(i) >= header['NAXIS1']).any() :
        raise ValueError(f"i must be lower than {header['NAXIS1']}")
    if (np.array(j) >= header['NAXIS2']).any() :
        raise ValueError(f"j must be lower than {header['NAXIS2']}")

    if absolute :
        x = (i+1-header['CRPIX1']) * header['CDELT1'] + header['CRVAL1']
        y = (j+1-header['CRPIX2']) * header['CDELT2'] + header['CRVAL2']
    else :
        x = (i+1-header['CRPIX1']) * header['CDELT1']
        y = (j+1-header['CRPIX2']) * header['CDELT2']
    x, y = 60*x, 60*y # Conversion from degrees to minutes of arc
    return x, y

def _coordinates_to_indices(header : fits.Header, x : Union[float, np.ndarray],
    y : Union[float, np.ndarray], absolute : bool = False) :
    """ TODO """
    if header['NAXIS'] == 1 :
        raise ValueError("header must be the header of a cube or a map")
    xmin, ymin = _indices_to_coordinates(header, 0, 0, absolute = absolute)
    xmax, ymax = _indices_to_coordinates(header, header['NAXIS1']-1, header['NAXIS2']-1, absolute = absolute)
    if (np.array(x) < xmin).any() :
        raise ValueError(f"x must be greater than {xmin}'")
    if (np.array(y) < ymin).any() :
        raise ValueError(f"y must be greater than {ymin}'")
    if (np.array(x) > xmax).any() :
        raise ValueError(f"x must be lower than {xmax}'")
    if (np.array(y) > ymax).any() :
        raise ValueError(f"y must be lower than {ymax}'")

    if absolute :
        i = (x - header['CRVAL1']) / header['CDELT1'] + header['CRPIX1'] - 1
        j = (y - header['CRVAL2']) / header['CDELT2'] + header['CRPIX2'] - 1
    else :
        i = x / header['CDELT1'] + header['CRPIX1'] - 1
        j = y / header['CDELT2'] + header['CRPIX2'] - 1
    return round(i), round(j)

def _bound_coordinates(header : fits.Header, absolute : bool = False) :
    """ Returns the absolute coordinates bounds """
    xymin = _indices_to_coordinates(header, 0, 0, absolute = absolute)
    xymax = _indices_to_coordinates(header, header['NAXIS1']-1, header['NAXIS2']-1, absolute = absolute)
    xbounds = xymin[0], xymax[0]
    ybounds = xymin[1], xymax[1]
    return xbounds, ybounds

def _is_logical(data: np.ndarray) -> bool :
    """ TODO """
    return np.all((data == 0) | (data == 1))

def plot_map(data: np.ndarray, header: fits.Header, ax : Optional[plt.Axes] = None, label_unit : str = 'angle',
    no_logical = False, norm : Optional[colors.Normalize] = None, cmap : Union[str, colors.Colormap] = default_cmap,
    vmin : Optional[float] = None, vmax : Optional[float] = None) :
    """ Plots a map. Returns the figure axis and the colorbar. """
    data, header = _check_validity(data, header)

    label_unit = label_unit.lower()
    if label_unit not in ['index', 'angle'] :
        raise ValueError(f"label_unit must be 'index' or 'angle', not {label_unit}")

    if cmap is None :
        cmap = default_cmap

    imshow_kwargs = {
        'norm' : norm,
        'vmin' : vmin,
        'vmax' : vmax
    }

    if header['CTYPE1'].upper() == 'INDICES' or header['CTYPE2'].upper() == 'INDICES' :
        label_unit = 'index'

    if label_unit == 'angle' :
        bounds = _bound_coordinates(header)
        imshow_kwargs.update({
            'extent' : bounds[0] + bounds[1],
            'aspect' : data.shape[0] / data.shape[1]\
                * abs( (bounds[0][1] - bounds[0][0]) / (bounds[1][1] - bounds[1][0]) )
        })
    else :
        imshow_kwargs.update({
            'extent' : (1, header['NAXIS1'], 1, header['NAXIS2'])
        })

    if _is_logical(data) and not no_logical :
        cmap = colors.LinearSegmentedColormap.from_list(
            'binary cmap', [plt.cm.gray(0.), plt.cm.gray(1.)], 2
        )
        imshow_kwargs.update({'vmin' : 0, 'vmax' : 1})
        colorbar_kwargs = {'ticks' : [0.25, 0.75]}
    else :
        colorbar_kwargs = {}

    if ax is not None :
        im = ax.imshow(data, cmap = cmap, **imshow_kwargs)
    else :
        im = plt.imshow(data, cmap = cmap, **imshow_kwargs)
        ax = plt.gca()

    cbar = _colorbar(im, label = None, **colorbar_kwargs)
    if _is_logical(data) and not no_logical :
        cbar.ax.set_yticklabels([0, 1])

    if label_unit == 'index' :
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
    else :
        ax.set_xlabel("$\delta x$ (')")
        ax.set_ylabel("$\delta y$ (')")

    return im, ax, cbar

# LaTeX display

def latex_line(line: str, short: bool=False) -> str:
    """ Returns a printable LaTeX version of the line `line_name`.
    If `short` is True, the transition is indicated, else it isn't."""

    line = line.strip().lower()

    # Complete name
    if line == '12cn10' :
        s = '^{12}CN' if short else '^{12}CN\\,(1-0)'
    elif line == '12co10' :
        s = '^{12}CO' if short else '^{12}CO\\,(1-0)'
    elif line == '12cs21' :
        s = '^{12}CS' if short else '^{12}CS\\,(2-1)'
    elif line == '13co10' :
        s = '^{13}CO' if short else '^{13}CO\\,(1-0)'
    elif line == '32so21' :
        s = '^{32}SO' if short else '^{32}SO\\,(2-1)'
    elif line == '34so21' :
        s = '^{34}SO' if short else '^{34}SO\\,(2-1)'
    elif line == 'c3h2' :
        s = 'C_3H_2'
    elif line == 'c17o10' :
        s = 'C^{17}O' if short else 'C^{17}O\\,(1-0)'
    elif line == 'c18o10' :
        s = 'C^{18}O' if short else 'C^{18}O\\,(1-0)'
    elif line == 'ccd' :
        s = 'CCD'
    elif line == 'cch10' :
        s = 'CCH' if short else 'CCH\\,(1-0)'
    elif line == 'ccs' :
        s = 'CCS'
    elif line == 'cfp10' :
        s = 'CF^+' if short else 'CF^+\\,(1-0)'
    elif line == 'ch3oh21' :
        s = 'CH_3OH' if short else 'CH_3OH\\,(2-1)'
    elif line == 'dcn10' :
        s = 'DCN' if short else 'DCN\\,(1-0)'
    elif line == 'dcop10' :
        s = 'DCO^+' if short else 'DCO^+\\,(1-0)'
    elif line == 'dnc10' :
        s = 'DNC' if short else 'DNC\\,(1-0)'
    elif line == 'h2co' :
        s = 'H_2CO'
    elif line == 'h13cn10' :
        s = 'H^{13}CN' if short else 'H^{13}CN\\,(1-0)'
    elif line == 'h13cop10' :
        s = 'H^{13}CO^+' if short else 'H^{13}CO^+\\,(1-0)'
    elif line == 'h40alpha' :
        s = 'H^{40}_{\\alpha}'
    elif line == 'hcn10' :
        s = 'HCN' if short else 'HCN\\,(1-0)'
    elif line == 'hco' :
        s = 'HCO'
    elif line == 'hcop10' :
        s = 'HCO^+' if short else 'HCO^+\\,(1-0)'
    elif line == 'hcsp10' :
        s = 'HCS^+\\,(1-0)' if short else 'HCS^+\\,(1-0)'
    elif line == 'hn13c10' :
        s = 'HN^{13}C' if short else 'HN^{13}C\\,(1-0)'
    elif line == 'hnc10' :
        s = 'HNC' if short else 'HNC\\,(1-0)'
    elif line == 'n2dp10' :
        s = 'N_2D^+' if short else 'N_2D^+\\,(1-0)'
    elif line == 'n2hp10' :
        s = 'N_2H^+' if short else 'N_2H^+\\,(1-0)'
    elif line == 'sio21' :
        s = 'SiO' if short else 'SiO\\,(2-1)'
    else:
        # By default, returns the input without raising an error
        return line
    
    return "\\mathrm{" + s + "}"

def latex_param(param: str) -> str:
    """ Returns a printable latex version of the physical parameter `param`. """

    param = param.strip().lower()

    if param == 'g0' :
        s = 'G_0'
    elif param == 'av' :
        s = 'A_V'
    elif param == 'tkin' :
        s = 'T_{kin}'
    elif param == 'nh2' :
        s = 'N_{H_2}'
    elif param == 'density' :
        s = 'n_{H_2}'

    else:
        # By default, returns the input without raising an error
        return param

    return s


if __name__ == "__main__":

    data, header = load_integral("13co10")

    plt.figure()
    plot_map(data, header)
    plt.title(f"${latex_line('13co10')}$ integral")
    plt.show()

    data, header = load_noise_map("13co10")

    plt.figure()
    plot_map(data, header)
    plt.title(f"${latex_line('13co10', short=True)}$ noise map")
    plt.show()

    data, header = load_param("G0")
    data = np.log10(data)

    plt.figure()
    plot_map(data, header)
    plt.title(f"${latex_param('g0')}$ map")
    plt.show()

    data = data > np.percentile(data, 95)

    plt.figure()
    plot_map(data, header)
    plt.title(f"95-th percentile of ${latex_param('g0')}$")
    plt.show()
