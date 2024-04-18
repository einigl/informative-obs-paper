import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from beetroots.modelling.forward_maps.neural_network_approx import NeuralNetworkApprox
from beetroots.space_transform.transform import MyScaler
from nnbma import NeuralNetwork

import infovar

from ism_lines_helpers import (
    molecule_and_transition,
    molecule,
    transition,
    is_line_of,
    filter_molecules,
    molecules_among_lines,
    molecule_to_latex,
    transition_to_latex,
    line_to_latex,
    remove_hyperfine,
    is_hyperfine,
    Settings
)

# Environments

def get_physical_env(
    name: str
):
    """
    TODO
    """

    conv_fact = 1.2786 / 2 # From radm to G0 (page 26 man. Pierre)

    if name == "all":
        return {
            "P": [1.0e+5, 1.0e+9],
            "radm": [1.0e+0, 1.0e+5],
            "Avmax": [1.0e+0, 4.0e+1],
            "angle": [0.0, 0.0]
        }

    if name == "horsehead":
        return {
            "P": [10**5.0, 10**6.5], # [10**5, 10**5]
            "radm": [10**1.0 / conv_fact, 10**2.4 / conv_fact],
            # "Avmax": [10**0.2, 10**1.4],
            "Avmax": [1, 25],
            "angle": [0.0, 0.0]
        }

# Helpers
    
def convert_unit(T, nu, dv):
    """
    [T]: K
    [nu]: Hz
    [dv]: km.s^-1
    """
    kb = 1.380_649e-23 # m^2.kg.s^-2.K^-1
    c = 299_792_458 # m.s^-1
    return (2 * 1e6 * (nu/c)**3 * kb) * T * dv

def integrate_channels(x, n):
    """
    x: intensity for a single velocity channel
    n: number of velocity channel to integrate
    """
    return n**0.5 * x


# Sampling
    
def _bounded_power_law(
    n_samples: int,
    bounds: Tuple[float, float],
    alpha: float=1.
):
    xmin, xmax = bounds
    beta = alpha #1 - alpha
    a = xmin**beta
    b = xmax**beta - a

    x = np.random.rand(n_samples)
    return (a + b*x)**(1/beta)


def params_sampling(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    seed: Optional[int]=None
) -> pd.DataFrame:
    """
    TODO
    """

    # Init

    rng = np.random.default_rng(seed=seed)

    # Sampling

    dict_is_log_scale_params = {
        "P": True,
        "radm": True,
        "Avmax": True,
        "angle": False,
    }

    dict_power_law_params = {
        "P": 1,
        "radm": -1.05,
        "Avmax": -2.24,
        "angle": None
    }

    param_names = ["P", "radm", "Avmax", "angle"]

    Theta = [None] * len(param_names)

    for i, name in enumerate(param_names):
        low, upp = bounds[name]
        alpha = dict_power_law_params[name]
        if dict_is_log_scale_params[name]:
            Theta[i] = _bounded_power_law(
                n_samples, (low, upp), alpha
            )
            # Theta[i] = 10**rng.uniform(
            #     np.log10(low), np.log10(upp), size=n_samples
            # ) # TODO : laisser le choix du prior
        else:
            Theta[i] = rng.uniform(
                low, upp, size=n_samples
            )

    return pd.DataFrame(np.column_stack(Theta), columns=param_names)
        

def _add_noise(
    df: pd.DataFrame,
    emir_df: pd.DataFrame,
    obs_time: float,
) -> pd.DataFrame:
    """
    Add noise to Meudon PDR prediction intensities
    """
    n_params = 5

    # EMIR lines
    n_channels = 20 # Already taken in account in selection.py
    dv = 0.5 # km/s # Already taken in account in selection.py
    linewidth = 10 # km/s # Already taken in account in selection.py

    # CI, CI, C+
    freqs_additional = [492*1e9, 809*1e9, 1910*1e9] # Hz
    sigma_a_additional = [0.5, 0.5, 2.25] # T (per channel)
    dv_additional = [1, 1, 0.193] # km/s
    n_channels_additional = [linewidth / _dv for _dv in dv_additional]

    sigma_a = emir_df["Noise RMS (Mathis units, log10) [1 min]"].to_list()\
        + [np.log10(convert_unit(integrate_channels(T, n), nu, dv))\
           for T, n, nu, dv in zip(sigma_a_additional, n_channels_additional, freqs_additional, dv_additional)]
    
    sigma_a = 10**np.array(sigma_a) / obs_time**0.5 # Atmospheric error

    percent = emir_df["Calibration error (%)"].to_list() + [20, 20, 5]
    sigma_m = np.log(1 + np.array(percent)/100) # Calibration error

    size = (df.shape[0], df.shape[1] - n_params)

    eps_a = np.random.normal(loc=0., scale=sigma_a, size=size)
    eps_m = np.random.lognormal(mean=-(sigma_m**2) / 2, sigma=sigma_m, size=size)

    y = eps_m * df.iloc[:, n_params:].values + eps_a

    df_noise = pd.concat([
        df.iloc[:, :n_params],
        pd.DataFrame(y, columns=df.columns[n_params:])
    ], axis=1)

    return df_noise


# PDR Code emulator

def pdr_model(
    df_params: pd.DataFrame,
    obs_time: float,
    noise: bool=True,
    kappa: Union[float, Tuple[float, float]]=1.,
) -> pd.DataFrame:
    """
    TODO
    """

    # Init

    model_name = "meudon_pdr_model_dense"
    path_model = os.path.join(os.path.dirname(__file__), "data", "models")
    
    # Load neural network
    
    net = NeuralNetwork.load(
        model_name,
        path_model
    )
   
    # Reference dataframe

    emir_df = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "data", "raw", "emir_lines_selection", "emir_table_filtered.csv"
    ))

    emir_lines = emir_df["line_id"].to_list()
    additional_lines = ["c_el3p_j1__el3p_j0", "c_el3p_j2__el3p_j1", "cp_el2p_j3_2__el2p_j1_2"] # 2 lines of CI, 1 line of C+

    # Restrictions

    net.restrict_to_output_subset(
        emir_lines + additional_lines
    )

    # Reorder inputs

    df_params = df_params[net.inputs_names]

    # PDR code predictions

    Y = 10**net.evaluate(df_params.values, transform_inputs=True)

    # Apply kappa

    if isinstance(kappa, (int, float)):
        kappa = (kappa, kappa)
    _kappa = 10**np.random.uniform(np.log10(kappa[0]), np.log10(kappa[1]))

    df = pd.DataFrame(
        np.hstack((df_params.values, _kappa * np.ones((Y.shape[0], 1)), _kappa * Y)),
        columns=net.inputs_names + ["kappa"] + net.current_output_subset
    )

    if not noise:
        return df
    
    return _add_noise(
        df, emir_df, obs_time
    )


def simulate(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    obs_time: float,
    noise: bool=True,
    kappa: Union[float, Tuple[float, float]]=1.,
    seed: Optional[int]=None,
) -> pd.DataFrame:
    """
    TODO
    """

    # Parameters sampling

    df_params = params_sampling(
        n_samples,
        bounds,
        seed=seed
    )

    # PDR code predictions

    return pdr_model(
        df_params,
        obs_time=obs_time,
        noise=noise,
        kappa=kappa
    )


# Getter

class PDRGetter(infovar.StandardGetter):

    def __init__(
        self,
        n_samples: int,
        env: str,
        obs_time: float,
        noise: bool=True,
        kappa: Union[float, Tuple[float, float]]=1.,
        seed: Optional[int]=None
    ):
        # Load and preprocess data

        bounds = get_physical_env(env)

        df = simulate(
            n_samples,
            bounds,
            obs_time,
            noise=noise,
            kappa=kappa,
            seed=seed
        )

        # Attributes
        n_params = 5 # P, radm, Avmax, angle, kappa
        self.x_names = df.columns[n_params:].to_list()
        self.y_names = df.columns[:n_params].to_list()
        self.x = df.values[:, n_params:]
        self.y = df.values[:, :n_params]


# LaTeX display

Settings.math_mode = False

def latex_line(line: str, short: bool=False) -> str:
    """ Returns a printable LaTeX version of the line `line_name` (without degenerate energy levels).
    If `short` is True, the transition is indicated, else it isn't."""

    if short:
        return molecule_to_latex(molecule(line))
    return line_to_latex(remove_hyperfine(line))

def latex_param(param: str) -> str:
    """ Returns a printable latex version of the physical parameter `param`. """

    param = param.strip().lower()

    if param == 'radm' :
        s = 'G_{UV}'
        # s = 'G_0'
        # s = 'G_{\\text{UV}}'
    elif param == 'avmax' :
        s = 'A_V'
        # s = 'A_{\\text{V}}'
    elif param == 'p' :
        s = 'P_{th}'
        # s = 'P_{\\text{th}}'
    elif param == 'kappa' :
        s = '\\kappa'
        # s = 'P_{\\text{th}}'
    else:
        # By default, returns the input without raising an error
        return param

    return s