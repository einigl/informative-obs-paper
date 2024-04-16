import numpy as np

import infovar


# Additional handler stats

class LinearInfoLog10(infovar.Statistic):
    """
    TODO
    """

    def __init__(self):
        self.cc = infovar.LinearInfo()

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        _variables = np.log10(np.abs(variables))
        _targets = np.log10(np.abs(targets))

        return self.cc(_variables, _targets)
