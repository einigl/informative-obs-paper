from typing import List, Optional

import numpy as np
from nnbma import FullyConnected

from .grid import Grid

class Regressor:

    def __init__(
        self,
        grid: Grid,
        hidden_layer_sizes: List[int],

    ):
        self.grid = grid
        self.hidden_layer_sizes = hidden_layer_sizes
        self.networks = [
            FullyConnected()
        ]
 
# Regressor doit être une Grid où chaque case est un réseau de neurones.
# Regressor délègue à grid la détermination des indices concernés (avec get).

# TODO: faire le regressor avant la grille pour être certain de ce que l'on veut !
