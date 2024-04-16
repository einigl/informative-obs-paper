from typing import List, Dict, Tuple, Union, Optional, Callable

import numpy as np

# assert set(bounds) == set(win_sizes)
# assert (overlap is not None) ^ (n_sections is not None)
# if overlap is not None:
#     assert set(bounds) == set(overlap)
# if n_sections is not None:
#     assert set(bounds) == set(n_sections)

import infovar

class Grid:

    def __init__(
        self,
        params: Union[str, List[str]],
        win_params: Union[str, List[str]],
        lines: List[Union[str, List[str]]],
    ):
        if not isinstance(params, (str, List)):
            raise TypeError(f"params must be str or List[str], not {type(params)}")
        if isinstance(params, str):
            params = [params]
        elif isinstance(params, List) and not all([isinstance(p, str) for p in params]):
            raise TypeError(f"params elements must be str")

        if not isinstance(win_params, (str, List)):
            raise TypeError(f"params must be str or List[str], not {type(win_params)}")
        if isinstance(win_params, str):
            win_params = [win_params]
        elif isinstance(win_params, List) and not all([isinstance(p, str) for p in win_params]):
            raise TypeError(f"params elements must be str")

        if not isinstance(lines, (str, List)):
            raise TypeError(f"lines must be str or List[str], not {type(params)}")
        if isinstance(lines, str):
            lines = [lines]
        elif isinstance(lines, List) and not all([isinstance(l, str) for l in lines]):
            raise TypeError(f"lines elements must be str")
        
        self.params = params
        self.win_params = win_params
        self.lines = lines

        self.path = None
        self.metric = None
        self.n_blocks = None
        self.dims = None
        
    def load_data(
        self,
        path: str,
        stat: str,
    ) -> None:
        handler = infovar.ContinuousHandler()
        handler.set_paths(path)

        self.data_list = [None] * self.lines
        for i, lines in enumerate(self.lines):
            d = handler.read(
                lines,
                self.params,
                self.win_params,
            )

            idx = [d["features"].index(p) for p in self.win_params]

            data = d["stats"][stat]["data"]
            data = np.moveaxis(data, idx, list(len(idx)))
            self.data_list[i] = data.flatten()

            if self.dims is None:
                self.n_blocks = data.size
                self.dims = data.shape
                self.centers = [d["stats"][stat]["center"][k] for k in idx]
                self.lowers = [d["stats"][stat]["lower"][k] for k in idx]
                self.uppers = [d["stats"][stat]["upper"][k] for k in idx]

    def setup(
        self,
        rule: Callable[[List[float]], List[int]]
    ) -> None:
        # TODO
        self.selected_lines = ...

    def get_indices(
        self,
        values: List[float]
    ) -> List[List[int]]:
        """
        Returns TODO
        """
        return [np.where(np.logical_and((self.lowers[i] <= v) & (v <=self.uppers[i])))[0].tolist() for i, v in enumerate(values)]
    
    # TODO: Il faut rajouter lower bounds et upper bounds dans ContinuousHandler

    def get(
        self,
        indices: List[int]
    ) -> List[List[str]]:
        self.

# Faut-il faire plutôt un get_block ?