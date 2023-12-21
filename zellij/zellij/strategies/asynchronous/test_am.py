# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T14:51:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import AMetaheuristic

import torch
import numpy as np


class Test(AMetaheuristic):
    def __init__(self, search_space, verbose=True):
        super().__init__(search_space, verbose)
        self._counter = 0
        self.state = 0

    def forward(self, X, Y):
        if self._counter < 5:
            self._counter += 1
            return [[self.rank]], {
                "rank": self.rank,
                "state": self.state,
                "counter": self._counter,
            }
        else:
            self.state += 5
            return [], {
                "rank": self.rank,
                "state": self.state,
                "counter": self._counter,
            }

    def next_state(self, state):
        if state:
            return np.array(state) + 1
        else:
            return np.array(list(range(1, 1000, 100)))

    def update_state(self, state):
        self._counter = 0
        self.state = state

    def get_state(self) -> object:
        return self.state
