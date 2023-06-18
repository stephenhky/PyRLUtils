
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class State(ABC):
    pass


class DiscreteState(State):
    def __init__(self, all_state_values: List, initial_values: Optional[List] = None):
        super().__init__()
        self._all_state_values = all_state_values
        self._state_value = initial_values if initial_values is not None and initial_values in self._all_state_values else self._all_state_values[0]

    def get_state_value(self):
        return self._state_value

    @property
    def state_value(self):
        return self.get_state_value()

    def set_state_value(self, state_value):
        if state_value in self._all_state_values:
            self._state_value = state_value
        else:
            raise ValueError('State value {} is invalid.'.format(state_value))

    def get_all_possible_state_values(self) -> List:
        return self._all_state_values


class InvalidRangeError(Exception):
    def __init__(self):
        self.message = "Invalid range error!"
        super().__init__(self.message)


class ContinuousState(State):
    def __init__(self, nbdims: int, ranges: List[np.array]):
        self.nbdims = nbdims

        try:
            assert (ranges.dtype == np.float64) or (ranges.dtype == np.float32) or (ranges.dtype == np.float16)
        except AssertionError:
            raise TypeError('It has to be floating type numpy.ndarray.')

        try:
            assert self.nbdims == ranges.shape[0]
        except AssertionError:
            raise ValueError('Number of ranges does not meet the number of dimensions.')

        try:
            for i in range(ranges.shape[0]):
                assert ranges[i, 0] <= ranges[i, 1]
        except AssertionError:
            raise InvalidRangeError()

        self.ranges = ranges

    def get_state_value_ranges(self):
        return self.ranges
