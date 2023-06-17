
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class State(ABC):
    pass


class DiscreteState(State):
    @abstractmethod
    def get_all_possible_state_values(self) -> List:
        raise NotImplementedError()


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
            raise ValueError('It has to be floating type numpy.ndarray.')

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
