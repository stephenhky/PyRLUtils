
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np


class State(ABC):
    pass


class DiscreteState(ABC, State, Enum):
    pass


class InvalidRangeError(Exception):
    def __init__(self):
        self.message = "Invalid range error!"
        super().__init__(self.message)


class ContinuousState(ABC, State):
    def __init__(self, nbdims: int, ranges: List[np.array]):
        self.nbdims = nbdims

        try:
            assert (ranges.dtype == np.float64) or (ranges.dtype == np.float32) or (ranges.dtype == np.float16)

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


class StateSpace(ABC):
    @abstractmethod
    def get_state(self, key) -> State:
        pass


class DiscreteStateSpace(ABC, StateSpace):
    pass
