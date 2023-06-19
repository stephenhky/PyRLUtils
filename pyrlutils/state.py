
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class State(ABC):
    @abstractmethod
    def set_state_value(self, state_value):
        pass

    @abstractmethod
    def get_state_value(self):
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
    def __init__(self, nbdims: int, ranges: List[np.array], init_value: Optional[float, np.ndarray] = None):
        self._nbdims = nbdims

        try:
            assert (ranges.dtype == np.float64) or (ranges.dtype == np.float32) or (ranges.dtype == np.float16)
        except AssertionError:
            raise TypeError('It has to be floating type numpy.ndarray.')

        try:
            assert self._nbdims > 0
        except AssertionError:
            raise ValueError('Number of dimensions must be positive.')

        if self._nbdims > 1:
            try:
                assert self._nbdims == ranges.shape[0]
            except AssertionError:
                raise ValueError('Number of ranges does not meet the number of dimensions.')

        if self._nbdims > 1:
            try:
                for i in range(ranges.shape[0]):
                    assert ranges[i, 0] <= ranges[i, 1]
            except AssertionError:
                raise InvalidRangeError()
        else:
            try:
                assert ranges[0] <= ranges[1]
            except AssertionError:
                raise InvalidRangeError()

        self._ranges = ranges if self._nbdims > 1 else np.expand_dims(ranges, axis=0)
        if init_value is None:
            self._state_value = np.zeros(self._nbdims)
            for i in range(self._nbdims):
                self._state_value[i] = np.random.uniform(self._ranges[i, 0], self._ranges[i, 1])
        else:
            try:
                assert init_value.shape[0] == self._nbdims
            except AssertionError:
                raise ValueError('Initialized value does not have the right dimension.')
            for i in range(self._nbdims):
                try:
                    assert (init_value[i] >= self._ranges[i, 0]) and (init_value[i] <= self.ranges[i, 1])
                except AssertionError:
                    raise InvalidRangeError('Initialized value at dimension {} (value: {}) is not within the permitted range ({} -> {})!'.format(i, init_value[i], self._ranges[i, 0], self._ranges[i, 1]))
            self._state_value = init_value

    def set_state_value(self, state_value: np.ndarray):
        try:
            assert state_value.shape[0] == self._nbdims
        except AssertionError:
            raise ValueError('Given value does not have the right dimension.')
        self._state_value = state_value

    def get_state_value(self) -> np.ndarray:
        return self._state_value

    def get_state_value_ranges(self) -> np.ndarray:
        return self._ranges

    def get_state_value_range_at_dimension(self, dimension: int) -> np.ndarray:
        return self._ranges[dimension]

    @property
    def ranges(self) -> np.ndarray:
        return self.get_state_value_ranges()

    @property
    def state_value(self) -> np.ndarray:
        return self.get_state_value()

    @property
    def nbdims(self) -> int:
        return self._nbdims


