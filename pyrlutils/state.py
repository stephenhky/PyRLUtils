
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union

import numpy as np


class State(ABC):
    @property
    def state_value(self):
        return self.get_state_value()

    @abstractmethod
    def set_state_value(self, state_value):
        pass

    @abstractmethod
    def get_state_value(self):
        pass

    @state_value.setter
    def state_value(self, new_state_value):
        self.set_state_value(new_state_value)


DiscreteStateValueType = Union[float, str, Tuple[int]]


class DiscreteState(State):
    def __init__(self, all_state_values: List[DiscreteStateValueType], initial_values: Optional[List[DiscreteStateValueType]] = None):
        super().__init__()
        self._all_state_values = all_state_values
        self._state_value = initial_values if initial_values is not None and initial_values in self._all_state_values else self._all_state_values[0]

    def get_state_value(self) -> DiscreteStateValueType:
        return self._state_value

    def set_state_value(self, state_value: DiscreteStateValueType):
        if state_value in self._all_state_values:
            self._state_value = state_value
        else:
            raise ValueError('State value {} is invalid.'.format(state_value))

    def get_all_possible_state_values(self) -> List[DiscreteStateValueType]:
        return self._all_state_values

    @property
    def state_value(self) -> DiscreteStateValueType:
        return self._state_value

    @state_value.setter
    def state_value(self, new_state_value: DiscreteStateValueType):
        self.set_state_value(new_state_value)

    @property
    def state_space_size(self):
        return len(self._all_state_values)


class InvalidRangeError(Exception):
    def __init__(self, message=None):
        self.message = "Invalid range error!" if message is None else message
        super().__init__(self.message)


class ContinuousState(State):
    def __init__(self, nbdims: int, ranges: np.array, init_value: Optional[Union[float, np.ndarray]] = None):
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
            try:
                assert ranges.shape[1] == 2
            except AssertionError:
                raise ValueError("Only the smallest and largest values in `ranges'.")
        else:
            try:
                assert ranges.shape[0] == 2
            except AssertionError:
                raise ValueError("Only the smallest and largest values in `ranges'.")

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
            if self._nbdims > 1:
                try:
                    assert init_value.shape[0] == self._nbdims
                except AssertionError:
                    raise ValueError('Initialized value does not have the right dimension.')
                for i in range(self._nbdims):
                    try:
                        assert (init_value[i] >= self._ranges[i, 0]) and (init_value[i] <= self.ranges[i, 1])
                    except AssertionError:
                        raise InvalidRangeError('Initialized value at dimension {} (value: {}) is not within the permitted range ({} -> {})!'.format(i, init_value[i], self._ranges[i, 0], self._ranges[i, 1]))
            else:
                try:
                    assert (init_value >= self._ranges[0, 0]) and (init_value <= self.ranges[0, 1])
                except AssertionError:
                    raise InvalidRangeError('Initialized value is out of range.')
            self._state_value = init_value

    def set_state_value(self, state_value: Union[float, np.ndarray]):
        if self.nbdims > 1:
            try:
                assert state_value.shape[0] == self._nbdims
            except AssertionError:
                raise ValueError('Given value does not have the right dimension.')
            for i in range(self.nbdims):
                try:
                    assert state_value[i] >= self.ranges[i, 0] and state_value[i] <= self.ranges[i, 1]
                except AssertionError:
                    raise InvalidRangeError()
        else:
            try:
                assert state_value >= self.ranges[0, 0] and state_value <= self.ranges[0, 1]
            except AssertionError:
                raise InvalidRangeError()

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
    def state_value(self) -> Union[float, np.ndarray]:
        return self.get_state_value()

    @state_value.setter
    def state_value(self, new_state_value):
        self.set_state_value(new_state_value)

    @property
    def nbdims(self) -> int:
        return self._nbdims


class Discrete2DCartesianState(DiscreteState):
    def __init__(self, x_lowlim: int, x_hilim: int, y_lowlim: int, y_hilim: int, initial_coordinate: List[int]=None):
        self._x_lowlim = x_lowlim
        self._x_hilim = x_hilim
        self._y_lowlim = y_lowlim
        self._y_hilim = y_hilim
        self._countx = self._x_hilim - self._x_lowlim + 1
        self._county = self._y_hilim - self._y_lowlim + 1
        if initial_coordinate is None:
            initial_coordinate = [self._x_lowlim, self._y_lowlim]
        initial_value =  (initial_coordinate[1] - self._y_lowlim) * self._countx + (initial_coordinate[0] - self._x_lowlim)
        super().__init__(list(range(self._countx*self._county)), initial_values=initial_value)

    def _encode_coordinates(self, x, y) -> int:
        return (y - self._y_lowlim) * self._countx + (x - self._x_lowlim)

    def encode_coordinates(self, coordinates: List[int]) -> int:
        assert len(coordinates) == 2
        return self._encode_coordinates(coordinates[0], coordinates[1])

    def decode_coordinates(self, hashcode) -> List[int]:
        return [hashcode % self._countx, hashcode // self._countx]
