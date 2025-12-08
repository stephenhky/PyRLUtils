
import sys
from typing import Union, Annotated, Literal, Optional

import numpy as np
from numpy.typing import NDArray
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from ..helpers.exceptions import InvalidRangeError
from .utils import State


class ContinuousState(State):
    def __init__(
            self,
            nbdims: int,
            ranges: Union[Annotated[NDArray[np.float64], Literal["2"]], Annotated[NDArray[np.float64], Literal["*", "2"]]],
            init_value: Optional[Union[float, Annotated[NDArray[np.float64], "1D Array"]]] = None
    ):
        super().__init__()
        self._nbdims = nbdims

        try:
            assert isinstance(ranges, np.ndarray)
        except AssertionError:
            raise TypeError('Range must be a numpy array.')

        try:
            assert (ranges.dtype == np.float64) or (ranges.dtype == np.float32) or (ranges.dtype == np.float16)
        except AssertionError:
            raise TypeError('It has to be floating type numpy.ndarray.')

        try:
            assert ranges.ndim == 1 or ranges.ndim == 2
            match ranges.ndim:
                case 1:
                    assert ranges.shape[0] == 2
                case 2:
                    assert ranges.shape[1] == 2
                case _:
                    raise ValueError("Ranges must be of shape (2, ) or (*, 2).")
        except AssertionError:
            raise ValueError("Ranges must be of shape (2, ) or (*, 2).")

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
                        assert self._ranges[i, 0] <= init_value[i] <= self.ranges[i, 1]
                    except AssertionError:
                        raise InvalidRangeError('Initialized value at dimension {} (value: {}) is not within the permitted range ({} -> {})!'.format(i, init_value[i], self._ranges[i, 0], self._ranges[i, 1]))
            else:
                try:
                    assert self._ranges[0, 0] <= init_value <= self.ranges[0, 1]
                except AssertionError:
                    raise InvalidRangeError('Initialized value is out of range.')
            self._state_value = init_value

    def set_state_value(self, state_value: Union[float, Annotated[NDArray[np.float64], "1D Array"]]):
        if self._nbdims > 1:
            try:
                assert state_value.shape[0] == self._nbdims
            except AssertionError:
                raise ValueError('Given value does not have the right dimension.')
            for i in range(self._nbdims):
                try:
                    assert self.ranges[i, 0] <= state_value[i] <= self.ranges[i, 1]
                except AssertionError:
                    raise InvalidRangeError()
        else:
            try:
                assert self.ranges[0, 0] <= state_value <= self.ranges[0, 1]
            except AssertionError:
                raise InvalidRangeError()

        self._state_value = state_value

    def get_state_value(self) -> Annotated[NDArray[np.float64], "1D Array"]:
        return self._state_value

    def get_state_value_ranges(self) -> Union[Annotated[NDArray[np.float64], Literal["2"]], Annotated[NDArray[np.float64], Literal["*", "2"]]]:
        return self._ranges

    def get_state_value_range_at_dimension(self, dimension: int) -> Annotated[NDArray[np.float64], Literal["2"]]:
        if dimension < self._nbdims:
            return self._ranges[dimension]
        else:
            raise ValueError(f"There are only {self._nbdims} dimensions!")

    @property
    def ranges(self) -> Union[Annotated[NDArray[np.float64], Literal["2"]], Annotated[NDArray[np.float64], Literal["*", "2"]]]:
        return self.get_state_value_ranges()

    @property
    def state_value(self) -> Union[float, NDArray[np.float64]]:
        return self.get_state_value()

    @state_value.setter
    def state_value(self, new_state_value):
        self.set_state_value(new_state_value)

    @property
    def nbdims(self) -> int:
        return self._nbdims

    def __hash__(self):
        return hash(tuple(self._state_value))

    def __eq__(self, other: Self):
        if self.nbdims != other.nbdims:
            raise ValueError(f"The two states have two different dimensions. ({self.nbdims} vs. {other.nbdims})")
        for i in range(self.nbdims):
            if self.state_value[i] != other.state_value[i]:
                return False
        return True
