
from typing import Union, Optional, Annotated, Literal

import numpy as np
from deprecation import deprecated
from numpy.typing import NDArray

from .utils import DiscreteState


def normalize_cartesian_coordinates(
        coordinates: Union[list[int], NDArray[np.int64], tuple[int, ...]],
        nbdims: int
) -> NDArray[np.int64]:
    if isinstance(coordinates, np.ndarray) and coordinates.shape == (nbdims,):
        return coordinates
    elif (isinstance(coordinates, list) or isinstance(coordinates, tuple)) and len(coordinates) == nbdims and all(map(lambda num: isinstance(num, int), coordinates)):
        return np.array(coordinates, dtype=np.int64)
    else:
        raise TypeError("Given coordinates type not allowed!")


class Discrete2DCartesianState(DiscreteState):
    def __init__(
            self,
            x_lowlim: int,
            x_hilim: int,
            y_lowlim: int,
            y_hilim: int,    # np.int is not allowed
            initial_coordinate: Optional[Union[list[int], Annotated[NDArray[np.int64], Literal["2"]]]]=None,
            terminal_state_values: Optional[list[Union[list[int], Annotated[NDArray[np.int64], Literal["2"]]]]] = None
    ):
        self._x_lowlim = x_lowlim
        self._x_hilim = x_hilim
        self._y_lowlim = y_lowlim
        self._y_hilim = y_hilim
        self._countx = self._x_hilim - self._x_lowlim + 1
        self._county = self._y_hilim - self._y_lowlim + 1
        if initial_coordinate is None:
            self._state_value = np.array([self._x_lowlim, self._y_lowlim], dtype=np.int64)
        else:
            self._state_value = normalize_cartesian_coordinates(initial_coordinate, 2)
        self._terminal_dict = {}
        if terminal_state_values is not None:
            for terminal_coordinates in terminal_state_values:
                self.set_terminal_given_coordinates(terminal_coordinates)

    def get_state_value(self) -> NDArray[np.int64]:
        return self._state_value

    def set_state_value(self, val: Union[list[int], Annotated[NDArray[np.int64], Literal["2"]], tuple[int, int]]) -> None:
        self._state_value = normalize_cartesian_coordinates(val, 2)

    def get_whether_terminal_given_coordinates(
            self,
            coordinates: Union[list[int], Annotated[NDArray[np.int64], Literal["2"]], tuple[int, int]]
    ) -> bool:
        normalized_coordinates = normalize_cartesian_coordinates(coordinates, 2)
        return self._terminal_dict.get(tuple(normalized_coordinates), False)

    def set_terminal_given_coordinates(
            self,
            coordinates: Union[list[int], Annotated[NDArray[np.int64], Literal["2"]], tuple[int, int]],
            terminal_value: bool
    ) -> None:
        normalized_coordinates = normalize_cartesian_coordinates(coordinates, 2)
        tupled_normalized_coordinates = tuple(int(num) for num in normalized_coordinates)
        self._terminal_dict[tupled_normalized_coordinates] = terminal_value

    @property
    def x_lowlim(self) -> int:
        return self._x_lowlim

    @property
    def x_hilim(self) -> int:
        return self._x_hilim

    @property
    def y_lowlim(self) -> int:
        return self._y_lowlim

    @property
    def y_hilim(self) -> int:
        return self._y_hilim

    @property
    def state_space_size(self) -> int:
        return self._countx * self._county

    @property
    def is_terminal(self) -> bool:
        return self.get_whether_terminal_given_coordinates(self.get_state_value())

    @deprecated(deprecated_in="0.2.0", removed_in="0.3.0", details="No longer encoding coordinates")
    def _encode_coordinates(self, x, y) -> int:
        return (y - self._y_lowlim) * self._countx + (x - self._x_lowlim)

    @deprecated(deprecated_in="0.2.0", removed_in="0.3.0", details="No longer encoding coordinates")
    def encode_coordinates(self, coordinates: Union[list[int], Annotated[NDArray[np.int64], Literal["2"]], tuple[int, int]]) -> int:
        if isinstance(coordinates, list):
            assert len(coordinates) == 2
        return self._encode_coordinates(coordinates[0], coordinates[1])

    @deprecated(deprecated_in="0.2.0", removed_in="0.3.0", details="No longer encoding coordinates")
    def decode_coordinates(self, hashcode) -> list[int]:
        return [hashcode % self._countx + self._x_lowlim, hashcode // self._countx + self._y_lowlim]
