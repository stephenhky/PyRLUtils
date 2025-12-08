
from typing import Optional
import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .utils import DiscreteState, DiscreteStateValueType


class DiscreteCategoricalState(DiscreteState):
    def __init__(
            self,
            all_state_values: list[DiscreteStateValueType],
            initial_value: Optional[DiscreteStateValueType] = None,
            terminals: Optional[dict[DiscreteStateValueType, bool]]=None
    ):
        super().__init__()
        self._all_state_values = all_state_values
        self._state_values_to_indices = {
            state_value: idx
            for idx, state_value in enumerate(self._all_state_values)
        }
        if initial_value is not None:
            self._current_index = self._state_values_to_indices[initial_value]
        else:
            self._current_index = 0
        if terminals is None:
            self._terminal_dict = {
                state_value: False
                for state_value in self._all_state_values
            }
        else:
            self._terminal_dict = terminals.copy()
            for state_value in self._all_state_values:
                if self._terminal_dict.get(state_value) is None:
                    self._terminal_dict[state_value] = False

    def _get_state_value_from_index(self, index: int) -> DiscreteStateValueType:
        return self._all_state_values[index]

    def get_state_value(self) -> DiscreteStateValueType:
        return self._get_state_value_from_index(self._current_index)

    def set_state_value(self, state_value: DiscreteStateValueType) -> None:
        if state_value in self._all_state_values:
            self._current_index = self._state_values_to_indices[state_value]
        else:
            raise ValueError('State value {} is invalid.'.format(state_value))

    def get_all_possible_state_values(self) -> list[DiscreteStateValueType]:
        return self._all_state_values

    def query_state_index_from_value(self, value: DiscreteStateValueType) -> int:
        return self._state_values_to_indices[value]

    @property
    def state_index(self) -> int:
        return self._current_index

    @state_index.setter
    def state_index(self, new_index: int) -> None:
        if new_index >= len(self._all_state_values):
            raise ValueError(f"Invalid index {new_index}; it must be less than {self.nb_state_values}.")
        self._current_index = new_index

    @property
    def state_space_size(self):
        return len(self._all_state_values)

    @property
    def nb_state_values(self) -> int:
        return len(self._all_state_values)

    @property
    def is_terminal(self) -> bool:
        return self._terminal_dict[self._all_state_values[self._current_index]]

    def __hash__(self):
        return self._current_index

    def __eq__(self, other: Self) -> bool:
        return self._current_index == other._current_index
