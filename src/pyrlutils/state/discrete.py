"""
Discrete state implementations for reinforcement learning.
"""

from typing import Optional
import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .utils import DiscreteState, DiscreteStateValueType


class DiscreteCategoricalState(DiscreteState):
    """
    A discrete state that can take on a finite number of categorical values.
    """

    def __init__(
            self,
            all_state_values: list[DiscreteStateValueType],
            initial_value: Optional[DiscreteStateValueType] = None,
            terminals: Optional[dict[DiscreteStateValueType, bool]]=None
    ):
        """
        Initialize the discrete categorical state.

        Args:
            all_state_values: A list of all possible state values.
            initial_value: The initial state value (defaults to the first value in all_state_values).
            terminals: A dictionary mapping state values to boolean indicating if they are terminal states.
                       If None, all states are considered non-terminal.
        """
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
        """
        Get the state value corresponding to an index.

        Args:
            index: The index in the all_state_values list.

        Returns:
            The state value at the given index.
        """
        return self._all_state_values[index]

    def get_state_value(self) -> DiscreteStateValueType:
        """
        Get the current state value.

        Returns:
            The current state value.
        """
        return self._get_state_value_from_index(self._current_index)

    def set_state_value(self, state_value: DiscreteStateValueType) -> None:
        """
        Set the current state value.

        Args:
            state_value: The state value to set.

        Raises:
            ValueError: If the state_value is not in the list of all possible state values.
        """
        if state_value in self._all_state_values:
            self._current_index = self._state_values_to_indices[state_value]
        else:
            raise ValueError('State value {} is invalid.'.format(state_value))

    def get_all_possible_state_values(self) -> list[DiscreteStateValueType]:
        """
        Get all possible state values.

        Returns:
            A list of all possible state values.
        """
        return self._all_state_values

    def query_state_index_from_value(self, value: DiscreteStateValueType) -> int:
        """
        Get the index corresponding to a state value.

        Args:
            value: The state value to query.

        Returns:
            The index of the state value in the all_state_values list.
        """
        return self._state_values_to_indices[value]

    @property
    def state_index(self) -> int:
        """
        Get the current state index.

        Returns:
            The current state index.
        """
        return self._current_index

    @state_index.setter
    def state_index(self, new_index: int) -> None:
        """
        Set the current state index.

        Args:
            new_index: The new state index to set.

        Raises:
            ValueError: If the new_index is out of bounds.
        """
        if new_index >= len(self._all_state_values):
            raise ValueError(f"Invalid index {new_index}; it must be less than {self.nb_state_values}.")
        self._current_index = new_index

    @property
    def state_space_size(self):
        """
        Get the size of the state space.

        Returns:
            The number of possible state values.
        """
        return len(self._all_state_values)

    @property
    def nb_state_values(self) -> int:
        """
        Get the number of state values.

        Returns:
            The number of possible state values.
        """
        return len(self._all_state_values)

    @property
    def is_terminal(self) -> bool:
        """
        Check if the current state is terminal.

        Returns:
            True if the current state is terminal, False otherwise.
        """
        return self._terminal_dict[self._all_state_values[self._current_index]]

    def __hash__(self):
        """
        Hash the state based on its current index.

        Returns:
            The hash of the state.
        """
        return self._current_index

    def __eq__(self, other: Self) -> bool:
        """
        Check if two states are equal based on their current index.

        Args:
            other: Another state to compare with.

        Returns:
            True if the states are equal, False otherwise.
        """
        return self._current_index == other._current_index