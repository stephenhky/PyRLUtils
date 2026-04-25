"""
State utility classes and type definitions for reinforcement learning.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Any


class State(ABC):
    """
    Abstract base class for state representations.
    """

    @property
    @abstractmethod
    def state_value(self):
        """
        Get the current state value.

        Returns:
            The current state value.
        """
        raise NotImplemented()


DiscreteStateValueType = Union[str, int, tuple[int], Enum]


class DiscreteState(State, ABC):
    """
    Abstract base class for discrete state representations.
    """

    @abstractmethod
    def get_state_value(self) -> Any:
        """
        Get the current state value.

        Returns:
            The current state value.
        """
        raise NotImplemented()

    @abstractmethod
    def set_state_value(self, val: Any) -> None:
        """
        Set the current state value.

        Args:
            val: The state value to set.
        """
        raise NotImplemented()

    @property
    def state_value(self) -> Any:
        """
        Get the current state value.

        Returns:
            The current state value.
        """
        return self.get_state_value()

    @state_value.setter
    def state_value(self, new_state_value: Any) -> None:
        """
        Set the current state value.

        Args:
            new_state_value: The new state value to set.
        """
        self.set_state_value(new_state_value)

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the current state is terminal.

        Returns:
            True if the current state is terminal, False otherwise.
        """
        raise NotImplemented()

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        """
        Get the size of the state space.

        Returns:
            The number of possible state values.
        """
        raise NotImplemented()