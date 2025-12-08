
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Any


class State(ABC):
    @property
    @abstractmethod
    def state_value(self):
        raise NotImplemented()


DiscreteStateValueType = Union[str, int, tuple[int], Enum]


class DiscreteState(State, ABC):
    @abstractmethod
    def get_state_value(self) -> Any:
        raise NotImplemented()

    @abstractmethod
    def set_state_value(self, val: Any) -> None:
        raise NotImplemented()

    @property
    def state_value(self) -> Any:
        return self.get_state_value()

    @state_value.setter
    def state_value(self, new_state_value: Any) -> None:
        self.set_state_value(new_state_value)

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplemented()

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        raise NotImplemented()
