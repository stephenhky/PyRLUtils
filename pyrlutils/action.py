
from abc import ABC, abstractmethod

from .state import State


class Action(ABC):
    @abstractmethod
    def act(self, state: State) -> State:
        pass

    def __call__(self, state: State) -> State:
        return self.act(state)
