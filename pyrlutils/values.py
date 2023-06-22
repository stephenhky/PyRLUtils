
from abc import ABC, abstractmethod

from .state import State
from .action import Action


class RewardFunction(ABC):
    @abstractmethod
    def reward(self, old_state: State, action: Action, new_state: State) -> float:
        pass

    def __call__(self, old_state: State, action: Action, new_state: State) -> float:
        return self.reward(old_state, action, new_state)

    