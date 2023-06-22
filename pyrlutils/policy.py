
from abc import ABC, abstractmethod

from .state import State
from .action import Action


class Policy(ABC):
    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass

    def __call__(self, state: State) -> Action:
        return self.get_action(state)


class DeterministicPolicy(Policy):
    pass


class StochasticPolicy(Policy):
    @abstractmethod
    def get_probability(self, state: State, action: Action) -> float:
        pass
