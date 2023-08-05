
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .state import State
from .action import Action, DiscreteActionValueType


class Policy(ABC):
    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass

    def __call__(self, state: State) -> Action:
        return self.get_action(state)

    @property
    def is_stochastic(self) -> bool:
        pass


class DeterministicPolicy(Policy):
    @property
    def is_stochastic(self) -> bool:
        return False


class StochasticPolicy(Policy):
    @abstractmethod
    def get_probability(self, state: State, action_value, *args, **kwargs) -> float:
        pass

    @property
    def is_stochastic(self) -> bool:
        return True


class DiscreteStochasticPolicy(StochasticPolicy):
    @abstractmethod
    def get_probability(self, state: State, action_value: DiscreteActionValueType) -> float:
        pass


class ContinuousStochasticPolicy(StochasticPolicy):
    @abstractmethod
    def get_probability(self, state: State, action_value: DiscreteActionValueType, value: Union[float, np.ndarray]) -> float:
        pass
