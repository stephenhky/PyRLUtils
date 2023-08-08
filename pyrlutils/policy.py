
from abc import ABC, abstractmethod
from typing import Union, Dict
from warnings import warn

import numpy as np

from .state import State, DiscreteState, DiscreteStateValueType
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
    @abstractmethod
    def add_deterministic_rule(self, *args, **kwargs):
        pass

    @property
    def is_stochastic(self) -> bool:
        return False


class DiscreteDeterminsticPolicy(DeterministicPolicy):
    def __init__(self, actions_dict: Dict[DiscreteActionValueType, Action]):
        self._state_to_action = {}
        self._actions_dict = actions_dict

    def add_policy_rule(self, state_value: DiscreteStateValueType, action_value: DiscreteActionValueType):
        if state_value in self._state_to_action:
            warn('State value {} exists in rule; it will be replaced.'.format(state_value))
        self._state_to_action[state_value] = action_value

    def get_action_value(self, state_value: DiscreteStateValueType) -> DiscreteActionValueType:
        return self._state_to_action.get(state_value)

    def get_action(self, state: DiscreteState) -> Action:
        return self._actions_dict[self.get_action_value(state.state_value)]


class StochasticPolicy(Policy):
    @abstractmethod
    def get_probability(self, *args, **kwargs) -> float:
        pass

    @property
    def is_stochastic(self) -> bool:
        return True


class DiscreteStochasticPolicy(StochasticPolicy):
    @abstractmethod
    def get_probability(self, state_value: DiscreteStateValueType, action_value: DiscreteActionValueType) -> float:
        pass


class ContinuousStochasticPolicy(StochasticPolicy):
    @abstractmethod
    def get_probability(self, state_value: Union[float, np.ndarray], action_value: DiscreteActionValueType, value: Union[float, np.ndarray]) -> float:
        pass
