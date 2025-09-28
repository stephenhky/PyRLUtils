
from abc import ABC, abstractmethod
from typing import Union, Annotated
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .state import State, DiscreteState, DiscreteStateValueType
from .action import Action, DiscreteActionValueType


class Policy(ABC):
    @abstractmethod
    def get_action(self, state: State) -> Action:
        raise NotImplemented()

    @abstractmethod
    def get_action_value(self, state: State) -> DiscreteActionValueType:
        raise NotImplemented()

    def __call__(self, state: State) -> Action:
        return self.get_action(state)

    @property
    def is_stochastic(self) -> bool:
        raise NotImplemented()


class DeterministicPolicy(Policy):
    @abstractmethod
    def add_deterministic_rule(self, *args, **kwargs):
        raise NotImplemented()

    @property
    def is_stochastic(self) -> bool:
        return False


class DiscreteDeterminsticPolicy(DeterministicPolicy):
    def __init__(self, actions_dict: dict[DiscreteActionValueType, Action]):
        self._state_to_action = {}
        self._actions_dict = actions_dict

    def add_deterministic_rule(
            self,
            state_value: DiscreteStateValueType,
            action_value: DiscreteActionValueType
    ) -> None:
        if state_value in self._state_to_action:
            warn('State value {} exists in rule; it will be replaced.'.format(state_value))
        self._state_to_action[state_value] = action_value

    def get_action_value(
            self,
            state_value: DiscreteStateValueType
    ) -> DiscreteActionValueType:
        return self._state_to_action.get(state_value)

    def get_action(self, state: DiscreteState) -> Action:
        return self._actions_dict[self.get_action_value(state.state_value)]

    def __eq__(self, other) -> bool:
        if len(self._state_to_action) != len(set(self._state_to_action.keys()).union(other._state_to_action.keys())):
            return False
        if len(self._actions_dict) != len(set(self._actions_dict.keys()).union(other._actions_dict.keys())):
            return False
        for action in self._actions_dict.keys():
            if self._actions_dict[action] != other._actions_dict[action]:
                return False
        for state in self._state_to_action.keys():
            if self._state_to_action[state] != other._state_to_action[state]:
                return False
        return True


class DiscreteContinuousPolicy(DeterministicPolicy):
    @abstractmethod
    def get_action(self, state: State) -> Action:
        raise NotImplemented()


class StochasticPolicy(Policy):
    @abstractmethod
    def get_probability(self, *args, **kwargs) -> float:
        raise NotImplemented()

    @property
    def is_stochastic(self) -> bool:
        return True


class DiscreteStochasticPolicy(StochasticPolicy):
    def __init__(self, actions_dict: dict[DiscreteActionValueType, Action]):
        self._state_to_action = {}
        self._actions_dict = actions_dict

    def add_stochastic_rule(
            self,
            state_value: DiscreteStateValueType,
            action_values: list[DiscreteActionValueType],
            probs: Union[list[float], Annotated[NDArray[np.float64], "1D Array"]] = None
    ):
        if probs is not None:
            assert len(action_values) == len(probs)
            probs = np.array(probs)
        else:
            probs = np.repeat(1./len(action_values), len(action_values))

        if state_value in self._state_to_action:
            warn('State value {} exists in rule; it will be replaced.'.format(state_value))
        self._state_to_action[state_value] = {
            action_value: prob
            for action_value, prob in zip(action_values, probs)
        }

    def get_probability(
            self,
            state_value: DiscreteStateValueType,
            action_value: DiscreteActionValueType
    ) -> float:
        if state_value not in self._state_to_action:
            return 0.0
        if action_value in self._state_to_action[state_value]:
            return self._state_to_action[state_value][action_value]
        else:
            return 0.0

    def get_action_value(self, state: State) -> DiscreteActionValueType:
        allowed_actions = list(self._state_to_action[state].keys())
        probs = np.array(list(self._state_to_action[state].values()))
        sumprobs = np.sum(probs)
        return np.random.choice(allowed_actions, p=probs/sumprobs)

    def get_action(self, state: DiscreteState) -> Action:
        return self._actions_dict[self.get_action_value(state.state_value)]


class ContinuousStochasticPolicy(StochasticPolicy):
    @abstractmethod
    def get_probability(
            self,
            state_value: Union[float, Annotated[NDArray[np.float64], "1D Array"]],
            action_value: DiscreteActionValueType,
            value: Union[float, Annotated[NDArray[np.float64], "1D Array"]]
    ) -> float:
        raise NotImplemented()


DiscretePolicy = Union[DiscreteDeterminsticPolicy, DiscreteStochasticPolicy]
ContinuousPolicy = Union[ContinuousStochasticPolicy]
