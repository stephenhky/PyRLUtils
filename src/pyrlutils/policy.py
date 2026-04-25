"""
Policy implementations for reinforcement learning.
"""

from abc import ABC, abstractmethod
from typing import Union, Annotated
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .state.utils import State, DiscreteState, DiscreteStateValueType
from .action import Action, DiscreteActionValueType


class Policy(ABC):
    """
    Abstract base class for policies.
    """

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Get the action to take in the given state.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        raise NotImplemented()

    @abstractmethod
    def get_action_value(self, state: State) -> DiscreteActionValueType:
        """
        Get the action value to take in the given state.

        Args:
            state: The current state.

        Returns:
            The action value to take.
        """
        raise NotImplemented()

    def __call__(self, state: State) -> Action:
        """
        Call the policy as a function to get the action for a state.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        return self.get_action(state)

    @property
    @abstractmethod
    def is_stochastic(self) -> bool:
        """
        Whether the policy is stochastic.

        Returns:
            True if the policy is stochastic, False otherwise.
        """
        raise NotImplemented()


class DeterministicPolicy(Policy):
    """
    Abstract base class for deterministic policies.
    """

    @abstractmethod
    def add_deterministic_rule(self, *args, **kwargs):
        """
        Add a deterministic rule to the policy.
        """
        raise NotImplemented()

    @property
    def is_stochastic(self) -> bool:
        """
        Whether the policy is stochastic.

        Returns:
            False, as this is a deterministic policy.
        """
        return False


class DiscreteDeterminsticPolicy(DeterministicPolicy):
    """
    A deterministic policy for discrete state and action spaces.
    """

    def __init__(self, actions_dict: dict[DiscreteActionValueType, Action]):
        """
        Initialize the deterministic policy.

        Args:
            actions_dict: A dictionary mapping action values to Action objects.
        """
        self._state_to_action = {}
        self._actions_dict = actions_dict

    def add_deterministic_rule(
            self,
            state_value: DiscreteStateValueType,
            action_value: DiscreteActionValueType
    ) -> None:
        """
        Add a deterministic rule mapping a state value to an action value.

        If the state value already exists in the rule, a warning is issued and the rule is replaced.

        Args:
            state_value: The state value.
            action_value: The action value to take in that state.
        """
        if state_value in self._state_to_action:
            warn('State value {} exists in rule; it will be replaced.'.format(state_value))
        self._state_to_action[state_value] = action_value

    def get_action_value(
            self,
            state_value: DiscreteStateValueType
    ) -> DiscreteActionValueType:
        """
        Get the action value for a given state value.

        Args:
            state_value: The state value.

        Returns:
            The action value to take in that state, or None if no rule is defined.
        """
        return self._state_to_action.get(state_value)

    def get_action(self, state: DiscreteState) -> Action:
        """
        Get the action to take in the given state.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        return self._actions_dict[self.get_action_value(state.state_value)]

    def __eq__(self, other) -> bool:
        """
        Check if two policies are equal.

        Args:
            other: Another policy to compare with.

        Returns:
            True if the policies are equal, False otherwise.
        """
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
    """
    A deterministic policy for continuous state and discrete action spaces.
    """

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Get the action to take in the given state.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        raise NotImplemented()


class StochasticPolicy(Policy):
    """
    Abstract base class for stochastic policies.
    """

    @abstractmethod
    def get_probability(self, *args, **kwargs) -> float:
        """
        Get the probability of taking an action in a state.

        Args:
            *args: Arguments specific to the policy.
            **kwargs: Keyword arguments specific to the policy.

        Returns:
            The probability of taking the action.
        """
        raise NotImplemented()

    @property
    def is_stochastic(self) -> bool:
        """
        Whether the policy is stochastic.

        Returns:
            True, as this is a stochastic policy.
        """
        return True


class DiscreteStochasticPolicy(StochasticPolicy):
    """
    A stochastic policy for discrete state and action spaces.
    """

    def __init__(self, actions_dict: dict[DiscreteActionValueType, Action]):
        """
        Initialize the stochastic policy.

        Args:
            actions_dict: A dictionary mapping action values to Action objects.
        """
        self._state_to_action = {}
        self._actions_dict = actions_dict

    def add_stochastic_rule(
            self,
            state_value: DiscreteStateValueType,
            action_values: list[DiscreteActionValueType],
            probs: Union[list[float], Annotated[NDArray[np.float64], "1D Array"]] = None
    ):
        """
        Add a stochastic rule mapping a state value to a distribution over action values.

        If the state value already exists in the rule, a warning is issued and the rule is replaced.

        Args:
            state_value: The state value.
            action_values: A list of action values.
            probs: A list or array of probabilities for each action value. If None, uniform probabilities are used.
        """
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
        """
        Get the probability of taking an action in a state.

        Args:
            state_value: The state value.
            action_value: The action value.

        Returns:
            The probability of taking the action in the state, or 0.0 if not defined.
        """
        if state_value not in self._state_to_action:
            return 0.0
        if action_value in self._state_to_action[state_value]:
            return self._state_to_action[state_value][action_value]
        else:
            return 0.0

    def get_action_value(self, state: State) -> DiscreteActionValueType:
        """
        Get the action value to take in the given state by sampling from the policy's distribution.

        Args:
            state: The current state.

        Returns:
            The action value to take.
        """
        allowed_actions = list(self._state_to_action[state].keys())
        probs = np.array(list(self._state_to_action[state].values()))
        sumprobs = np.sum(probs)
        return np.random.choice(allowed_actions, p=probs/sumprobs)

    def get_action(self, state: DiscreteState) -> Action:
        """
        Get the action to take in the given state.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        return self._actions_dict[self.get_action_value(state.state_value)]


class ContinuousStochasticPolicy(StochasticPolicy):
    """
    A stochastic policy for continuous state and continuous action spaces.
    """

    @abstractmethod
    def get_probability(
            self,
            state_value: Union[float, Annotated[NDArray[np.float64], "1D Array"]],
            action_value: DiscreteActionValueType,
            value: Union[float, Annotated[NDArray[np.float64], "1D Array"]]
    ) -> float:
        """
        Get the probability of taking an action in a state.

        Args:
            state_value: The state value (can be a float or a 1D array).
            action_value: The action value.
            value: The value of the action (for continuous actions, can be a float or a 1D array).

        Returns:
            The probability of taking the action.
        """
        raise NotImplemented()


DiscretePolicy = Union[DiscreteDeterminsticPolicy, DiscreteStochasticPolicy]
ContinuousPolicy = Union[ContinuousStochasticPolicy]