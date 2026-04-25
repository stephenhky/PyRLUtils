"""
Reward function implementations for reinforcement learning.
"""

from abc import ABC, abstractmethod


class IndividualRewardFunction(ABC):
    """
    Abstract base class for individual reward functions.
    
    An individual reward function maps a state-action-next_state triple to a reward value.
    """

    @abstractmethod
    def reward(self, state_value, action_value, next_state_value) -> float:
        """
        Get the reward for transitioning from state to next_state via action.

        Args:
            state_value: The current state value.
            action_value: The action taken.
            next_state_value: The resulting state value.

        Returns:
            The reward as a float.
        """
        pass

    def __call__(self, state_value, action_value, next_state_value) -> float:
        """
        Call the reward function as a function.

        Args:
            state_value: The current state value.
            action_value: The action taken.
            next_state_value: The resulting state value.

        Returns:
            The reward as a float.
        """
        return self.reward(state_value, action_value, next_state_value)


class RewardFunction(ABC):
    """
    Abstract base class for reward functions that compute total (discounted) rewards.
    """

    def __init__(self, discount_factor: float, individual_reward_function: IndividualRewardFunction):
        """
        Initialize the reward function.

        Args:
            discount_factor: The discount factor gamma (between 0 and 1).
            individual_reward_function: The individual reward function to use for single steps.
        """
        self._discount_factor = discount_factor
        self._individual_reward_function = individual_reward_function

    @property
    def discount_factor(self) -> float:
        """
        Get the discount factor.

        Returns:
            The discount factor gamma.
        """
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, discount_factor: float):
        """
        Set the discount factor.

        Args:
            discount_factor: The discount factor to set.
        """
        self._discount_factor = discount_factor

    def individual_reward(self, state_value, action_value, next_state_value) -> float:
        """
        Get the individual reward for a state-action-next_state triple.

        Args:
            state_value: The current state value.
            action_value: The action taken.
            next_state_value: The resulting state value.

        Returns:
            The individual reward as a float.
        """
        return self._individual_reward_function(state_value, action_value, next_state_value)

    @abstractmethod
    def total_reward(self, state_value, action_value) -> float:
        """
        Get the total (discounted) reward for taking an action in a state.

        Args:
            state_value: The current state value.
            action_value: The action taken.

        Returns:
            The total discounted reward as a float.
        """
        pass

    def __call__(self, state_value, action_value) -> float:
        """
        Call the reward function as a function.

        Args:
            state_value: The current state value.
            action_value: The action taken.

        Returns:
            The total discounted reward as a float.
        """
        return self.total_reward(state_value, action_value)