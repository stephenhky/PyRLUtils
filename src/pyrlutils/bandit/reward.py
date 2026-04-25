"""
Reward functions for bandit algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any


class IndividualBanditRewardFunction(ABC):
    """
    Abstract base class for individual bandit reward functions.
    """

    @abstractmethod
    def reward(self, action_value: Any) -> float:
        """
        Get the reward for taking an action.

        Args:
            action_value: The action value taken.

        Returns:
            The reward as a float.
        """
        pass

    def __call__(self, action_value: Any) -> float:
        """
        Call the reward function as a function.

        Args:
            action_value: The action value taken.

        Returns:
            The reward as a float.
        """
        return self.reward(action_value)