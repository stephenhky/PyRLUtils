"""
Bandit algorithms for reinforcement learning.
"""

from abc import ABC, abstractmethod

import numpy as np

from .reward import IndividualBanditRewardFunction


class BanditAlgorithm(ABC):
    """
    Abstract base class for bandit algorithms.
    """

    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction):
        """
        Initialize the bandit algorithm.

        Args:
            action_values: A list of possible action values.
            reward_function: The reward function that maps action values to rewards.
        """
        self._action_values = action_values
        self._reward_function = reward_function

    @abstractmethod
    def _go_one_loop(self):
        """
        Perform one iteration of the bandit algorithm.
        """
        pass

    def loop(self, nbiterations: int):
        """
        Run the bandit algorithm for a specified number of iterations.

        Args:
            nbiterations: The number of iterations to run.
        """
        for _ in range(nbiterations):
            self._go_one_loop()

    def reward(self, action_value) -> float:
        """
        Get the reward for taking an action.

        Args:
            action_value: The action value taken.

        Returns:
            The reward as a float.
        """
        return self._reward_function(action_value)

    @abstractmethod
    def get_action(self):
        """
        Get the action to take according to the bandit algorithm.

        Returns:
            The action value to take.
        """
        pass

    @property
    def action_values(self):
        """
        Get the list of possible action values.

        Returns:
            A list of possible action values.
        """
        return self._action_values

    @property
    def reward_function(self) -> IndividualBanditRewardFunction:
        """
        Get the reward function.

        Returns:
            The reward function used by the bandit algorithm.
        """
        return self._reward_function


class SimpleBandit(BanditAlgorithm):
    """
    An epsilon-greedy bandit algorithm.
    """

    def __init__(
            self,
            action_values: list,
            reward_function: IndividualBanditRewardFunction,
            epsilon: float=0.05
    ):
        """
        Initialize the epsilon-greedy bandit algorithm.

        Args:
            action_values: A list of possible action values.
            reward_function: The reward function that maps action values to rewards.
            epsilon: The exploration rate (probability of choosing a random action).
        """
        super().__init__(action_values, reward_function)
        self._epsilon = epsilon
        self._initialize()

    def _initialize(self):
        """
        Initialize the action-value estimates and action counts.
        """
        self._Q = np.zeros(len(self._action_values))
        self._N = np.zeros(len(self._action_values), dtype=np.int32)

    def _go_one_loop(self):
        """
        Perform one iteration of the epsilon-greedy bandit algorithm.
        
        With probability epsilon, select a random action (exploration).
        Otherwise, select the action with the highest estimated value (exploitation).
        """
        r = np.random.uniform()
        if r < self.epsilon:
            selected_action_idx = np.argmax(self._Q)
        else:
            selected_action_idx = np.random.choice(range(len(self._action_values)))
        reward = self._reward_function(self._action_values[selected_action_idx])
        self._N[selected_action_idx] += 1
        self._Q[selected_action_idx] += (reward - self._Q[selected_action_idx]) / self._N[selected_action_idx]

    def get_action(self):
        """
        Get the action to take according to the epsilon-greedy policy.
        
        Returns the action with the highest estimated value (greedy choice).

        Returns:
            The action value to take.
        """
        selected_action_idx = np.argmax(self._Q)
        return self._action_values[selected_action_idx]

    @property
    def epsilon(self) -> float:
        """
        Get the exploration rate.

        Returns:
            The exploration rate (epsilon).
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, val: float):
        """
        Set the exploration rate.

        Args:
            val: The exploration rate to set.
        """
        self._epsilon = val


class GradientBandit(BanditAlgorithm):
    """
    A gradient-based bandit algorithm that uses softmax action selection.
    """

    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction, temperature: float=1.0, alpha: float=0.1):
        """
        Initialize the gradient bandit algorithm.

        Args:
            action_values: A list of possible action values.
            reward_function: The reward function that maps action values to rewards.
            temperature: The temperature parameter for softmax (default: 1.0).
            alpha: The learning rate (default: 0.1).
        """
        super().__init__(action_values, reward_function)
        self._T = temperature
        self._alpha = alpha
        self._initialize()

    def _initialize(self):
        """
        Initialize the action preferences and rewards history.
        """
        self._preferences = np.zeros(len(self._action_values))
        self._rewards_over_time = []

    def _get_probs(self) -> np.ndarray:
        """
        Get action probabilities using softmax.

        Returns:
            A numpy array of action probabilities.
        """
        # getting probabilities using softmax
        exp_preferences = np.exp(self._preferences / self.T)
        sum_exp_preferences = np.sum(exp_preferences)
        return exp_preferences / sum_exp_preferences

    def get_action(self):
        """
        Get the action to take according to the gradient bandit algorithm.
        
        Returns the action with the highest preference.

        Returns:
            The action value to take.
        """
        selected_action_idx = np.argmax(self._preferences)
        return self._action_values[selected_action_idx]

    def _go_one_loop(self):
        """
        Perform one iteration of the gradient bandit algorithm.
        
        Selects an action using softmax policy, observes reward, and updates action preferences.
        """
        probs = self._get_probs()
        selected_action_idx = np.random.choice(range(self._preferences.shape[0]), p=probs)
        reward = self._reward_function(self._action_values[selected_action_idx])
        self._rewards_over_time.append(reward)
        average_reward = np.mean(self._rewards_over_time) if len(self._rewards_over_time) > 0 else 0.

        for i in range(len(self._action_values)):
            if i == selected_action_idx:
                self._preferences[i] += self.alpha * (reward - average_reward) * (1 - probs[i])
            else:
                self._preferences[i] -= self.alpha * (reward - average_reward) * probs[i]

    @property
    def alpha(self) -> float:
        """
        Get the learning rate.

        Returns:
            The learning rate (alpha).
        """
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        """
        Set the learning rate.

        Args:
            val: The learning rate to set.
        """
        self._alpha = val

    @property
    def T(self) -> float:
        """
        Get the temperature parameter.

        Returns:
            The temperature parameter for softmax.
        """
        return self._T

    @T.setter
    def T(self, val: float):
        """
        Set the temperature parameter.

        Args:
            val: The temperature parameter to set.
        """
        self._T = val

    @property
    def temperature(self) -> float:
        """
        Get the temperature parameter.

        Returns:
            The temperature parameter for softmax.
        """
        return self._T