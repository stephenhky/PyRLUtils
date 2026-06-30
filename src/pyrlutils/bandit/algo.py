
from abc import ABC, abstractmethod

import numpy as np

from .reward import IndividualBanditRewardFunction


class BanditAlgorithm(ABC):
    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction):
        self.action_values = action_values
        self.reward_function = reward_function

    @abstractmethod
    def _go_one_loop(self):
        pass

    def loop(self, nbiterations: int):
        for _ in range(nbiterations):
            self._go_one_loop()

    def reward(self, action_value) -> float:
        return self.reward_function(action_value)

    @abstractmethod
    def get_action(self):
        pass


class SimpleBandit(BanditAlgorithm):
    def __init__(
            self,
            action_values: list,
            reward_function: IndividualBanditRewardFunction,
            epsilon: float=0.05
    ):
        super().__init__(action_values, reward_function)
        self.epsilon = epsilon
        self._initialize()

    def _initialize(self):
        self._Q = np.zeros(len(self.action_values))
        self._N = np.zeros(len(self.action_values), dtype=np.int32)

    def _go_one_loop(self):
        r = np.random.uniform()
        if r < self.epsilon:
            selected_action_idx = np.argmax(self._Q)
        else:
            selected_action_idx = np.random.choice(range(len(self.action_values)))
        reward = self.reward_function(self.action_values[selected_action_idx])
        self._N[selected_action_idx] += 1
        self._Q[selected_action_idx] += (reward - self._Q[selected_action_idx]) / self._N[selected_action_idx]

    def get_action(self):
        selected_action_idx = np.argmax(self._Q)
        return self.action_values[selected_action_idx]


class GradientBandit(BanditAlgorithm):
    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction, temperature: float=1.0, alpha: float=0.1):
        super().__init__(action_values, reward_function)
        self.temperature = temperature
        self.alpha = alpha
        self._initialize()

    def _initialize(self):
        self._preferences = np.zeros(len(self.action_values))
        self._rewards_over_time = []

    def _get_probs(self) -> np.ndarray:
        # getting probabilities using softmax
        exp_preferences = np.exp(self._preferences / self.temperature)
        sum_exp_preferences = np.sum(exp_preferences)
        return exp_preferences / sum_exp_preferences

    def get_action(self):
        selected_action_idx = np.argmax(self._preferences)
        return self.action_values[selected_action_idx]

    def _go_one_loop(self):
        probs = self._get_probs()
        selected_action_idx = np.random.choice(range(self._preferences.shape[0]), p=probs)
        reward = self.reward_function(self.action_values[selected_action_idx])
        self._rewards_over_time.append(reward)
        average_reward = np.mean(self._rewards_over_time) if len(self._rewards_over_time) > 0 else 0.

        for i in range(len(self.action_values)):
            if i == selected_action_idx:
                self._preferences[i] += self.alpha * (reward - average_reward) * (1 - probs[i])
            else:
                self._preferences[i] -= self.alpha * (reward - average_reward) * probs[i]

    @property
    def T(self) -> float:
        return self.temperature
