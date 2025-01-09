
import numpy as np

from .reward import IndividualBanditRewardFunction


class SimpleBandit:
    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction, epsilon: float=0.05):
        self._epsilon = epsilon
        self._action_values = action_values
        self._reward_function = reward_function
        self._initialize()

    def _initialize(self):
        self._Q = np.zeros(len(self._action_values))
        self._N = np.zeros(len(self._action_values), dtype=np.int32)

    def _go_one_loop(self):
        r = np.random.uniform()
        if r < self.epsilon:
            selected_action_idx = np.argmax(self._Q)
        else:
            selected_action_idx = np.random.choice(range(len(self._action_values)))
        reward = self._reward_function(self._action_values[selected_action_idx])
        self._N[selected_action_idx] += 1
        self._Q[selected_action_idx] += (reward - self._Q[selected_action_idx]) / self._N[selected_action_idx]

    def loop(self, nbiterations: int):
        for _ in range(nbiterations):
            self._go_one_loop()

    def get_action(self):
        selected_action_idx = np.argmax(self._Q)
        return self._action_values[selected_action_idx]

    def reward(self, action_value) -> float:
        return self._reward_function(action_value)

    @property
    def action_values(self):
        return self._action_values

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def reward_function(self) -> IndividualBanditRewardFunction:
        return self._reward_function

    @epsilon.setter
    def epsilon(self, val: float):
        self._epsilon = val


class GradientBandit:
    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction, temperature: float=1.0, alpha: float=0.1):
        self._action_values = action_values
        self._reward_function = reward_function
        self._T = temperature
        self._alpha = alpha
        self._initialize()

    def _initialize(self):
        self._preferences = np.zeros(len(self._action_values))
        self._rewards_over_time = []

    def _get_probs(self) -> np.ndarray:
        exp_preferences = np.exp(self._preferences / self.T)
        sum_exp_preferences = np.sum(exp_preferences)
        return exp_preferences / sum_exp_preferences

    def get_action(self):
        selected_action_idx = np.argmax(self._preferences)
        return self._action_values[selected_action_idx]

    def _go_one_loop(self):
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

    def loop(self, nbiterations: int):
        for _ in range(nbiterations):
            self._go_one_loop()

    @property
    def action_values(self):
        return self._action_values

    @property
    def reward_function(self) -> IndividualBanditRewardFunction:
        return self._reward_function

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        self._alpha = val

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, val: float):
        self._T = val