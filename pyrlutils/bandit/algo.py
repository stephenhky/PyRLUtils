
import numpy as np

from .reward import IndividualBanditRewardFunction


class SimpleBandit:
    def __init__(self, action_values: list, reward_function: IndividualBanditRewardFunction, epsilon=0.05):
        self._epsilon = epsilon
        self._action_values = action_values
        self._reward_function = reward_function
        self._initialize()

    def _initialize(self):
        self._Q = np.zeros(len(self._action_values))
        self._N = np.zeros(len(self._action_values), dtype=np.int32)

    def _go_one_loop(self):
        r = np.random.uniform()
        if r < self._epsilon:
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
