
from abc import ABC, abstractmethod


class IndividualRewardFunction(ABC):
    @abstractmethod
    def reward(self, state_value, action_value, next_state_value) -> float:
        pass

    def __call__(self, state_value, action_value, next_state_value) -> float:
        return self.reward(state_value, action_value, next_state_value)
