
from abc import ABC, abstractmethod


class IndividualBanditRewardFunction(ABC):
    @abstractmethod
    def reward(self, action_value) -> float:
        pass

    def __call__(self, action_value) -> float:
        return self.reward(action_value)
