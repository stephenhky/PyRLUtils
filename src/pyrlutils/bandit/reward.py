
from abc import ABC, abstractmethod
from typing import Any


class IndividualBanditRewardFunction(ABC):
    @abstractmethod
    def reward(self, action_value: Any) -> float:
        pass

    def __call__(self, action_value: Any) -> float:
        return self.reward(action_value)
