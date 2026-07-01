
from abc import ABC, abstractmethod


class IndividualRewardFunction(ABC):
    @abstractmethod
    def reward(self, state_value, action_value, next_state_value) -> float:
        pass

    def __call__(self, state_value, action_value, next_state_value) -> float:
        return self.reward(state_value, action_value, next_state_value)


class RewardFunction(ABC):
    def __init__(self, discount_factor: float, individual_reward_function: IndividualRewardFunction):
        self.discount_factor = discount_factor
        self.individual_reward_function = individual_reward_function

    def individual_reward(self, state_value, action_value, next_state_value) -> float:
        return self.individual_reward_function(state_value, action_value, next_state_value)

    @abstractmethod
    def total_reward(self, state_value, action_value) -> float:
        pass

    def __call__(self, state_value, action_value) -> float:
        return self.total_reward(state_value, action_value)


