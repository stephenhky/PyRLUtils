
from ..reward import IndividualRewardFunction


class IndividualBanditRewardFunction(IndividualRewardFunction):
    def __call__(self, action_value) -> float:
        return self.reward(None, action_value, None)