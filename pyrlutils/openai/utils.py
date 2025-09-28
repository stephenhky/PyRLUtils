
import gymnasium as gym

from ..transition import TransitionProbabilityFactory, NextStateTuple


class OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory(TransitionProbabilityFactory):
    def __init__(self, envname: str):
        super().__init__()
        self._envname = envname
        self._gymenv = gym.make(envname)
        self._convert_openai_gymenv_to_transprob()

    def _convert_openai_gymenv_to_transprob(self):
        P = self._gymenv.env.env.env.P
        for state_value, trans_dict in P.items():
            new_trans_dict = {}
            for action_value, next_state_list in trans_dict.items():
                new_trans_dict[action_value] = [
                    NextStateTuple(next_state[1], next_state[0], next_state[2], next_state[3])
                    for next_state in next_state_list
                ]
            self.add_state_transitions(state_value, new_trans_dict)

    @property
    def envname(self) -> str:
        return self._envname

    @property
    def gymenv(self) -> gym.Env:
        return self._gymenv