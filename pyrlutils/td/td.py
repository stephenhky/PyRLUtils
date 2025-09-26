
import numpy as np
from nptyping import NDArray, Shape, Float

from .utils import decay_schedule, AbstractTemporalDifferenceLearner


class SingleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int
    ) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]:
        V = np.zeros(self.nb_states)
        V_track = np.zeros((episodes, self.nb_states))
        alphas = decay_schedule(
            self._init_alpha, self._min_alpha, self._alpha_decay_ratio, episodes
        )

        for i in range(episodes):
            self._state.set_state_value(self._init_state_index)
            done = False
            while not done:
                old_state_index = self._state.state_index
                old_state_value = self._state.state_value
                action_value = self._policy.get_action_value(self._state)
                action_func = self._actions_dict[action_value]
                self._state = action_func(self._state)
                new_state_index = self._state.state_index
                new_state_value = self._state.state_value
                reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                done = self._state.is_terminal

                td_target = reward + self._gamma * V[new_state_index] * (not done)
                td_error = td_target - V[old_state_index]
                V[old_state_index] = V[old_state_index] + alphas[i] * td_error

            V_track[i, :] = V

        return V, V_track


class MultipleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int,
            n_steps: int=3
    ) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]:
        pass
