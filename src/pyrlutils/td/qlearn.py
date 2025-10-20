
from typing import Annotated

import numpy as np
from npdict import NumpyNDArrayWrappedDict

from .utils import AbstractStateActionValueFunctionTemporalDifferenceLearner, decay_schedule, select_action
from ..policy import DiscreteDeterminsticPolicy


class QLearner(AbstractStateActionValueFunctionTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int
    ) -> tuple[
        Annotated[NumpyNDArrayWrappedDict, "2D array"],
        Annotated[NumpyNDArrayWrappedDict, "1D array"],
        DiscreteDeterminsticPolicy,
        Annotated[NumpyNDArrayWrappedDict, "3D array"],
        list[DiscreteDeterminsticPolicy]
    ]:
        Q = NumpyNDArrayWrappedDict(
            [
                self._state.get_all_possible_state_values(),
                self._action_names
            ],
            default_initial_value=0.0
        )
        Q_track = NumpyNDArrayWrappedDict(
            [
                list(range(episodes)),
                self._state.get_all_possible_state_values(),
                self._action_names
            ],
            default_initial_value=0.0
        )
        pi_track = []

        Q_array, Q_track_array = Q.to_numpy(), Q_track.to_numpy()
        alphas = decay_schedule(
            self.init_alpha, self.min_alpha, self.alpha_decay_ratio, episodes
        )
        epsilons = decay_schedule(
            self.init_epsilon, self.min_epsilon, self.epsilon_decay_ratio, episodes
        )

        for i in range(episodes):
            self._state.state_index = self.initial_state_index
            done = False
            action_value = select_action(self._state.state_value, Q, epsilons[i])
            while not done:
                old_state_value = self._state.state_value
                new_action_value = select_action(self._state.state_value, Q, epsilons[i])
                new_action_func = self._actions_dict[new_action_value]
                self._state = new_action_func(self._state)
                new_state_value = self._state.state_value
                reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                done = self._state.is_terminal

                new_state_index = Q.get_key_index(0, new_state_value)
                max_Q_given_state = Q.to_numpy()[new_state_index, :].max()
                td_target = reward + self.gamma * max_Q_given_state * (not done)
                td_error = td_target - Q[old_state_value, action_value]
                Q[old_state_value, action_value] = Q[old_state_value, action_value] + alphas[i] * td_error

            Q_track_array[i, :, :] = Q_array
            pi_track.append(DiscreteDeterminsticPolicy(
                {
                    state_value: select_action(state_value, Q, epsilon=0.0)
                    for state_value in self._state.get_all_possible_state_values()
                }
            ))

        V_array = np.max(Q_array, axis=1)
        V = NumpyNDArrayWrappedDict.from_numpyarray_given_keywords(
            [self._state.get_all_possible_state_values()],
            V_array
        )
        pi = DiscreteDeterminsticPolicy(
                {
                    state_value: select_action(state_value, Q, epsilon=0.0)
                    for state_value in self._state.get_all_possible_state_values()
                }
        )

        return Q, V, pi, Q_track, pi_track
