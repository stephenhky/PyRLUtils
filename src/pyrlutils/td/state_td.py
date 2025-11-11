
from typing import Annotated

import numpy as np
from npdict import NumpyNDArrayWrappedDict

from .utils import decay_schedule, TimeDifferencePathElements, AbstractStateValueFunctionTemporalDifferenceLearner


class SingleStepTemporalDifferenceLearner(AbstractStateValueFunctionTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int
    ) -> tuple[Annotated[NumpyNDArrayWrappedDict, "1D Array"], Annotated[NumpyNDArrayWrappedDict, "2D Array"]]:
        V = NumpyNDArrayWrappedDict(
            [self._state.get_all_possible_state_values()],
            default_initial_value=0.0
        )
        V_track = NumpyNDArrayWrappedDict(
            [list(range(episodes)), self._state.get_all_possible_state_values()],
            default_initial_value=0.0
        )
        V_array, V_track_array = V.to_numpy(), V_track.to_numpy()
        alphas = decay_schedule(
            self.init_alpha, self.min_alpha, self.alpha_decay_ratio, episodes
        )

        for i in range(episodes):
            self._state.state_index = self.initial_state_index
            done = False
            while not done:
                old_state_value = self._state.state_value
                action_value = self._policy.get_action_value(self._state.state_value)
                action_func = self._actions_dict[action_value]
                self._state = action_func(self._state)
                new_state_value = self._state.state_value
                reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                done = self._state.is_terminal

                td_target = reward + self.gamma * V[new_state_value] * (not done)
                td_error = td_target - V[old_state_value]
                V[old_state_value] = V[old_state_value] + alphas[i] * td_error

            V_track_array[i, :] = V_array

        return V, V_track


class MultipleStepTemporalDifferenceLearner(AbstractStateValueFunctionTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int,
            n_steps: int=3
    ) -> tuple[Annotated[NumpyNDArrayWrappedDict, "1D Array"], Annotated[NumpyNDArrayWrappedDict, "2D Array"]]:
        V = NumpyNDArrayWrappedDict(
            [self._state.get_all_possible_state_values()],
            default_initial_value=0.0
        )
        V_track = NumpyNDArrayWrappedDict(
            [list(range(episodes)), self._state.get_all_possible_state_values()],
            default_initial_value=0.0
        )
        V_array, V_track_array = V.to_numpy(), V_track.to_numpy()
        alphas = decay_schedule(
            self.init_alpha, self.min_alpha, self.alpha_decay_ratio, episodes
        )
        discounts = np.logspace(0, n_steps-1, num=n_steps+1, base=self.gamma, endpoint=False)

        for i in range(episodes):
            self._state.state_index = self.initial_state_index
            done = False
            path = []

            while not done or path is not None:
                path = path[1:]     # worth revisiting this line

                new_state_value = self._state._get_state_value_from_index(self._state.nb_state_values-1)
                while not done and len(path) < n_steps:
                    old_state_value = self._state.state_value
                    action_value = self._policy.get_action_value(self._state.state_value)
                    action_func = self._actions_dict[action_value]
                    self._state = action_func(self._state)
                    new_state_value = self._state.state_value
                    reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                    done = self._state.is_terminal

                    path.append(
                        TimeDifferencePathElements(
                            this_state_value=old_state_value,
                            reward=reward,
                            next_state_value=new_state_value,
                            done=done
                        )
                    )
                    if done:
                        break

                n = len(path)
                estimated_state_value = path[0].this_state_value
                rewards = np.array([this_moment.reward for this_moment in path])
                partial_return = discounts[n:] * rewards
                bs_val = discounts[-1] * V[new_state_value] * (not done)
                ntd_target = np.sum(np.append(partial_return, bs_val))
                ntd_error = ntd_target - V[estimated_state_value]
                V[(estimated_state_value,)] = V[estimated_state_value] + alphas[i] * ntd_error
                if len(path) == 1 and path[0].done:
                    path = None

            V_track_array[i, :] = V_array

        return V, V_track
