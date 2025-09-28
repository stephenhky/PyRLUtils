
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from .utils import decay_schedule, AbstractTemporalDifferenceLearner, TimeDifferencePathElements


class SingleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int
    ) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        V = np.zeros(self.nb_states)
        V_track = np.zeros((episodes, self.nb_states))
        alphas = decay_schedule(
            self.init_alpha, self.min_alpha, self.alpha_decay_ratio, episodes
        )

        for i in range(episodes):
            self._state.set_state_value(self.initial_state_index)
            done = False
            while not done:
                old_state_index = self._state.state_index
                old_state_value = self._state.state_value
                action_value = self._policy.get_action_value(self._state.state_value)
                action_func = self._actions_dict[action_value]
                self._state = action_func(self._state)
                new_state_index = self._state.state_index
                new_state_value = self._state.state_value
                reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                done = self._state.is_terminal

                td_target = reward + self.gamma * V[new_state_index] * (not done)
                td_error = td_target - V[old_state_index]
                V[old_state_index] = V[old_state_index] + alphas[i] * td_error

            V_track[i, :] = V

        return V, V_track


class MultipleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
    def learn(
            self,
            episodes: int,
            n_steps: int=3
    ) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        V = np.zeros(self.nb_states)
        V_track = np.zeros((episodes, self.nb_states))
        alphas = decay_schedule(
            self.init_alpha, self.min_alpha, self.alpha_decay_ratio, episodes
        )
        discounts = np.logspace(0, n_steps-1, num=n_steps+1, base=self.gamma, endpoint=False)

        for i in range(episodes):
            self._state.set_state_value(self.initial_state_index)
            done = False
            path = []

            while not done or path is not None:
                path = path[1:]     # worth revisiting this line

                next_state_index = -1
                while not done and len(path) < n_steps:
                    old_state_index = self._state.state_index
                    old_state_value = self._state.state_value
                    action_value = self._policy.get_action_value(self._state.state_value)
                    action_func = self._actions_dict[action_value]
                    self._state = action_func(self._state)
                    new_state_index = self._state.state_index
                    new_state_value = self._state.state_value
                    reward = self._indrewardfcn(old_state_value, action_value, new_state_value)
                    done = self._state.is_terminal

                    path.append(
                        TimeDifferencePathElements(
                            this_state_index=old_state_index,
                            reward=reward,
                            next_state_index=new_state_index,
                            done=done
                        )
                    )
                    if done:
                        break

                n = len(path)
                estimated_state_index = path[0].this_state_index
                rewards = np.array([this_moment.reward for this_moment in path])
                partial_return = discounts[n:] * rewards
                bs_val = discounts[-1] * V[next_state_index] * (not done)
                ntd_target = np.sum(np.append(partial_return, bs_val))
                ntd_error = ntd_target - V[estimated_state_index]
                V[estimated_state_index] = V[estimated_state_index] + alphas[i] * ntd_error
                if len(path) == 1 and path[0].done:
                    path = None

            V_track[i, :] = V

        return V, V_track
