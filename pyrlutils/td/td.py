
from typing import Annotated, Optional
from abc import ABC, abstractmethod

import numpy as np
from npdict import NumpyNDArrayWrappedDict
from numpy.typing import NDArray

from ..policy import DiscretePolicy
from ..transition import TransitionProbabilityFactory
from .utils import decay_schedule, TimeDifferencePathElements


class AbstractTemporalDifferenceLearner(ABC):
    def __init__(
            self,
            transprobfac: TransitionProbabilityFactory,
            gamma: float=1.0,
            init_alpha: float=0.5,
            min_alpha: float=0.01,
            alpha_decay_ratio: float=0.3,
            policy: Optional[DiscretePolicy]=None,
            initial_state_index: int=0
    ):
        self._gamma = gamma
        self._init_alpha = init_alpha
        self._min_alpha = min_alpha
        try:
            assert 0.0 <= alpha_decay_ratio <= 1.0
        except AssertionError:
            raise ValueError("alpha_decay_ratio must be between 0 and 1!")
        self._alpha_decay_ratio = alpha_decay_ratio
        self._transprobfac = transprobfac
        self._state, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self._policy = policy
        try:
            assert 0 <= initial_state_index < self._state.nb_state_values
        except AssertionError:
            raise ValueError(f"Initial state index must be between 0 and {self._state.nb_state_values}")
        self._init_state_index = initial_state_index

    @abstractmethod
    def learn(self, *args, **kwargs) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        raise NotImplementedError()

    @property
    def nb_states(self) -> int:
        return self._state.nb_state_values

    @property
    def policy(self) -> DiscretePolicy:
        return self._policy

    @policy.setter
    def policy(self, val: DiscretePolicy):
        self._policy = val

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, val: float):
        self._gamma = val

    @property
    def init_alpha(self) -> float:
        return self._init_alpha

    @init_alpha.setter
    def init_alpha(self, val: float):
        self._init_alpha = val

    @property
    def min_alpha(self) -> float:
        return self._min_alpha

    @min_alpha.setter
    def min_alpha(self, val: float):
        self._min_alpha = val

    @property
    def alpha_decay_ratio(self) -> float:
        return self._alpha_decay_ratio

    @property
    def initial_state_index(self) -> int:
        return self._init_state_index

    @initial_state_index.setter
    def initial_state_index(self, val: int):
        self._init_state_index = val


class SingleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
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
                V[(old_state_value,)] = V[old_state_value] + alphas[i] * td_error

            V_track_array[i, :] = V_array

        return V, V_track


class MultipleStepTemporalDifferenceLearner(AbstractTemporalDifferenceLearner):
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
