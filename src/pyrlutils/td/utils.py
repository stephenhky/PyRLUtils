
from typing import Annotated, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from npdict import NumpyNDArrayWrappedDict

from ..state import DiscreteStateValueType, DiscreteState
from ..action import DiscreteActionValueType, Action
from ..policy import DiscretePolicy
from ..reward import IndividualRewardFunction


def decay_schedule(
        init_value: float,
        min_value: float,
        decay_ratio: float,
        max_steps: int,
        log_start: int=-2,
        log_base: int=10
) -> Annotated[NDArray[np.float64], "1D Array"]:
    decay_steps = int(max_steps*decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def select_action(
        state_value: DiscreteStateValueType,
        Q: Union[Annotated[NDArray[np.float64], "2D Array"], NumpyNDArrayWrappedDict],
        epsilon: float,
) -> Union[DiscreteActionValueType, int]:
    if np.random.random() <= epsilon:
        if isinstance(Q, NumpyNDArrayWrappedDict):
            return np.random.choice(Q._lists_keystrings[1])
        else:
            return np.random.choice(np.arange(Q.shape[1]))

    q_matrix = Q.to_numpy() if isinstance(Q, NumpyNDArrayWrappedDict) else Q
    state_index = Q.get_key_index(0, state_value) if isinstance(Q, NumpyNDArrayWrappedDict) else state_value
    max_index = np.argmax(q_matrix[state_index, :])

    if isinstance(Q, NumpyNDArrayWrappedDict):
        return Q._lists_keystrings[1][max_index]
    else:
        return max_index


@dataclass
class TimeDifferencePathElements:
    this_state_value: DiscreteStateValueType
    reward: float
    next_state_value: DiscreteStateValueType
    done: bool


# Note: we must not use TransitionProbabilityFactory as an initial parameter

class AbstractStateValueFunctionTemporalDifferenceLearner(ABC):
    def __init__(
            self,
            state: DiscreteState,
            actions_dict: dict[DiscreteActionValueType, Action],
            individual_rewardfcn: IndividualRewardFunction,
            gamma: float=1.0,
            init_alpha: float=0.5,
            min_alpha: float=0.01,
            alpha_decay_ratio: float=0.3,
            policy: Optional[DiscretePolicy]=None,
            initial_state_index: int=0
    ):
        self.gamma = gamma
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        try:
            assert 0.0 <= alpha_decay_ratio <= 1.0
        except AssertionError:
            raise ValueError("alpha_decay_ratio must be between 0 and 1!")
        self.alpha_decay_ratio = alpha_decay_ratio
        # self._transprobfac = transprobfac
        # self._state, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state = state
        self._actions_dict = actions_dict
        self._indrewardfcn = individual_rewardfcn
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self.policy = policy
        try:
            assert 0 <= initial_state_index < self._state.nb_state_values
        except AssertionError:
            raise ValueError(f"Initial state index must be between 0 and {self._state.nb_state_values}")
        self.initial_state_index = initial_state_index

    @abstractmethod
    def learn(self, *args, **kwargs) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        raise NotImplementedError()

    @property
    def nb_states(self) -> int:
        return self._state.nb_state_values



class AbstractStateActionValueFunctionTemporalDifferenceLearner(ABC):
    def __init__(
            self,
            state: DiscreteState,
            actions_dict: dict[DiscreteActionValueType, Action],
            individual_rewardfcn: IndividualRewardFunction,
            gamma: float=1.0,
            init_alpha: float=0.5,
            min_alpha: float=0.01,
            alpha_decay_ratio: float=0.3,
            init_epsilon: float=1.0,
            min_epsilon: float=0.1,
            epsilon_decay_ratio: float=0.9,
            policy: Optional[DiscretePolicy]=None,
            initial_state_index: int=0
    ):
        self.gamma = gamma
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        try:
            assert 0.0 <= alpha_decay_ratio <= 1.0
        except AssertionError:
            raise ValueError("alpha_decay_ratio must be between 0 and 1!")
        self.alpha_decay_ratio = alpha_decay_ratio
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_ratio = epsilon_decay_ratio

        # self._transprobfac = transprobfac
        # self._state, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state = state
        self._actions_dict = actions_dict
        self._indrewardfcn = individual_rewardfcn
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self.policy = policy
        try:
            assert 0 <= initial_state_index < self._state.nb_state_values
        except AssertionError:
            raise ValueError(f"Initial state index must be between 0 and {self._state.nb_state_values}")
        self.initial_state_index = initial_state_index

    @abstractmethod
    def learn(self, *args, **kwargs) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        raise NotImplementedError()

    @property
    def nb_states(self) -> int:
        return self._state.nb_state_values
