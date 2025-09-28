
from abc import ABC, abstractmethod
from typing import Optional, Annotated
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..policy import DiscretePolicy
from ..transition import TransitionProbabilityFactory


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
            raise ValueError("Initial state index must be between 0 and {}".format(len(self._state_names)))
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


@dataclass
class TimeDifferencePathElements:
    this_state_index: int
    reward: float
    next_state_index: int
    done: bool
