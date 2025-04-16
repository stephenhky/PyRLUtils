
import numpy as np

from ..policy import DiscretePolicy
from ..transition import TransitionProbabilityFactory
from .utils import decay_schedule


class TDLearner:
    def __init__(
            self,
            transprobfac: TransitionProbabilityFactory,
            gamma: float = 1.0,
            init_alpha: float = 0.5,
            min_alpha: float = 0.01,
            alpha_decay_ratio: float=0.3,
            policy: DiscretePolicy = None,
            initial_state_index: int = 0
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
        self._states, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state_names = self._states.get_all_possible_state_values()
        self._states_to_indices = {state: idx for idx, state in enumerate(self._state_names)}
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self._policy = policy
        try:
            assert 0 <= initial_state_index < len(self._states_names)
        except AssertionError:
            raise ValueError("Initial state index must be between 0 and {}".format(len(self._state_names)))
        self._init_state_index = initial_state_index

    def learn(self, episodes: int):
        V = np.zeros(self.nb_states)
        V_track = np.zeros((episodes, self.nb_states))
        alphas = decay_schedule(self._init_alpha, self._min_alpha, self._alpha_decay_ratio, episodes)

        for i in range(episodes):
            pass

    @property
    def nb_states(self) -> int:
        return len(self._states_names)

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
