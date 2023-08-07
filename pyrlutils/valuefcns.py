
import numpy as np

from .transition import TransitionProbabilityFactory
from .policy import DeterministicPolicy


class OptimalPolicyOnValueFunctions:
    def __init__(self, discount_factor: float, transprobfac: TransitionProbabilityFactory):
        self._gamma = discount_factor
        self._transprobfac = transprobfac
        self._states, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._states_to_indices = {state: idx for idx, state in enumerate(self._states.get_all_possible_state_values())}
        self._actions_to_indices = {action_value: idx for idx, action_value in self._actions_dict.keys()}

        self._evaluated = False
        self._improved = False

    def _policy_evaluation(self):
        V = np.zeros(len(self._states_to_indices))



