
from ..transition import TransitionProbabilityFactory
from ..policy import DiscreteDeterminsticPolicy


class TDLearner:
    def __init__(self, discount_factor: float, transprobfac: TransitionProbabilityFactory):
        try:
            assert 0. <= discount_factor <= 1.
        except AssertionError:
            raise ValueError('Discount factor must be between 0 and 1.')
        self._gamma = discount_factor
        self._transprobfac = transprobfac
        self._states, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state_names = self._states.get_all_possible_state_values()
        self._states_to_indices = {state: idx for idx, state in enumerate(self._state_names)}
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}

        self._policy = DiscreteDeterminsticPolicy(self._actions_dict)
