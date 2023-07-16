
from types import LambdaType

import numpy as np

from .state import DiscreteState


class NextStateTuple:
    def __init__(self, next_state_value, probability: float, reward: float, terminal: bool):
        self._next_state_value = next_state_value
        self._probability = probability
        self._reward = reward
        self._terminal = terminal

    @property
    def next_state_value(self):
        return self._next_state_value

    @property
    def probability(self) -> float:
        return self._probability

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def terminal(self) -> bool:
        return self._terminal


class TransitionProbabilityFactory:
    def __init__(self):
        self.transprobs = {}
        self.all_state_values = []
        self.all_action_values = []
        self.objects_generated = False

    def add_state_transitions(self, state_value, action_values_to_next_state: dict):
        if state_value not in self.all_state_values:
            self.all_state_values.append(state_value)

        this_state_transition_dict = {}

        for action_value, next_state_tuples in action_values_to_next_state.items():
            this_state_transition_dict[action_value] = []
            for next_state_tuple in next_state_tuples:
                if action_value not in self.all_action_values:
                    self.all_action_values.append(action_value)
                if not isinstance(next_state_tuple, NextStateTuple):
                    if isinstance(next_state_tuple, dict):
                        next_state_tuple = NextStateTuple(
                            next_state_tuple['next_state_value'],
                            next_state_tuple['probability'],
                            next_state_tuple['reward'],
                            next_state_tuple['terminal']
                        )
                    else:
                        raise TypeError('"action_values_to_next_state" has to be a dictionary or NextStateTuple instance.')

                if next_state_tuple.next_state_value not in self.all_state_values:
                    self.all_state_values.append(next_state_tuple.next_state_value)

                this_state_transition_dict[action_value].append(next_state_tuple)

        self.transprobs[state_value] = this_state_transition_dict

    def _get_probs_for_eachstate(self, action_value):
        state_nexttuples = {}
        for state_value, action_nexttuples_pair in self.transprobs.items():
            for this_action_value, nexttuples in action_nexttuples_pair.items():
                if this_action_value == action_value:
                    state_nexttuples[state_value] = nexttuples
        return state_nexttuples

    def _generate_action_function(self, state_nexttuples: dict) -> LambdaType:

        def _action_function(state: DiscreteState):
            nexttuples = state_nexttuples[state.state_value]
            nextstates = [nexttuple.next_state_value for nexttuple in nexttuples]
            probs = [nexttuple.probability for nexttuple in nexttuples]
            return np.random.choice(nextstates, p=probs)

        return _action_function


    def generate_mdp_objects(self) -> tuple[DiscreteState, dict]:
        state = DiscreteState(self.all_state_values)
        actions_dict = {}
        for action_value in self.all_action_values:
            state_nexttuple = self._get_probs_for_eachstate(action_value)
            actions_dict[action_value] = self._generate_action_function(state_nexttuple)

        return state, actions_dict

