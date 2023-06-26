
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
        self.all_actions_dict = {}

    def add_state_transitions(self, state_value, action_values_to_next_state: dict):
        if state_value not in self.all_state_values:
            self.all_state_values.append(state_value)

        this_state_transition_dict = {}

        for action_value, next_state_tuple in action_values_to_next_state.items():
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

            this_state_transition_dict[action_value] = next_state_tuple

        self.transprobs[state_value] = this_state_transition_dict


    def generate_mdp_objects(self):
        state = DiscreteState(self.all_state_values)

