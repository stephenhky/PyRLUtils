
from types import LambdaType, FunctionType
from typing import Union
from dataclasses import dataclass

import numpy as np

from .state import DiscreteState, DiscreteStateValueType
from .reward import IndividualRewardFunction
from .action import Action, DiscreteActionValueType


@dataclass
class NextStateTuple:
    next_state_value: DiscreteStateValueType
    probability: float
    reward: float
    terminal: bool


class TransitionProbabilityFactory:
    def __init__(self):
        self._transprobs = {}
        self._all_state_values = []
        self._all_action_values = []
        self._objects_generated = False

    def add_state_transitions(
            self,
            state_value: DiscreteStateValueType,
            action_values_to_next_state: dict[DiscreteActionValueType, Union[list[NextStateTuple], dict]]
    ):
        if state_value not in self._all_state_values:
            self._all_state_values.append(state_value)

        this_state_transition_dict = {}

        for action_value, next_state_tuples in action_values_to_next_state.items():
            this_state_transition_dict[action_value] = []
            for next_state_tuple in next_state_tuples:
                if action_value not in self._all_action_values:
                    self._all_action_values.append(action_value)
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

                if next_state_tuple.next_state_value not in self._all_state_values:
                    self._all_state_values.append(next_state_tuple.next_state_value)

                this_state_transition_dict[action_value].append(next_state_tuple)

        self._transprobs[state_value] = this_state_transition_dict

    def _get_probs_for_eachstate(
            self,
            action_value: DiscreteActionValueType
    ) -> dict[DiscreteStateValueType, list[NextStateTuple]]:
        state_nexttuples = {}
        for state_value, action_nexttuples_pair in self._transprobs.items():
            for this_action_value, nexttuples in action_nexttuples_pair.items():
                if this_action_value == action_value:
                    state_nexttuples[state_value] = nexttuples
        return state_nexttuples

    def _generate_action_function(
            self,
            state_nexttuples: dict[DiscreteStateValueType, list[NextStateTuple]]
    ) -> Union[FunctionType, LambdaType]:

        def _action_function(state: DiscreteState) -> DiscreteState:
            nexttuples = state_nexttuples[state.state_value]
            nextstates = [nexttuple.next_state_value for nexttuple in nexttuples]
            probs = [nexttuple.probability for nexttuple in nexttuples]
            next_state_value = np.random.choice(nextstates, p=probs)
            state.set_state_value(next_state_value)
            return state

        return _action_function

    def _generate_individual_reward_function(self) -> IndividualRewardFunction:

        def _individual_reward_function(
                state_value: DiscreteStateValueType,
                action_value: DiscreteActionValueType,
                next_state_value: DiscreteStateValueType
        ) -> float:
            if state_value not in self._transprobs.keys():
                return 0.

            if action_value not in self._transprobs[state_value].keys():
                return 0.

            reward = 0.
            for next_tuple in self._transprobs[state_value][action_value]:
                if next_tuple.next_state_value == next_state_value:
                    reward += next_tuple.reward
            return reward

        class ThisIndividualRewardFunction(IndividualRewardFunction):
            def reward(
                    self,
                    state_value: DiscreteStateValueType,
                    action_value: DiscreteActionValueType,
                    next_state_value: DiscreteStateValueType
            ) -> float:
                return _individual_reward_function(state_value, action_value, next_state_value)

        return ThisIndividualRewardFunction()

    def get_probability(
            self,
            state_value: DiscreteStateValueType,
            action_value: DiscreteActionValueType,
            new_state_value: DiscreteStateValueType
    ) -> float:
        if state_value not in self._transprobs.keys():
            return 0.

        if action_value not in self._transprobs[state_value]:
            return 0.

        probs = 0.
        for next_state_tuple in self._transprobs[state_value][action_value]:
            if next_state_tuple.next_state_value == new_state_value:
                probs += next_state_tuple.probability
        return probs

    @property
    def transition_probabilities(self) -> dict[DiscreteStateValueType, dict[DiscreteActionValueType, list[NextStateTuple]]]:
        return self._transprobs

    def generate_mdp_objects(self) -> tuple[DiscreteState, dict[DiscreteActionValueType, Action], IndividualRewardFunction]:
        state = DiscreteState(self._all_state_values)
        actions_dict = {}
        for action_value in self._all_action_values:
            state_nexttuple = self._get_probs_for_eachstate(action_value)
            actions_dict[action_value] = Action(self._generate_action_function(state_nexttuple))
            for next_tuples in state_nexttuple.values():
                for next_tuple in next_tuples:
                    state._terminal_dict[next_tuple.next_state_value] = next_tuple.terminal

        individual_reward_fcn = self._generate_individual_reward_function()
        self._objects_generated = True
        return state, actions_dict, individual_reward_fcn

    @property
    def objects_generated(self) -> bool:
        return self._objects_generated
