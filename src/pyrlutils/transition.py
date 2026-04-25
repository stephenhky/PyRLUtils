"""
Transition probability utilities for Markov Decision Processes.
"""

from typing import Union, Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .state.discrete import DiscreteCategoricalState
from .state.utils import DiscreteStateValueType
from .reward import IndividualRewardFunction
from .action import Action, DiscreteActionValueType


@dataclass
class NextStateTuple:
    """
    Represents a possible next state in a transition.

    Attributes:
        next_state_value: The value of the next state.
        probability: The probability of transitioning to this next state.
        reward: The reward received upon transitioning to this next state.
        terminal: Whether the next state is terminal.
    """
    next_state_value: Union[DiscreteStateValueType, npt.NDArray[np.int64]]
    probability: float
    reward: float
    terminal: bool


class TransitionProbabilityFactory:
    """
    Factory for creating transition probability structures and generating MDP objects.

    This class allows defining state-transition probabilities and rewards, and can
    generate the corresponding state, action, and reward function objects for an MDP.
    """

    def __init__(self):
        """Initialize the factory with empty transition structures."""
        self._transprobs = {}
        self._all_state_values = []
        self._all_action_values = []
        self._objects_generated = False

    def add_state_transitions(
            self,
            state_value: DiscreteStateValueType,
            action_values_to_next_state: dict[DiscreteActionValueType, Union[list[NextStateTuple], dict]]
    ):
        """
        Add transitions for a given state and its actions.

        Args:
            state_value: The current state value.
            action_values_to_next_state: A dictionary mapping action values to a list of
                NextStateTuple objects or dictionaries specifying the transition outcomes.
        """
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
        """
        Get, for each state, the list of next state tuples for a given action.

        Args:
            action_value: The action value to filter transitions by.

        Returns:
            A dictionary mapping state values to lists of NextStateTuple objects.
        """
        state_nexttuples = {}
        for state_value, action_nexttuples_pair in self._transprobs.items():
            for this_action_value, nexttuples in action_nexttuples_pair.items():
                if this_action_value == action_value:
                    state_nexttuples[state_value] = nexttuples
        return state_nexttuples

    def _generate_action_function(
            self,
            state_nexttuples: dict[DiscreteStateValueType, list[NextStateTuple]]
    ) -> Callable:
        """
        Generate a state transition function for a given action.

        Args:
            state_nexttuples: Mapping from state values to lists of possible next state tuples.

        Returns:
            A function that takes a state and returns the next state after applying the action.
        """
        def _action_function(state: DiscreteCategoricalState) -> DiscreteCategoricalState:
            nexttuples = state_nexttuples[state.state_value]
            nextstates = [nexttuple.next_state_value for nexttuple in nexttuples]
            probs = [nexttuple.probability for nexttuple in nexttuples]
            next_state_value = np.random.choice(nextstates, p=probs)
            state.set_state_value(next_state_value)
            return state

        return _action_function

    def _generate_individual_reward_function(self) -> IndividualRewardFunction:
        """
        Generate an individual reward function based on the defined transitions.

        Returns:
            An IndividualRewardFunction instance that computes rewards for state-action-next_state triples.
        """
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
        """
        Get the probability of transitioning from a state-action pair to a specific next state.

        Args:
            state_value: The current state value.
            action_value: The action taken.
            new_state_value: The target next state value.

        Returns:
            The transition probability (0.0 if the transition is not defined).
        """
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
        """
        Get the full transition probability dictionary.

        Returns:
            A nested dictionary mapping state values to action values to lists of NextStateTuple objects.
        """
        return self._transprobs

    def generate_mdp_objects(self) -> tuple[DiscreteCategoricalState, dict[DiscreteActionValueType, Action], IndividualRewardFunction]:
        """
        Generate MDP objects (state, actions, reward function) from the defined transitions.

        Returns:
            A tuple containing:
                - DiscreteCategoricalState: The state object.
                - dict[DiscreteActionValueType, Action]: Mapping of action values to Action objects.
                - IndividualRewardFunction: The reward function.
        """
        state = DiscreteCategoricalState(self._all_state_values)
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
        """
        Check whether MDP objects have been generated.

        Returns:
            True if generate_mdp_objects has been called, False otherwise.
        """
        return self._objects_generated