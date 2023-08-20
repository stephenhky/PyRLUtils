
import random
from copy import copy
from typing import Tuple

import numpy as np

from .transition import TransitionProbabilityFactory
from .policy import DiscreteDeterminsticPolicy


class OptimalPolicyOnValueFunctions:
    def __init__(self, discount_factor: float, transprobfac: TransitionProbabilityFactory):
        try:
            assert discount_factor >= 0. and discount_factor <= 1.
        except AssertionError:
            raise ValueError('Discount factor must be between 0 and 1.')
        self._gamma = discount_factor
        self._transprobfac = transprobfac
        self._states, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state_names = self._states.get_all_possible_state_values()
        self._states_to_indices = {state: idx for idx, state in enumerate(self._state_names)}
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in self._action_names}

        self._evaluated = False
        self._improved = False

        self._theta = 1e-10
        self._policy_evaluation_maxiter = 10000

    def _policy_evaluation(self, policy: DiscreteDeterminsticPolicy) -> np.ndarray:
        prev_V = np.zeros(len(self._states_to_indices))

        for _ in range(self._policy_evaluation_maxiter):
            V = np.zeros(len(self._states_to_indices))
            for state_value in self._state_names:
                state_index = self._states_to_indices[state_value]
                action_value = policy.get_action_value(state_value)
                for next_state_tuple in self._transprobfac[state_value][action_value]:
                    prob = next_state_tuple.probability
                    reward = next_state_tuple.reward
                    next_state_value = next_state_tuple.next_state_value
                    next_state_index = self._states_to_indices[next_state_value]
                    terminal = next_state_tuple.terminal

                    V[state_index] += prob * (reward + (self._gamma*prev_V[next_state_index] if not terminal else 0.))

            if np.max(np.abs(V-prev_V)) < self._theta:
                break

            prev_V = V.copy()

        return V

    def _policy_improvement(self, V: np.ndarray) -> DiscreteDeterminsticPolicy:
        Q = np.zeros((len(self._states_to_indices), len(self._actions_to_indices)))

        for state_value in self._state_names:
            state_index = self._states_to_indices[state_value]
            for action_value in self._action_names:
                action_index = self._actions_to_indices[action_value]
                for next_state_tuple in self._transprobfac[state_value][action_value]:
                    prob = next_state_tuple.probability
                    reward = next_state_tuple.reward
                    next_state_value = next_state_tuple.next_state_value
                    next_state_index = self._states_to_indices[next_state_value]
                    terminal = next_state_tuple.terminal

                    Q[state_index, action_index] += prob * (reward + (self._gamma*V[next_state_index] if not terminal else 0.))

        optimal_policy = DiscreteDeterminsticPolicy(self._actions_dict)
        optimal_action_indices = np.argmax(Q, axis=1)
        for state_value, action_index in zip(self._state_names, optimal_action_indices):
            action_value = self._action_names[action_index]
            optimal_policy.add_policy_rule(state_value, action_value)
        return optimal_policy

    def _policy_iteration(self) -> Tuple[np.ndarray, DiscreteDeterminsticPolicy]:
        policy = DiscreteDeterminsticPolicy(self._actions_dict)
        for state_value in self._state_names:
            policy.add_policy_rule(state_value, random.choice(self._action_names))

        done = False
        while not done:
            old_policy = copy(policy)

            V = self._policy_evaluation(policy)
            policy = self._policy_improvement(V)

            if policy == old_policy:
                done = True

        return V, policy
