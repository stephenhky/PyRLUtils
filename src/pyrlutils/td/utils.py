"""
Utility functions and base classes for temporal difference learning algorithms.
"""

from typing import Annotated, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from npdict import NumpyNDArrayWrappedDict

from ..state.utils import DiscreteStateValueType
from ..state.discrete import DiscreteCategoricalState
from ..action import DiscreteActionValueType, Action
from ..policy import DiscretePolicy
from ..reward import IndividualRewardFunction


def decay_schedule(
        init_value: float,
        min_value: float,
        decay_ratio: float,
        max_steps: int,
        log_start: int = -2,
        log_base: int = 10
) -> Annotated[NDArray[np.float64], "1D Array"]:
    """
    Generate a decay schedule from an initial value to a minimum value.

    The schedule uses a logarithmic decay for a fraction of the steps (determined by decay_ratio)
    and then holds the minimum value for the remaining steps.

    Args:
        init_value: The initial value of the schedule.
        min_value: The minimum value to decay to.
        decay_ratio: The fraction of max_steps over which the decay occurs.
        max_steps: The total number of steps.
        log_start: The starting exponent for the logarithmic space (default: -2).
        log_base: The base of the logarithm (default: 10).

    Returns:
        A 1D numpy array of length max_steps containing the decay schedule.
    """
    decay_steps = int(max_steps * decay_ratio)
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
    """
    Select an action using an epsilon-greedy policy.

    With probability epsilon, a random action is selected. Otherwise, the action with the
    highest Q-value is selected.

    Args:
        state_value: The current state value.
        Q: The action-value function, either as a numpy array or a NumpyNDArrayWrappedDict.
        epsilon: The probability of selecting a random action.

    Returns:
        The selected action value (or index if Q is a numpy array).
    """
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
    """
    Elements of a temporal difference update step.

    Attributes:
        this_state_value: The current state value.
        reward: The reward received after taking the action.
        next_state_value: The next state value.
        done: Whether the next state is terminal.
    """
    this_state_value: DiscreteStateValueType
    reward: float
    next_state_value: DiscreteStateValueType
    done: bool


# Note: we must not use TransitionProbabilityFactory as an initial parameter

class AbstractStateValueFunctionTemporalDifferenceLearner(ABC):
    """
    Abstract base class for temporal difference learners that learn state value functions.
    """

    def __init__(
            self,
            state: DiscreteCategoricalState,
            actions_dict: dict[DiscreteActionValueType, Action],
            individual_rewardfcn: IndividualRewardFunction,
            gamma: float = 1.0,
            init_alpha: float = 0.5,
            min_alpha: float = 0.01,
            alpha_decay_ratio: float = 0.3,
            policy: Optional[DiscretePolicy] = None,
            initial_state_index: int = 0
    ):
        """
        Initialize the learner.

        Args:
            state: The state object.
            actions_dict: A dictionary mapping action values to Action objects.
            individual_rewardfcn: The individual reward function.
            gamma: The discount factor (default: 1.0).
            init_alpha: The initial learning rate (default: 0.5).
            min_alpha: The minimum learning rate (default: 0.01).
            alpha_decay_ratio: The fraction of steps over which the learning rate decays (default: 0.3).
            policy: The policy to follow (optional, default: None).
            initial_state_index: The index of the initial state (default: 0).
        """
        self._gamma = gamma
        self._init_alpha = init_alpha
        self._min_alpha = min_alpha
        try:
            assert 0.0 <= alpha_decay_ratio <= 1.0
        except AssertionError:
            raise ValueError("alpha_decay_ratio must be between 0 and 1!")
        self._alpha_decay_ratio = alpha_decay_ratio
        # self._transprobfac = transprobfac
        # self._state, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state = state
        self._actions_dict = actions_dict
        self._indrewardfcn = individual_rewardfcn
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self._policy = policy
        try:
            assert 0 <= initial_state_index < self._state.nb_state_values
        except AssertionError:
            raise ValueError(f"Initial state index must be between 0 and {self._state.nb_state_values}")
        self._init_state_index = initial_state_index

    @abstractmethod
    def learn(self, *args, **kwargs) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        """
        Run the learning algorithm.

        Returns:
            A tuple containing:
                - A 1D array of the learned value function (shape: [nb_states]).
                - A 2D array of the tracked value function over episodes (shape: [episodes, nb_states]).
        """
        raise NotImplementedError()

    @property
    def nb_states(self) -> int:
        """
        Get the number of states.

        Returns:
            The number of states in the state space.
        """
        return self._state.nb_state_values

    @property
    def policy(self) -> DiscretePolicy:
        """
        Get the current policy.

        Returns:
            The policy being used or followed.
        """
        return self._policy

    @policy.setter
    def policy(self, val: DiscretePolicy):
        """
        Set the policy.

        Args:
            val: The policy to set.
        """
        self._policy = val

    @property
    def gamma(self) -> float:
        """
        Get the discount factor.

        Returns:
            The discount factor gamma.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, val: float):
        """
        Set the discount factor.

        Args:
            val: The discount factor to set.
        """
        self._gamma = val

    @property
    def init_alpha(self) -> float:
        """
        Get the initial learning rate.

        Returns:
            The initial learning rate.
        """
        return self._init_alpha

    @init_alpha.setter
    def init_alpha(self, val: float):
        """
        Set the initial learning rate.

        Args:
            val: The initial learning rate to set.
        """
        self._init_alpha = val

    @property
    def min_alpha(self) -> float:
        """
        Get the minimum learning rate.

        Returns:
            The minimum learning rate.
        """
        return self._min_alpha

    @min_alpha.setter
    def min_alpha(self, val: float):
        """
        Set the minimum learning rate.

        Args:
            val: The minimum learning rate to set.
        """
        self._min_alpha = val

    @property
    def alpha_decay_ratio(self) -> float:
        """
        Get the alpha decay ratio.

        Returns:
            The fraction of steps over which the learning rate decays.
        """
        return self._alpha_decay_ratio

    @property
    def initial_state_index(self) -> int:
        """
        Get the initial state index.

        Returns:
            The index of the initial state.
        """
        return self._init_state_index

    @initial_state_index.setter
    def initial_state_index(self, val: int):
        """
        Set the initial state index.

        Args:
            val: The initial state index to set.
        """
        self._init_state_index = val


class AbstractStateActionValueFunctionTemporalDifferenceLearner(ABC):
    """
    Abstract base class for temporal difference learners that learn state-action value functions (Q-functions).
    """

    def __init__(
            self,
            state: DiscreteCategoricalState,
            actions_dict: dict[DiscreteActionValueType, Action],
            individual_rewardfcn: IndividualRewardFunction,
            gamma: float = 1.0,
            init_alpha: float = 0.5,
            min_alpha: float = 0.01,
            alpha_decay_ratio: float = 0.3,
            init_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            epsilon_decay_ratio: float = 0.9,
            policy: Optional[DiscretePolicy] = None,
            initial_state_index: int = 0
    ):
        """
        Initialize the learner.

        Args:
            state: The state object.
            actions_dict: A dictionary mapping action values to Action objects.
            individual_rewardfcn: The individual reward function.
            gamma: The discount factor (default: 1.0).
            init_alpha: The initial learning rate (default: 0.5).
            min_alpha: The minimum learning rate (default: 0.01).
            alpha_decay_ratio: The fraction of steps over which the learning rate decays (default: 0.3).
            init_epsilon: The initial exploration rate for epsilon-greedy (default: 1.0).
            min_epsilon: The minimum exploration rate (default: 0.1).
            epsilon_decay_ratio: The fraction of steps over which the exploration rate decays (default: 0.9).
            policy: The policy to follow (optional, default: None).
            initial_state_index: The index of the initial state (default: 0).
        """
        self._gamma = gamma
        self._init_alpha = init_alpha
        self._min_alpha = min_alpha
        try:
            assert 0.0 <= alpha_decay_ratio <= 1.0
        except AssertionError:
            raise ValueError("alpha_decay_ratio must be between 0 and 1!")
        self._alpha_decay_ratio = alpha_decay_ratio
        self._init_epsilon = init_epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay_ratio = epsilon_decay_ratio

        # self._transprobfac = transprobfac
        # self._state, self._actions_dict, self._indrewardfcn = self._transprobfac.generate_mdp_objects()
        self._state = state
        self._actions_dict = actions_dict
        self._indrewardfcn = individual_rewardfcn
        self._action_names = list(self._actions_dict.keys())
        self._actions_to_indices = {action_value: idx for idx, action_value in enumerate(self._action_names)}
        self._policy = policy
        try:
            assert 0 <= initial_state_index < self._state.nb_state_values
        except AssertionError:
            raise ValueError(f"Initial state index must be between 0 and {self._state.nb_state_values}")
        self._init_state_index = initial_state_index

    @abstractmethod
    def learn(self, *args, **kwargs) -> tuple[Annotated[NDArray[np.float64], "1D Array"], Annotated[NDArray[np.float64], "2D Array"]]:
        """
        Run the learning algorithm.

        Returns:
            A tuple containing:
                - A 1D array of the learned state-action value function (shape: [nb_states * nb_actions] or similar).
                - A 2D array of the tracked state-action value function over episodes (shape: [episodes, nb_states * nb_actions] or similar).
        """
        raise NotImplementedError()

    @property
    def nb_states(self) -> int:
        """
        Get the number of states.

        Returns:
            The number of states in the state space.
        """
        return self._state.nb_state_values

    @property
    def policy(self) -> DiscretePolicy:
        """
        Get the current policy.

        Returns:
            The policy being used or followed.
        """
        return self._policy

    @policy.setter
    def policy(self, val: DiscretePolicy):
        """
        Set the policy.

        Args:
            val: The policy to set.
        """
        self._policy = val

    @property
    def gamma(self) -> float:
        """
        Get the discount factor.

        Returns:
            The discount factor gamma.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, val: float):
        """
        Set the discount factor.

        Args:
            val: The discount factor to set.
        """
        self._gamma = val

    @property
    def init_alpha(self) -> float:
        """
        Get the initial learning rate.

        Returns:
            The initial learning rate.
        """
        return self._init_alpha

    @init_alpha.setter
    def init_alpha(self, val: float):
        """
        Set the initial learning rate.

        Args:
            val: The initial learning rate to set.
        """
        self._init_alpha = val

    @property
    def min_alpha(self) -> float:
        """
        Get the minimum learning rate.

        Returns:
            The minimum learning rate.
        """
        return self._min_alpha

    @min_alpha.setter
    def min_alpha(self, val: float):
        """
        Set the minimum learning rate.

        Args:
            val: The minimum learning rate to set.
        """
        self._min_alpha = val

    @property
    def alpha_decay_ratio(self) -> float:
        """
        Get the alpha decay ratio.

        Returns:
            The fraction of steps over which the learning rate decays.
        """
        return self._alpha_decay_ratio

    @property
    def init_epsilon(self) -> float:
        """
        Get the initial exploration rate.

        Returns:
            The initial exploration rate for epsilon-greedy.
        """
        return self._init_epsilon

    @init_epsilon.setter
    def init_epsilon(self, val: float):
        """
        Set the initial exploration rate.

        Args:
            val: The initial exploration rate to set.
        """
        self._init_epsilon = val

    @property
    def min_epsilon(self) -> float:
        """
        Get the minimum exploration rate.

        Returns:
            The minimum exploration rate for epsilon-greedy.
        """
        return self._min_epsilon

    @min_epsilon.setter
    def min_epsilon(self, val: float):
        """
        Set the minimum exploration rate.

        Args:
            val: The minimum exploration rate to set.
        """
        self._min_epsilon = val

    @property
    def epsilon_decay_ratio(self) -> float:
        """
        Get the epsilon decay ratio.

        Returns:
            The fraction of steps over which the exploration rate decays.
        """
        return self._epsilon_decay_ratio

    @epsilon_decay_ratio.setter
    def epsilon_decay_ratio(self, val: float):
        """
        Set the epsilon decay ratio.

        Args:
            val: The epsilon decay ratio to set.
        """
        self._epsilon_decay_ratio = val

    @property
    def initial_state_index(self) -> int:
        """
        Get the initial state index.

        Returns:
            The index of the initial state.
        """
        return self._init_state_index

    @initial_state_index.setter
    def initial_state_index(self, val: int):
        """
        Set the initial state index.

        Args:
            val: The initial state index to set.
        """
        self._init_state_index = val