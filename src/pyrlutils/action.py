"""
Action implementations for reinforcement learning.
"""

from types import LambdaType, FunctionType
from typing import Union

from .state.utils import State

DiscreteActionValueType = Union[float, str]

class Action:
    """
    An action that can be taken in a state, represented by a function that modifies the state.
    """

    def __init__(self, actionfunc: Union[FunctionType, LambdaType]):
        """
        Initialize the action with a function that modifies the state.

        Args:
            actionfunc: A function that takes a state and modifies it, returning the modified state.
        """
        self._actionfunc = actionfunc

    def act(self, state: State, *args, **kwargs) -> State:
        """
        Apply the action to a state.

        Args:
            state: The state to apply the action to.
            *args: Additional positional arguments to pass to the action function.
            **kwargs: Additional keyword arguments to pass to the action function.

        Returns:
            The modified state after applying the action.
        """
        self._actionfunc(state, *args, **kwargs)
        return state

    def __call__(self, state: State) -> State:
        """
        Apply the action to a state (makes the Action class callable).

        Args:
            state: The state to apply the action to.

        Returns:
            The modified state after applying the action.
        """
        return self.act(state)

    @property
    def action_function(self) -> Union[FunctionType, LambdaType]:
        """
        Get the action function.

        Returns:
            The function that defines the action.
        """
        return self._actionfunc

    @action_function.setter
    def action_function(self, new_func: Union[FunctionType, LambdaType]) -> None:
        """
        Set the action function.

        Args:
            new_func: The new function that defines the action.
        """
        self._actionfunc = new_func