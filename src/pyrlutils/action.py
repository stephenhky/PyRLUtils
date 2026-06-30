
from types import LambdaType, FunctionType
from typing import Union

from .state import State


DiscreteActionValueType = Union[float, str]

class Action:
    def __init__(self, actionfunc: Union[FunctionType, LambdaType]):
        self.action_function = actionfunc

    def act(self, state: State, *args, **kwargs) -> State:
        self.action_function(state, *args, **kwargs)
        return state

    def __call__(self, state: State) -> State:
        return self.act(state)
