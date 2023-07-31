
from types import LambdaType
from typing import Union

from .state import State


DiscreteActionValueType = Union[float, str]

class Action:
    def __init__(self, actionfunc: LambdaType):
        self._actionfunc = actionfunc

    def act(self, state: State, *args, **kwargs) -> State:
        self._actionfunc(state, *args, **kwargs)
        return state

    def __call__(self, state: State) -> State:
        return self.act(state)
