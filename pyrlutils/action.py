
from types import LambdaType, FunctionType
from typing import Union

from .state import State


DiscreteActionValueType = Union[float, str]

class Action:
    def __init__(self, actionfunc: Union[FunctionType, LambdaType]):
        self._actionfunc = actionfunc

    def act(self, state: State, *args, **kwargs) -> State:
        self._actionfunc(state, *args, **kwargs)
        return state

    def __call__(self, state: State) -> State:
        return self.act(state)

    @property
    def action_function(self) -> Union[FunctionType, LambdaType]:
        return self._actionfunc

    @action_function.setter
    def action_function(self, new_func: Union[FunctionType, LambdaType]) -> None:
        self._actionfunc = new_func
