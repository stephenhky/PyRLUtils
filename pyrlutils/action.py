
from types import LambdaType

from .state import State


class Action:
    def __init__(self, actionfunc: LambdaType):
        self._actionfunc = actionfunc

    def act(self, state: State) -> State:
        self._actionfunc(state)
        return state

    def __call__(self, state: State) -> State:
        return self.act(state)
