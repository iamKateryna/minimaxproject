from enum import IntEnum, StrEnum
import gymnasium.spaces
from abc import abstractmethod
import numpy as np
from .constants import *

"""
The following classes are the types of objects that we are currently supporting
"""


class Entity:
    def __init__(self, x: int, y: int) -> None:  # row and column
        self.x = x
        self.y = y

    def change_position(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def idem_position(self, x: int, y: int) -> bool:
        return self.x == x and self.y == y

    @property
    def can_interact(self):
        return True

    @property
    def coordinates(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def numpy_coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Agent(Entity):
    def __init__(self, x: int, y: int, action_space_number: int = 4):
        super().__init__(x, y)
        self._action_space = gymnasium.spaces.Discrete(action_space_number, 0)
        self._initial_position = (x, y)

    def reset(self) -> None:
        self.change_position(*self._initial_position)

    @abstractmethod
    def act(self, action_id: int):
        pass

    @property
    def action_space(self):
        return self._action_space

    def __str__(self):
        return "A"


class PrimaryAgent(Agent):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, PRIMARY_AGENT_ACTION_SPACE_NUMBER)

    def act(self, action_id: int):
        action = Actions(action_id)
        x, y = self.coordinates

        if action == Actions.UP:
            y += 1
        elif action == Actions.DOWN:
            y -= 1
        elif action == Actions.LEFT:
            x -= 1
        elif action == Actions.RIGHT:
            x += 1

        self.change_position(x, y)

    
    def get_target_coordinates(self, action_id: int):
        action = Actions(action_id)
        x, y = self.coordinates

        if action == Actions.UP:
            y += 1
        elif action == Actions.DOWN:
            y -= 1
        elif action == Actions.LEFT:
            x -= 1
        elif action == Actions.RIGHT:
            x += 1

        return (x, y)


class SecondAgent(Agent):   
    def __init__(self, x: int, y: int):
        super().__init__(x, y, SECOND_AGENT_ACTION_SPACE_NUMBER)

    def act(self, action_id: int):
        action = Actions(action_id)
        x, y = self.coordinates

        if action == Actions.UP:
            y += 1
        elif action == Actions.DOWN:
            y -= 1
        elif action == Actions.LEFT:
            x -= 1
        elif action == Actions.RIGHT:
            x += 1

        self.change_position(x, y)


    def get_target_coordinates(self, action_id: int):
        action = Actions(action_id)
        x, y = self.coordinates

        if action == Actions.UP:
            y += 1
        elif action == Actions.DOWN:
            y -= 1
        elif action == Actions.LEFT:
            x -= 1
        elif action == Actions.RIGHT:
            x += 1

        return (x, y)


class Obstacle(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)

    @property
    def can_interact(self):
        return False

    def __str__(self):
        return "X"


class Empty(Entity):
    def __init__(self, x, y, label=" "):
        super().__init__(x, y)
        self.label = label

    def __str__(self):
        return self.label


"""
Enum with the actions that the agent can execute
"""


class Actions(IntEnum):
    UP = 0  # move up
    RIGHT = 1  # move right
    DOWN = 2  # move down
    LEFT = 3  # move left
    NONE = 4  # none or pick
    DROP = 5


class OfficeWorldObjects(StrEnum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    MAIL = "e"
    COFFEE_1 = "f" # add second coffee sign
    COFFEE_2 = "h" # add second coffee sign
    OFFICE = "g"
    PLANT = "n"
