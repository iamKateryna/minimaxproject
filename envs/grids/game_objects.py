from enum import Enum, StrEnum
import gymnasium.spaces
from abc import abstractmethod
from constants import *

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

    def idem_position(self, x: int, y: int) -> None:
        return self.x == x and self.y == y

    @property
    def can_interact(self):
        return True

    @property
    def coordinates(self) -> tuple[int, int]:
        return (self.x, self.y)


class Agent(Entity):
    def __init__(self, x: int, y: int, action_space_number: int):
        super().__init__(x, y)
        self.agent_name = agent_name
        self._action_space = gymnasium.spaces.Discrete(action_space_number, 0)
        self._initial_position = (x, y)

    def reset(self):
        self.change_position(*self._initial_position)
    
    @abstractmethod
    def do_action(self, action: str):
        pass

    @property
    def action_space(self):
        return self._action_space

    def __str__(self):
        return "A"

class PrimaryAgent(Agent):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, action_space, PRIMARY_AGENT_ACTION_SPACE_NUMBER)
    
    def do_action(self, action_id: int):
        action = Actions(a)
        x, y = self.coordinates
        
        if action == Actions.up: y+=1
        elif action == Actions.down: y-=1
        elif action == Actions.left: x-=1
        elif action == Actions.right: x+=1

        self.change_position(x, y)

class SecondAgent(Agent):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, action_space, SECOND_AGENT_ACTION_SPACE_NUMBER)
    
    def do_action(self, action_id: int):
        action = Actions(a)
        x, y = self.coordinates
        
        if action == Actions.up: y+=1
        elif action == Actions.down: y-=1
        elif action == Actions.left: x-=1
        elif action == Actions.right: x+=1

        self.change_position(x, y)

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


class Actions(Enum):
    up = 0  # move up
    right = 1  # move right
    down = 2  # move down
    left = 3  # move left
    none = 4  # none or pick
    drop = 5


class OfficeWorldObjects(StrEnum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    MAIL = "e"
    COFFEE = "f"
    OFFICE = "g"
    PLANT = "n"
