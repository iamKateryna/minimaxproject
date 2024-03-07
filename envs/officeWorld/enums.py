from enum import Enum

class MapType(Enum):
    SIMPLIFIED = "simplified"
    BASE = "base"

class CoffeeType(Enum):
    SINGLE = "single"
    UNLIMITED = "unlimited"

class ExplorationDecay(Enum):
    EPISODE = "episode"
    STEP = "step"

class AgentType(Enum):
    MINMAX = "minmax"
    QLEARNING = "qlearning"
    RANDOM = "random"
