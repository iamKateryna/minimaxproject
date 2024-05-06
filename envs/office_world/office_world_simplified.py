from .game_objects import *
from .map_collection import MapCollection
from .office_world import OfficeWorld


class OfficeWorldSimlified(OfficeWorld):

    def __init__(self, map_height: int = 9, map_width: int = 6, map_object = MapCollection.MAP_4_OBJECTS):
        self._objects = map_object
        self._map_height = map_height
        self._map_width = map_width
        self._forbidden_actions = self._load_forbidden_actions()
        self._forbidden_transitions = self._load_forbidden_transitions()

    def _add_doors(self, location_to_forbidden_actions) -> dict[tuple[int, int], set[Actions]]:

        # for y in [1]:
        #     for x in [5]:
        #         location_to_forbidden_actions[(x, y)].remove(Actions.RIGHT)
        #         location_to_forbidden_actions[(x + 1, y)].remove(Actions.LEFT)
        
        for y in [1, 4]:
            for x in [2, 5]:
                location_to_forbidden_actions[(x, y)].remove(Actions.RIGHT)
                location_to_forbidden_actions[(x + 1, y)].remove(Actions.LEFT)

        for x in [1, 4, 7]:
            location_to_forbidden_actions[(x, 2)].remove(Actions.UP)
            location_to_forbidden_actions[(x, 3)].remove(Actions.DOWN)

        return location_to_forbidden_actions