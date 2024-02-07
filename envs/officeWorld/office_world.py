from types import NoneType
from typing import Type
import random
import numpy as np
from .game_objects import *
from collections import defaultdict
import gymnasium.spaces
from .map_collection import MapCollection


class OfficeWorld:
    OBSTACLE_OBJECT_TYPES = [OfficeWorldObjects.PLANT, 
                             OfficeWorldObjects.MAIL,
                             OfficeWorldObjects.COFFEE_1,
                             OfficeWorldObjects.COFFEE_2,
                             OfficeWorldObjects.OFFICE]


    def __init__(self, map_height: int = 12, map_width: int = 9, map_object = MapCollection.MAP_4_OBJECTS):
        self._objects = map_object
        self._map_height = map_height
        self._map_width = map_width
        self._forbidden_actions = self._load_forbidden_actions()
        self._forbidden_transitions = self._load_forbidden_transitions()

    def _get_obstacle_coordinates(self) -> list[tuple[int, int]]:
        obstacle_coordinates = []

        for coordinates, object_ in self._objects.items():
            if object_ in self.OBSTACLE_OBJECT_TYPES:
                obstacle_coordinates.append(coordinates)

        return obstacle_coordinates

    def _load_forbidden_transitions(self) -> set[tuple[int, int, Actions]]:
        if not self._forbidden_actions:
            raise TypeError

        return {
            (*location, action)
            for location, forbidden_actions in self._forbidden_actions.items()
            for action in forbidden_actions
        }
    

    def _add_doors(self, location_to_forbidden_actions) -> dict[tuple[int, int], set[Actions]]:

        for y in [1, 7]:
            for x in [2, 5, 8]:
                location_to_forbidden_actions[(x, y)].remove(Actions.RIGHT)
                location_to_forbidden_actions[(x + 1, y)].remove(Actions.LEFT)

        for x in [1, 4, 7, 10]:
            location_to_forbidden_actions[(x, 5)].remove(Actions.UP)
            location_to_forbidden_actions[(x, 6)].remove(Actions.DOWN)
        for x in [1, 10]:
            location_to_forbidden_actions[(x, 2)].remove(Actions.UP)
            location_to_forbidden_actions[(x, 3)].remove(Actions.DOWN)

        return location_to_forbidden_actions


    def _load_forbidden_actions(self) -> dict[tuple[int, int], set[Actions]]:
        location_to_forbidden_actions = defaultdict(set)

        # forbid going through all walls
        for x in range(self._map_height):
            for y in range(0, self._map_width, 3):
                location_to_forbidden_actions[(x, y)].add(Actions.DOWN)
                location_to_forbidden_actions[(x, y + 2)].add(Actions.UP)
        for y in range(self._map_width):
            for x in range(0, self._map_height, 3):
                location_to_forbidden_actions[(x, y)].add(Actions.LEFT)
                location_to_forbidden_actions[(x + 2, y)].add(Actions.RIGHT)

        # add 'doors' - map-specific
        location_to_forbidden_actions = self._add_doors(location_to_forbidden_actions)

        # for y in [1, 7]:
        #     for x in [2, 5, 8]:
        #         location_to_forbidden_actions[(x, y)].remove(Actions.RIGHT)
        #         location_to_forbidden_actions[(x + 1, y)].remove(Actions.LEFT)

        # for x in [1, 4, 7, 10]:
        #     location_to_forbidden_actions[(x, 5)].remove(Actions.UP)
        #     location_to_forbidden_actions[(x, 6)].remove(Actions.DOWN)
        # for x in [1, 10]:
        #     location_to_forbidden_actions[(x, 2)].remove(Actions.UP)
        #     location_to_forbidden_actions[(x, 3)].remove(Actions.DOWN)

        return location_to_forbidden_actions

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(
            low=0,
            high=max([self._map_height, self._map_width]),
            shape=(2,),
            dtype=np.uint8,
        )

    @property
    def forbidden_transitions(self) -> set[tuple[int, int, Actions]]:
        return self._forbidden_transitions

    # @property
    # def forbidden_actions(self) -> dict[tuple[int, int], set[Actions]]:
    #     return self._forbidden_actions

    @property
    def shape(self) -> tuple[int, int]:
        return (self._map_height, self._map_width)

    @property
    def objects(self) -> dict[tuple[int, int], OfficeWorldObjects]:
        return self._objects

    def generate_coordinates(self) -> tuple[int, int]:
        obstacle_coordinates = self._get_obstacle_coordinates()
        x, y = None, None

        while not x or not y or (x, y) in obstacle_coordinates:
            x = random.randint(0, self._map_height - 1)
            y = random.randint(0, self._map_width - 1)

        return x, y

    def get_forbidden_actions(self, coordinates) -> set[Actions]:
        return self._forbidden_actions[coordinates]
    

    def get_true_propositions_single_coffee(self, agent_coordinates: tuple[int, int], suffix: int, coffee_1: bool, coffee_2: bool): #add second coffee sign (also change rms)
        """
        Returns the string with the propositions that are True in this state, 
        checks for available coffee in the coffee machines
        """

        ret = ""
        
        # Check if agent is on the coffee cell and coffee is available
        if agent_coordinates in self._objects:
            proposition = self._objects[agent_coordinates]
            # if coffee available, get coffee and make it unavailable
            if proposition == "f" and coffee_1:
                ret += "f" + str(suffix)
                coffee_1 = False
            # if coffee unavailable, do nothing
            elif proposition == "f" and not coffee_1:
                pass
            elif proposition == "h" and coffee_2:
                ret += "h" + str(suffix)
                coffee_2 = False
            # if coffee unavailable, do nothing
            elif proposition == "h" and not coffee_2:
                pass
            # any other proposition -> regular propositions
            else:
                ret += proposition
                ret += str(suffix)
                
        return ret, coffee_1, coffee_2
        

    def get_true_propositions(self, agent_coordinates: tuple[int, int], suffix: int): 
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""

        # regular propositions
        if agent_coordinates in self._objects:
            ret += self._objects[agent_coordinates]
            ret += str(suffix)

        return ret

    # def get_model(self):
    #     """
    #     This method returns a model of the environment.
    #     We use the model to compute optimal policies using value iteration.
    #     The optimal policies are used to set the average reward per step of each task to 1.
    #     """
    #     S = [(x, y) for x in range(self.map_height) for y in range(self.map_width)]  # States
    #     A = self.actions.copy()  # Actions
    #     L = self.objects.copy()  # Labeling function
    #     # Transitions (s,a) -> s' (they are deterministic)
    #     T = {}

    #     for state in S:
    #         x, y = s

    #     for s in S:
    #         x, y = s
    #         for a in A:
    #             T[(s, a)] = self._get_new_position(x, y, a)
    #     return S, A, L, T  # SALT xD
