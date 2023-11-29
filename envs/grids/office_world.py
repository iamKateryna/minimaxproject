from types import NoneType
from typing import Type
import random
import numpy as np
from .game_objects import *
from collections import defaultdict
import gymnasium.spaces


class OfficeWorld:
    OBSTACLE_OBJECT_TYPES = [OfficeWorldObjects.PLANT]

    MAP_1_OBJECTS: dict[tuple[int, int], str] = {
        (1, 1): OfficeWorldObjects.A,
        (1, 7): OfficeWorldObjects.B,
        (10, 7): OfficeWorldObjects.C,
        (10, 1): OfficeWorldObjects.D,
        (7, 4): OfficeWorldObjects.MAIL,
        (8, 2): OfficeWorldObjects.COFFEE,
        (3, 6): OfficeWorldObjects.COFFEE,
        (4, 4): OfficeWorldObjects.OFFICE,
        (4, 1): OfficeWorldObjects.PLANT,
        (7, 1): OfficeWorldObjects.PLANT,
        (4, 7): OfficeWorldObjects.PLANT,
        (7, 7): OfficeWorldObjects.PLANT,
        (1, 4): OfficeWorldObjects.PLANT,
        (10, 4): OfficeWorldObjects.PLANT,
    }

    def __init__(self, map_height: int = 12, map_width: int = 9, map_number: int = 1):
        self._objects = self._load_map_objects(map_number)
        self._map_height = map_height
        self._map_width = map_width
        self._forbidden_actions = self._load_forbidden_actions()
        self._forbidden_transitions = self._load_forbidden_transitions()

    def _get_obstacle_coordinates(self) -> list[tuple[int, int]]:
        obstacle_coordinates = []

        for coordinates, object in self._objects:
            if object in self.OBSTACLE_OBJECT_TYPES:
                obstacle_coordinates.append(coordinates)

        return obstacle_coordinates

    def generate_coordinates(self) -> tuple[int, int]:
        obstacle_coordinates = self._get_obstacle_coordinates()
        x, y = None, None

        while not x or not y or (x, y) in obstacle_coordinates:
            x = random.randint(1, self._map_height)
            y = random.randint(1, self._map_width)

        return x, y

    def _load_forbidden_transitions(self) -> set[tuple[int, int, Actions]]:
        if not self._forbidden_actions:
            raise TypeError

        return {
            (*location, action)
            for location, forbidden_actions in self._forbidden_actions.items()
            for action in forbidden_actions
        }

    def _load_forbidden_actions(self) -> dict[tuple[int, int], set[Actions]]:
        location_to_forbidden_actions = defaultdict(set)

        for x in range(self._map_height):
            for y in [0, 3, 6]:
                location_to_forbidden_actions[(x, y)].add(Actions.down)
                location_to_forbidden_actions[(x, y + 2)].add(Actions.up)
        for y in range(self._map_width):
            for x in [0, 3, 6, 9]:
                location_to_forbidden_actions[(x, y)].add(Actions.left)
                location_to_forbidden_actions[(x + 2, y)].add(Actions.right)

        # adding 'doors'
        for y in [1, 7]:
            for x in [2, 5, 8]:
                location_to_forbidden_actions[(x, y)].remove(Actions.right)
                location_to_forbidden_actions[(x + 1, y)].remove(Actions.left)

        for x in [1, 4, 7, 10]:
            location_to_forbidden_actions[(x, 5)].remove(Actions.up)
            location_to_forbidden_actions[(x, 6)].remove(Actions.down)
        for x in [1, 10]:
            location_to_forbidden_actions[(x, 2)].remove(Actions.up)
            location_to_forbidden_actions[(x, 3)].remove(Actions.down)

        return location_to_forbidden_actions

    def get_forbidden_actions(self, coordinates) -> set[Actions]:
        return self._forbidden_actions[coordinates]

    def _load_map_objects(
        self, map_number
    ) -> dict[tuple[int, int], str]:
        if map_number == 1:
            objects = self.MAP_1_OBJECTS
        else:
            raise NotImplementedError

        return objects

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
    def objects(self) -> dict[tuple[int, int], str]:
        return self._objects

    def get_true_propositions(self, agent_coordinates: tuple[int, int]):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""

        if agent_coordinates in self._objects:
            ret += self._objects[agent_coordinates]

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
