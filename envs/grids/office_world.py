from envs.grids.game_objects import Actions
import random
import math
import os
import numpy as np
from game_objects import *
from pettingzoo import ParallelEnv
from collections import defaultdict


class OfficeWorld:
    OBSTACLE_OBJECT_TYPES = [OfficeWorldObjects.PLANT]

    MAP_1_OBJECTS = {(1, 1): OfficeWorldObjects.A,
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
                     (10, 4): OfficeWorldObjects.PLANT
                     }

    def __init__(self, map_height: int = 12, map_width: int = 9, map_number: int = 1):
        self._load_map(map_number)
        self.map_height = map_height
        self.map_width = map_width
        self.forbidden_transactions = self._load_forbidden_transactions()

    def _get_obstacle_coordinates(self) -> list[tuple[int, int]]:
        obstacle_coordinates = []

        for coordinates, object in self.objects:
            if object in self.OBSTACLE_OBJECT_TYPES:
                obstacle_coordinates.append(coordinates)

        return obstacle_coordinates

    def generate_coordinates(self) -> tuple[int, int]:
        obstacle_coordinates = self._get_obstacle_coordinates()
        x, y = None, None

        while not x or y or (x, y) in exception_coordinates:
            x = random.randint(1, self.map_height)
            y = random.randint(1, self.map_width)

        return x, y

    def _load_forbidden_transactions(self) -> None:
        forbidden_transitions = defaultdict(set)

        for x in range(self.map_height):
            for y in [0, 3, 6]:
                self.forbidden_transitions[(x, y)].add(Actions.down)
                self.forbidden_transitions[(x, y + 2)].add(Actions.up)
        for y in range(self.map_width):
            for x in [0, 3, 6, 9]:
                self.forbidden_transitions[(x, y)].add(Actions.left)
                self.forbidden_transitions[(x + 2, y)].add(Actions.right)

        # adding 'doors'
        for y in [1, 7]:
            for x in [2, 5, 8]:
                self.forbidden_transitions[(x, y)].remove(Actions.right)
                self.forbidden_transitions[(x+1, y)].remove(Actions.left)

        for x in [1, 4, 7, 10]:
            self.forbidden_transitions[(x, 5)].remove(Actions.up)
            self.forbidden_transitions[(x, 6)].remove(Actions.down)
        for x in [1, 10]:
            self.forbidden_transitions[(x, 2)].remove(Actions.up)
            self.forbidden_transitions[(x, 3)].remove(Actions.down)

        return forbidden_transitions

    def _load_map_objects(self, map_number) -> None:
        if map_number == 1:
            self.objects = self.MAP_1_OBJECTS
        else:
            raise NotImplementedError

    def _load_map(self, map_number) -> None:
        self._load_map_objects(map_number)

    @property
    def forbidden_transactions(self) -> dict[tuple[int, int], Action]:
        return self.forbidden_transactions

    @property
    def shape(self) -> dict[tuple[int, int], Action]:
        return (self.map_height, self.map_width)

    def get_true_propositions(self, agent_coordinates: int):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""

        if agent_coordinates in self.objects:
            ret += self.objects[agent_coordinates]

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
