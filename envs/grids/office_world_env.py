import numpy as np
import gymnasium.spaces
from game_objects import *

from envs.grids.office_world import OfficeWorld
from pettingzoo import ParallelEnv

class OfficeWorldEnv(ParallelEnv):
    metadata = {
        "name": "multi_agent_office_world_v0",
    }

    def __init__(self):
        self.office_world = OfficeWorld()
        self.possible_agents = ["primary_agent", "second_agent"]
        self.id_to_agent = {
            "primary_agent": self._generate_agent(PrimaryAgent),
            "second_agent" : self._generate_agent(SecondAgent)
        }

    def _generate_agent(self, agent_class: type[Agent]) -> Agent:
        x, y = self.office_world.generate_coordinates()

        return agent_class(x, y)

    def reset(self, seed=None, options=None):
        self.primary_agent.reset()
        self.second_agent.reset()

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.id_to_agent[agent].action_space
