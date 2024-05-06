from re import L
import logging
import random
from copy import copy
import functools
from .game_objects import *
from .office_world import OfficeWorld
from .office_world_simplified import OfficeWorldSimlified
from pettingzoo import ParallelEnv
from reward_machines.reward_machine_environment import RewardMachineEnv

class OfficeWorldEnv(ParallelEnv):
    metadata = {
        "name": "multi_agent_office_world_v0",
    }
    PRIMARY_AGENT_ID = "primary_agent"
    SECOND_AGENT_ID = "second_agent"

    def __init__(self, map_object, map_type, coffee_type, predator_prey, allow_stealing, agents_can_be_in_same_cell):
        self.map_type = map_type
        self.coffee_type = coffee_type # "unlimited"/"single"

        self.predator_prey = predator_prey
        self.allow_stealing = allow_stealing
        if self.predator_prey or self.allow_stealing:
            self.agents_can_be_in_same_cell = True
        else: 
            self.agents_can_be_in_same_cell = agents_can_be_in_same_cell

        if map_type=="base":
            self.office_world = OfficeWorld(map_object=map_object)
        elif map_type=="simplified":
            self.office_world = OfficeWorldSimlified(map_object=map_object)
        else:           
            raise NotImplementedError(f"Map for {map_type} is not implemented, available options -> 'simplified' and 'base'")

        self.possible_agents = [self.PRIMARY_AGENT_ID, self.SECOND_AGENT_ID]
        self.id_to_agent = {
            self.PRIMARY_AGENT_ID: self._generate_agent(PrimaryAgent, 1),
            self.SECOND_AGENT_ID: self._generate_agent(SecondAgent, 2),
        }
        self.timestep = 0
    

    @property
    def primary_agent(self):
        return self.id_to_agent[self.PRIMARY_AGENT_ID]

    @property
    def second_agent(self):
        return self.id_to_agent[self.SECOND_AGENT_ID]
    
    @property
    def all_agents(self):
        return self.possible_agents

    def _generate_agent(self, agent_class: type[Agent], agent_id) -> Agent:
        x, y = self.office_world.generate_coordinates()
        # logging.info(f'coords: ({x,y})' )
        # if self.predator_prey:
        #     if agent_id == 2:
        #         x, y = 0, 4
        #     else:
        #         x, y = 3, 3
        
        # else: 
        #     if agent_id == 2:
        #         x, y = 0, 3
        #     else:
        #         x, y = 0, 3

        return agent_class(x, y)

    def _get_observations(self):
        observations = {}

        for agent_id, agent in self.id_to_agent.items():
            agent_observation = agent.numpy_coordinates
            forbidden_actions = self.office_world.get_forbidden_actions(
                agent.coordinates
            )

            action_mask = [
                0 if a in forbidden_actions else 1 for a in range(agent.action_space.n)
            ]

            observations[agent_id] = {
                "observation": agent_observation,
                # "action_mask": action_mask,
            }

        return observations
    
     
    def _get_events(self):
        events = ""
        
        for agent_id, agent in self.id_to_agent.items():
             
            if agent_id == self.PRIMARY_AGENT_ID:
                agent_suffix = 1
            elif agent_id == self.SECOND_AGENT_ID:
                agent_suffix = 2
                
            if self.coffee_type == "unlimited":

                event = self.office_world.get_true_propositions(
                     agent.coordinates, agent_suffix)
                
            elif self.coffee_type == "single":

                event, self.coffee_1_available, self.coffee_2_available = self.office_world.get_true_propositions_single_coffee(
                    agent.coordinates, agent_suffix, self.coffee_1_available, self.coffee_2_available) # add second coffee sign
            else: 
                raise NotImplementedError(f"Events for coffee machine type {self.coffee_type} are not implemented, available options -> 'unlimited' and 'single'")

            events += event

        if self.predator_prey and (self.primary_agent.coordinates ==  self.second_agent.coordinates):
                events += 't' # t for trapped

        # if the stealing mode is on, the agent without coffee has a 50% chance of stealing a cup of coffee from the other agent
        # if self.allow_stealing and (self.primary_agent.coordinates ==  self.second_agent.coordinates):
        #         if random.random() > 0.5:
        #             events += 't' # t for trapped

        return events

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.id_to_agent[agent].action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.office_world.observation_space

    def reset(self, seed=None, options=None):
        self.coffee_1_available = True
        self.coffee_2_available = True
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.id_to_agent = {
            self.PRIMARY_AGENT_ID: self._generate_agent(PrimaryAgent, 1),
            self.SECOND_AGENT_ID: self._generate_agent(SecondAgent, 2),
        }

        # for agent in self.id_to_agent.values():
        #     agent.reset()

        observations = self._get_observations()
        infos = {agent_id: {} for agent_id, _ in self.id_to_agent.items()}

        return observations, infos

    def step(self, actions):

        if self.agents_can_be_in_same_cell:
            for agent_id, agent in self.id_to_agent.items():
                agent_action = actions[agent_id]

                if agent_action not in self.office_world.get_forbidden_actions(agent.coordinates):
                    agent.act(agent_action)
        
        else:
            # cannot move to the same cell
            target_coordinates = {}

            for agent_id, agent in self.id_to_agent.items():
                agent_action = actions[agent_id] # get action

                # check if it is allowed
                if agent_action not in self.office_world.get_forbidden_actions(agent.coordinates):
                    # if yes, calculate taget coordinates
                    target_coordinates[agent_id] = agent.get_target_coordinates(agent_action)

            # check, if target coordinates are the same
            if len(target_coordinates) == 2 and (target_coordinates["primary_agent"]==target_coordinates["second_agent"]):
                
                # if yes, randomly select an agent to execute action, other agents do not move
                selected_agent_id = random.choice(list(self.id_to_agent.keys()))
                # act
                selected_agent = self.id_to_agent[selected_agent_id]
                selected_agent.act(actions[selected_agent_id])

            else:
                
                # else, regular actions
                for agent_id, agent in self.id_to_agent.items():
                    agent_action = actions[agent_id]

                    if agent_action not in self.office_world.get_forbidden_actions(agent.coordinates):
                        agent.act(agent_action)

        self.timestep += 1

        terminations = {agent_id: False for agent_id,
                        _ in self.id_to_agent.items()}
        rewards = {agent_id: 0 for agent_id, _ in self.id_to_agent.items()}

        truncations = {agent_id: False for agent_id,
                       _ in self.id_to_agent.items()} # always false, RMs stop the game

        observations = self._get_observations()
        infos = {agent_id: {} for agent_id, _ in self.id_to_agent.items()}

        return observations, rewards, terminations, truncations, infos
    

    def show(self):
        # 12 by 9 map
        if self.map_type == "base":
            for y in range(8, -1, -1):
                if y % 3 == 2:
                    for x in range(12):
                        if x % 3 == 0:
                            print("_", end="")
                            if 0 < x < 11:
                                print("_", end="")
                        if (x, y, Actions.UP) in self.office_world.forbidden_transitions:
                            print("_", end="")
                        else:
                            print(" ", end="")
                    print()
                for x in range(12):
                    if (x, y, Actions.LEFT) in self.office_world.forbidden_transitions:
                        print("|", end="")
                    elif x % 3 == 0:
                        print(" ", end="")

                    if (x, y) == self.primary_agent.coordinates and (
                        x,
                        y,
                    ) == self.second_agent.coordinates:
                        print("12", end="")
                    elif (x, y) == self.primary_agent.coordinates:
                        print("1", end="")
                    elif (x, y) == self.second_agent.coordinates:
                        print("2", end="")
                    elif (x, y) in self.office_world.objects:
                        print(self.office_world.objects[(x, y)], end="")
                    else:
                        print(" ", end="")
                    if (x, y, Actions.RIGHT) in self.office_world.forbidden_transitions:
                        print("|", end="")
                    elif x % 3 == 2:
                        print(" ", end="")
                print()
                if y % 3 == 0:
                    for x in range(12):
                        if x % 3 == 0:
                            print("_", end="")
                            if 0 < x < 11:
                                print("_", end="")
                        if (x, y, Actions.DOWN) in self.office_world.forbidden_transitions:
                            print("_", end="")
                        else:
                            print(" ", end="")
                    print()
        
        # 6 by 9 map
        if self.map_type == "simplified":

            for y in range(5, -1, -1):
                if y % 3 == 2:
                    for x in range(9):
                        if x % 3 == 0:
                            print("_", end="")
                            if 0 < x < 8:
                                print("_", end="")
                        if (x, y, Actions.UP) in self.office_world.forbidden_transitions:
                            print("_", end="")
                        else:
                            print(" ", end="")
                    print()
                for x in range(9):
                    if (x, y, Actions.LEFT) in self.office_world.forbidden_transitions:
                        print("|", end="")
                    elif x % 3 == 0:
                        print(" ", end="")

                    if (x, y) == self.primary_agent.coordinates and (
                        x,
                        y,
                    ) == self.second_agent.coordinates:
                        print("12", end="")
                    elif (x, y) == self.primary_agent.coordinates:
                        print("1", end="")
                    elif (x, y) == self.second_agent.coordinates:
                        print("2", end="")
                    elif (x, y) in self.office_world.objects:
                        print(self.office_world.objects[(x, y)], end="")
                    else:
                        print(" ", end="")
                    if (x, y, Actions.RIGHT) in self.office_world.forbidden_transitions:
                        print("|", end="")
                    elif x % 3 == 2:
                        print(" ", end="")
                print()
                if y % 3 == 0:
                    for x in range(9):
                        if x % 3 == 0:
                            print("_", end="")
                            if 0 < x < 8:
                                print("_", end="")
                        if (x, y, Actions.DOWN) in self.office_world.forbidden_transitions:
                            print("_", end="")
                        else:
                            print(" ", end="")
                    print()
                    