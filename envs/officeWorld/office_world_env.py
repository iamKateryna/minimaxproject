from re import L
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

    def __init__(self, map_object, map_type):
        self.map_type = map_type

        if map_type=="base":
            self.office_world = OfficeWorld(map_object=map_object)
        elif map_type=="simplified":
            self.office_world = OfficeWorldSimlified(map_object=map_object)
        else:           
            raise NotImplementedError

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
        # place both agents next to the closest coffee to the office
        # x, y = 3, 7
        # place both agents on an equal distance from both coffee machines
        if agent_id == 2:
            x, y = 6, 0
        else:
            x, y = 6, 0

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
                "action_mask": action_mask,
            }

        return observations
    
     
    def _get_events(self):
        events = ""
        
        for agent_id, agent in self.id_to_agent.items():
             
            if agent_id == self.PRIMARY_AGENT_ID:
                agent_suffix = 1
            elif agent_id == self.SECOND_AGENT_ID:
                agent_suffix = 2
                
            event = self.office_world.get_true_propositions(
                agent.coordinates, agent_suffix)
            events += event

        return events

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.id_to_agent[agent].action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.office_world.observation_space

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        for agent in self.id_to_agent.values():
            agent.reset()

        observations = self._get_observations()
        infos = {agent_id: {} for agent_id, _ in self.id_to_agent.items()}

        return observations, infos

    def step(self, actions):
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
            selected_agent_id = random.choice(self.id_to_agent.keys())
            # act
            selected_agent = self.id_to_agent[selected_agent_id]
            selected_agent.act(actions[selected_agent_id])
        else:
            # else, regular actions
            for agent_id, agent in self.id_to_agent.items():
                agent_action = actions[agent_id]

                if agent_action not in self.office_world.get_forbidden_actions(agent.coordinates):
                    agent.act(agent_action)




        # agent_ids = list(self.id_to_agent.keys())
        # random.shuffle(agent_ids)

        # for agent_id in agent_ids:
        #     agent = self.id_to_agent[agent_id]
        #     agent_action = actions[agent_id]

        #     target_coordinates = agent.get_target_coordinates(agent_action)

        # for agent_id, agent in self.id_to_agent.items():
        #     agent_action = actions[agent_id]

        #     if agent_action not in self.office_world.get_forbidden_actions(agent.coordinates):
        #         agent.act(agent_action)

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


    # def render(self, mode="human"):
    #     if mode == "human":
    #         # commands
    #         str_to_action = {"w": 0, "d": 1, "s": 2, "a": 3}

    #         # play the game!
    #         done = True
    #         while True:
    #             if done:
    #                 print("New episode --------------------------------")
    #                 observations = self.reset()
    #                 self.show()
    #                 print("Features:", observations)
    #                 print("Events:", self._get_events())
    #                 done = False

    #             print(
    #                 "\nSelect action for the primary agent?(WASD keys or q to quite) ",
    #                 end="",
    #             )
    #             action1 = input()

    #             if action1 == "q":
    #                 break

    #             print(
    #                 "\nSelect action for the second agent?(WASD keys or q to quite) ",
    #                 end="",
    #             )
    #             action2 = input()
    #             print()

    #             if action2 == "q":
    #                 break

    #             # Executing action
    #             if action1 in str_to_action and action2 in str_to_action:
    #                 actions_to_execute = {
    #                     self.PRIMARY_AGENT_ID: str_to_action[action1],
    #                     self.SECOND_AGENT_ID: str_to_action[action2],
    #                 }

    #                 self.step(actions_to_execute)
    #                 self.show()
    #                 print("Events:", self._get_events())
    #             else:
    #                 print("Forbidden action")
    #     else:
    #         raise NotImplementedError
