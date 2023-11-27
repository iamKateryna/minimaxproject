from envs.grids.game_objects import Actions
import random, math, os
import numpy as np
from game_objects import Agent, OfficeWorldObjects
from enum import StrEnum

class OfficeWorld:
    OBSTACLE_OBJECT_TYPES = [OfficeWorldObjects.PLANT]

    MAP_1_OBJECTS = {(1,1): OfficeWorldObjects.A,
                    (1,7): OfficeWorldObjects.B,
                    (10,7): OfficeWorldObjects.C,
                    (10,1): OfficeWorldObjects.D,
                    (7,4): OfficeWorldObjects.MAIL,
                    (8,2): OfficeWorldObjects.COFFEE,
                    (3,6): OfficeWorldObjects.COFFEE,
                    (4,4): OfficeWorldObjects.OFFICE,
                    (4,1): OfficeWorldObjects.PLANT,
                    (7,1): OfficeWorldObjects.PLANT,
                    (4,7): OfficeWorldObjects.PLANT,
                    (7,7): OfficeWorldObjects.PLANT,
                    (1,4): OfficeWorldObjects.PLANT,
                    (10,4): OfficeWorldObjects.PLANT
    }
    
    def __init__(self, map_number = 1):
        self._load_map(map_number)
        self.map_height, self.map_width = 12,9
    
    def _get_action_space(self) -> list[Actions]:
        return [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]
    
    def _get_agent_by_number(self, agent_number: int) -> Agent:
        if agent_number == 1:
            return self.agent_1
        elif agent_number == 2:
            return self.agent_2
        else:
            raise NotImplementedError
    
    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x, y, action) not in self.forbidden_transitions:
            if action == Actions.up: y+=1
            elif action == Actions.down: y-=1
            elif action == Actions.left: x-=1
            elif action == Actions.right: x+=1

        return x, y

    def _get_obstacle_coordinates(self) -> list[tuple[int, int]]:
        obstacle_coordinates = []

        for coordinates, object in self.objects:
            if object in self.OBSTACLE_OBJECT_TYPES:
                obstacle_coordinates.append(coordinates)
        
        return obstacle_coordinates
    
    def _load_agents(self) -> None:
        def generate_coordinates(exception_coordinates):
            x, y = None, None
            
            while not x or y or (x, y) in exception_coordinates:
                x = random.randint(1, self.map_height)
                y = random.randint(1, self.map_width)
            
            return x, y
                    
        obstacle_coordinates = self._get_obstacle_coordinates()
        agent1_x, agent1_y = generate_coordinates(obstacle_coordinates)
        agent2_x, agent2_y = generate_coordinates(obstacle_coordinates)
        action_space = self._get_action_space()

        self.agent_1 = Agent(agent1_x, agent1_y, action_space)
        self.agent_2 = Agent(agent2_x, agent2_y, action_space)
    
    def _load_forbidden_transaction(self) -> None:
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down)) 
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
            
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1, 10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        
    def _load_map_objects(self, map_number) -> None:
        if map_number == 1:
            self.objects = self.MAP_1_OBJECTS
        else:
            raise NotImplementedError

    def _load_map(self, map_number) -> None:
        self._load_map_objects(map_number)
        self._load_forbidden_transaction()
        self._load_agents()

    def execute_action(self, agent_number, a):
        """
        We execute 'action' in the game
        """
        agent = self._get_agent_by_number(agent_number)
        x,y = agent.get_coordinates()
        agent = self._get_new_position(x, y, a)
    
    def get_features(self, agent_number: int):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        agent = self._get_agent_by_number(agent_number)
        coordinates = agent.get_coordinates()
        return np.array(coordinates)

    def get_true_propositions(self, agent_number: int):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        agent = self._get_agent_by_number(agent_number)

        if agent in self.objects:
            agent_coordinates = agent.get_coordinates()
            ret += self.objects[agent_coordinates]

        return ret
    
        def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        S = [(x,y) for x in range(12) for y in range(9)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD

    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    def reset(self):
        self.agent_1.reset()
        self.agent_2.reset()