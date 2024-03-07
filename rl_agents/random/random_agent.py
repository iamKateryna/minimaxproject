import random

from ..base.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        self.epsilon = 0.01 # to avoid errors when decaying epsilon for other agents


    def get_action(self, state = None):
        return random.choice(range(self.action_space.n))
    

    def name(self):
        return 'random'
