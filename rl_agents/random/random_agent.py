import random


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}

    def get_action(self, state = None, episode = None):
        return random.choice(range(self.action_space.n))
    
    def name(self):
        return 'random'
