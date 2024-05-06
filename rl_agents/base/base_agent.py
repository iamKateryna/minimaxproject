from abc import ABC, abstractmethod
from math import log
import pickle

class BaseAgent(ABC):
    def __init__(self, action_space, q_table, exploration_rate, learning_rate):
        self.action_space = action_space
        self.q_table = q_table

        self.epsilon = exploration_rate
        self.lr = learning_rate


    def decay_epsilon(self, num_steps):
        """Decay exploration rate exponantially for num_steps"""
        decay = 10**(log(0.01,10)/num_steps)
        self.epsilon = max(self.epsilon * decay, 0.01)

        """Decay exploration rate linearly for num_steps"""
        # decrement = (1.0 - 0.01) / num_steps
        # self.epsilon = max(self.epsilon - decrement, 0.01)


    def decay_lr(self, num_steps):
        """Decay learning rate for num_steps"""
        decay = 10**(log(0.01,10)/num_steps)
        self.lr = max(self.lr*decay, 0.01) 

    
    def save_policy(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)


    @abstractmethod
    def get_action(self, state = None):
        """Choose an action based on the state."""
        pass


    @abstractmethod
    def name(self):
        return 'base'