import random
from math import isclose
import numpy as np

class MinMaxQLearningAgent:
    def __init__(self, action_space, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.1, q_init=2) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}
        self.q_init = q_init # initial q-value for unseen states
        
        self.own_action_space = action_space
        self.opponent_action_space = action_space
        self.current_episode = None


    def get_qvalue(self, state, own_action, opponent_action):
        return self.q_table[state][own_action][opponent_action]

    
    def get_value(self, state):

        best_value = -float('inf')

        for action in range(self.own_action_space.n):
            for opponent_action in range(self.opponent_action_space.n):

                q_value = self.get_qvalue(state, action, opponent_action)

                if q_value > best_value:
                    best_value = q_value

        return best_value

    
    def get_policy(self, state):
        best_action = None
        best_action_value = -float('inf')

        all_q_values = []

        for action in range(self.own_action_space.n):
            max_value_for_action = -float('inf')

            for opponent_action in range(self.opponent_action_space.n):
                q_value = self.get_qvalue(state, action, opponent_action)
                all_q_values.append(q_value)

                if q_value > max_value_for_action:
                    max_value_for_action = q_value


            if max_value_for_action > best_action_value:
                best_action_value = max_value_for_action
                best_action = action

        print(f"All q-values: {all_q_values} ")
        return best_action


    def get_action(self, state, episode = None):
        # Exploration vs exploitation tradeof
        if random.random() < self.epsilon:
            return random.choice(range(self.own_action_space.n))
        else:
            # print(f"Exploitation")
            return self.get_policy(state)


    def init_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {}

        for action in range(self.own_action_space.n):
            if action not in self.q_table[state]:
                self.q_table[state][action] = {}

            for opponent_action in range(self.opponent_action_space.n):
                if opponent_action not in self.q_table[state][action]:
                    self.q_table[state][action][opponent_action] = self.q_init

    def learn(self, experience):
        for state, own_action, opponent_action, reward, next_state, done in experience:

            # check if next_state is in q_table
            if next_state not in self.q_table:
                self.init_q_values(next_state)

            # check if state is in q_table (important for countterfactual experiences)
            if state not in self.q_table:
                self.init_q_values(state)

            if done:
                value = reward
            else:
                value = reward + self.gamma * self.get_value(next_state)

            current_q = self.get_qvalue(state, own_action, opponent_action) # q_value of our action action at state state

            # print(f"self.q_table[state][own_action][opponent_action]: {self.q_table[state][own_action][opponent_action]} ")
            
            self.q_table[state][own_action][opponent_action] += self.lr * (value - current_q)
