import random
import math
import numpy as np
import logging

from ..base.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, action_space, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.0, q_init=2, policy = 'boltzmann', q_table={}):
        self.q_table = q_table
        
        self.epsilon = exploration_rate
        self.lr = learning_rate

        self.gamma = discount_factor
        self.q_init = q_init # initial q-value for unseen states
        self.action_space = action_space

        self.policy = policy

    
    def get_qvalue(self, state, action):
        return self.q_table[state][action]


    def get_value(self, state):
        q_values = [self.get_qvalue(state, action) for action in range(self.action_space.n)] # array of q_values per action at state state
        
        max_ = float("-inf")
        max_values = set()
        
        for q_value in q_values:
            if math.isclose(q_value, max_):
                max_values.add(q_value)
            elif q_value > max_:
                max_ = q_value
                max_values.clear()
                
        if len(max_values) > 1:
            return random.choice(tuple(max_values))
            
        return max_

    def get_policy(self, state, episode_num, print_on):
        q_values = [self.get_qvalue(state, action) for action in range(self.action_space.n)] # array of q_values per action at state state
        if self.policy == 'epsilongreedy':
            if random.random() < self.epsilon:
                return random.choice(range(self.action_space.n))
            else:
                maxQ = max(q_values)
                best_actions = [action for action in range(self.action_space.n) if math.isclose(q_values[action],maxQ, rel_tol=1e-8)]
                return random.choice(best_actions)
                # return best_actions[0]
        elif self.policy == 'boltzmann':
            if self.epsilon == 0.0:
                return random.choice(best_actions)
            else:
                q_values = np.array(q_values)
                max_q = np.max(q_values)
                probabilities = np.exp(q_values-max_q / self.epsilon)

                if episode_num and episode_num > print_on:
                        logging.info(f' probabilities: {probabilities}')
                
                probabilities_sum = probabilities.sum()
                if probabilities_sum > 0:
                        probabilities /= probabilities_sum
                else:
                    # Handle the case where all exp_values are zero by assigning equal probability to all actions
                    probabilities = np.ones_like(probabilities) / len(probabilities)

                if episode_num and episode_num > print_on:
                        logging.info(f'normalized probabilities: {probabilities}')

                return np.random.choice(self.action_space.n, p=probabilities)
        else:
            raise NotImplementedError

    # returns action for state state
    def get_action(self, state, episode_num=None, print_on=16_000):
        action = self.get_policy(state, episode_num, print_on)

        if episode_num and episode_num > print_on:
            logging.info(f"action: {action}")

        return action
        
    
    def init_q_values(self, state):
        self.q_table[state] = {action: self.q_init for action in range(self.action_space.n)}
        # self.update_counts[state] = {action: 0 for action in range(self.action_space.n)}

    # experience = [(state, action, reward, next_state, done) (state, action, reward, next_state, done), ...]
    def learn(self, experience, episode_num=None, print_on=16_000):
        i = 0
        for state, (action, ), reward, next_state, done in experience:

            if state not in self.q_table:
                self.init_q_values(state)
            
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
                
            q_value = self.get_qvalue(state, action) # q_value of our action action at state state

            self.q_table[state][action] += self.lr * (value - q_value)

            if episode_num and episode_num > print_on:
                logging.info(f"--- {i} --- state: {state}")
                logging.info(f"val: {self.q_table[state][action]}")

            i+=1

    def name(self):
        return 'qlearning'
