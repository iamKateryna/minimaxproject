import random
import logging
import math
import numpy as np

from ..base.base_agent import BaseAgent

class MinMaxQLearningAgent(BaseAgent):
    def __init__(self, action_space, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.5, q_init=2, q_table = {}, policy = 'boltzmann'):
        self.q_table = q_table

        self.epsilon = exploration_rate
        self.lr = learning_rate
        
        self.gamma = discount_factor
        self.q_init = q_init # initial q-value for unseen states
        self.own_action_space = action_space
        self.opponent_action_space = action_space

        self.policy = policy


    def get_qvalue(self, state, own_action, opponent_action):
        return self.q_table[state][own_action][opponent_action]
    

    def get_value(self, state):

        # best_value = -float('inf')
        worst_value = float('inf')


        for action in range(self.own_action_space.n):
            for opponent_action in range(self.opponent_action_space.n):

                q_value = self.get_qvalue(state, action, opponent_action)

                # if q_value > best_value:
                #     best_value = q_value

                if q_value < worst_value:
                    worst_value = q_value
            # logging.info(f'- GET VALUE: worst val {worst_value}')

        # return best_value
        return worst_value

    
    # def get_policy(self, state):
    #     best_action = [] # default action
    #     best_action_value = -float('inf')

    #     all_q_values = []

    #     for action in range(self.own_action_space.n):
    #         # max_value_for_action = -float('inf')
    #         min_value_for_action = float('inf')

    #         for opponent_action in range(self.opponent_action_space.n):
    #             q_value = self.get_qvalue(state, action, opponent_action)
    #             all_q_values.append(q_value)

    #             # if q_value > max_value_for_action:
    #             #     max_value_for_action = q_value

    #             if q_value < min_value_for_action:
    #                 min_value_for_action = q_value


    #         # if max_value_for_action > best_action_value:
    #         #     best_action_value = max_value_for_action
    #         #     best_action = action

    #         # logging.info(f'--- GET POLICY: in val per action: {min_value_for_action}, current best action val: {best_action_value}')
    #         if min_value_for_action > best_action_value:
    #             best_action_value = min_value_for_action
    #             best_action = [action]
    #         # for stochastic policy
    #         elif math.isclose(min_value_for_action,best_action_value, rel_tol=1e-8):
    #             best_action.append(action)

    #     # return best_action
    #     # for stochastic policy
    #     return random.choice(best_action)
    

    def get_policy(self, state, episode_num, print_on):
        best_action = [] # default action
        best_action_value = -float('inf')
        # minimal q_values per each our action
        min_q_values = []

        for action in range(self.own_action_space.n):
            # max_value_for_action = -float('inf')
            min_value_for_action = float('inf')

            for opponent_action in range(self.opponent_action_space.n):
                q_value = self.get_qvalue(state, action, opponent_action)

                if q_value < min_value_for_action:
                    min_value_for_action = q_value
                
            min_q_values.append(min_value_for_action)
            

            if min_value_for_action > best_action_value:
                best_action_value = min_value_for_action
                best_action = [action]
            # for stochastic policy
            elif math.isclose(min_value_for_action,best_action_value, rel_tol=1e-8):
                best_action.append(action)

        if episode_num and episode_num > print_on:
            logging.info(f'min_q_values: {min_q_values}')

        if self.policy == 'epsilongreedy':
            if random.random() < self.epsilon:
                return random.choice(range(self.own_action_space.n))
            else:
                return random.choice(best_action)
        elif self.policy == 'boltzmann':
            if self.epsilon == 0.0:
                return random.choice(best_action)
            else:
                min_q_values = np.array(min_q_values)
                max_q = np.max(min_q_values)
                probabilities = np.exp(min_q_values-max_q / self.epsilon)

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
                    
                return np.random.choice(self.own_action_space.n, p=probabilities)
        else:
            raise NotImplementedError


    def get_action(self, state, episode_num=None, print_on=16_000):
        action = self.get_policy(state, episode_num, print_on)

        if episode_num and episode_num > print_on:
            logging.info(f"action: {action}")

        return action


    def init_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {}

        for action in range(self.own_action_space.n):
            if action not in self.q_table[state]:
                self.q_table[state][action] = {}

            for opponent_action in range(self.opponent_action_space.n):
                if opponent_action not in self.q_table[state][action]:
                    self.q_table[state][action][opponent_action] = self.q_init

    def learn(self, experience, episode_num=None, print_on=16_000):
        i = 0
        for state, (own_action, opponent_action), reward, next_state, done in experience:

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
            
            new_q = self.lr * (value - current_q)

            self.q_table[state][own_action][opponent_action] += new_q

            if episode_num and episode_num > print_on:
                logging.info(f"--- {i} --- state: {state}")
                logging.info(f"val: {self.q_table[state][own_action][opponent_action]}")

            i+=1
            

    def name(self):
        return 'minmaxq'
