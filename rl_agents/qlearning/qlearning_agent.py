import random
import math

from ..base.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, action_space, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.5, q_init=2, q_table={}):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_init = q_init # initial q-value for unseen states
        self.action_space = action_space

        self.q_table = q_table

        # calculate how ofthen an agent visins different states
        self.update_counts = {}

    
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

    def get_policy(self, state):
        q_values = [self.get_qvalue(state, action) for action in range(self.action_space.n)] # array of q_values per action at state state
        maxQ = max(q_values)

        best_actions = [action for action in range(self.action_space.n) if math.isclose(q_values[action],maxQ, rel_tol=1e-8)]
        
        # for stochastic policy
        # return random.choice(best_actions)

        # for deterministic policy
        return best_actions[0]

    # returns action for state state
    def get_action(self, state):
        if random.random() < self.epsilon:
            # return random.choice([action for action in range(self.action_space.n)])
            return random.choice(range(self.action_space.n))
        else:
            return self.get_policy(state)
        
    
    def init_q_values(self, state):
        self.q_table[state] = {action: self.q_init for action in range(self.action_space.n)}
        self.update_counts[state] = {action: 0 for action in range(self.action_space.n)}

    # experience = [(state, action, reward, next_state, done) (state, action, reward, next_state, done), ...]
    def learn(self, experience):
        i = 0
        for state, (action, ), reward, next_state, done in experience:

            # print(f"\n crm round {i}")
            # print(f"Inside learn(): State -> {state}, actions -> {action}, Next state -> {next_state}")

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
                # print(f"Value -> {value}")
                value = reward + self.gamma * self.get_value(next_state)
                
            q_value = self.get_qvalue(state, action) # q_value of our action action at state state

            # print(f"self.q_table[state][action]: {self.q_table[state][action]}\nValue {value}\nQ_value: {q_value} ")
            self.q_table[state][action] += self.lr * (value - q_value)
            # print(f"self.q_table[state] -> {self.q_table[state]}")
            i+=1
            self.update_counts[state][action] +=1

    def name(self):
        return 'qlearning'
