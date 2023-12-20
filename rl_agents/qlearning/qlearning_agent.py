import random
import math


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.1, q_init=2):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}
        self.q_init = q_init # initial q-value for unseen states
        self.action_space = action_space
    
    def get_qvalue(self, state, action):
        # state_q = self.q_table.get(state)

        return self.q_table[state][action]

        # if state_q:
        #     return state_q.get(action, self.q_init)

        # return self.q_init
        
        # return self.q_table.get(state.get(action)), self.q_init)


    def get_value(self, state):
        q_values = [self.get_qvalue(state, action) for action in range(self.action_space.n)] # array of q_values per action at state state

        return max(q_values)

    # returns best_action for state state

    def get_policy(self, state):
        q_values = [self.get_qvalue(state, action) for action in range(self.action_space.n)] # array of q_values per action at state state
        maxQ = max(q_values)

        # best_actions = [action for action in range(self.action_space.n) if q_values[action] == maxQ]
        best_actions = [action for action in range(self.action_space.n) if math.isclose(q_values[action],maxQ, rel_tol=1e-8)]
        print(f"All q-values: {[q_values[action] for action in range(self.action_space.n)]} ")
        print(f"best_actions -> {best_actions}, maxQ -> {maxQ}")
        
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

    # experience = [(state, action, reward, next_state, done) (state, action, reward, next_state, done), ...]
    def learn(self, experience):
        for state, action, reward, next_state, done in experience:
            # check if next_state is in q_table
            if next_state not in self.q_table:
                self.init_q_values(next_state)

            # check if state is in q_table (important for countterfactual experiences)
            if state not in self.q_table:
                self.init_q_values(state)

            q_value = self.get_qvalue(state, action) # q_value of our action action at state state

            if done:
                value = reward
            else: 
                value = reward + self.gamma * self.get_value(next_state)
            
            print(f"self.q_table[state][action]: {self.q_table[state][action]}\nValue {value}\nQ_value: {q_value} ")
            self.q_table[state][action] += self.lr * (value - q_value)
            print(f"self.q_table[state] -> {self.q_table[state]}")
