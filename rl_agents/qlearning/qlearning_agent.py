import random
import math


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.95, min_epsilon = 0.2, decay_rate = 0.9995, q_init=2, q_table=None):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_init = q_init # initial q-value for unseen states
        self.action_space = action_space
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        if not q_table:
            self.q_table = {}
        else:
            self.q_table = q_table


    def decay_lr(self):
        self.lr = max(self.lr*0.9999954, 0.01)


    def decay_epsilon(self):
        # Update epsilon using exponential decay
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self.epsilon = max(self.epsilon - ((1-0.0001)/100000000), self.min_epsilon)

    
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
    def get_action(self, state, episode = None):
        # if episode and episode%5:
        # self.decay_epsilon()
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

        self.decay_lr()

    def name(self):
        return 'qlearning'
