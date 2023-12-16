import random


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_init=0.02):
        self.lr = learning_rate
        self.gamma = discount_factor
        #self.epsilon = exploration_rate
        self.q_table = {}
        self.q_init = q_init # initial q-value for unseen states
        self.action_space = action_space

    
    def get_qvalue(self, state, action):
        return self.q_table.get((tuple(state), action), self.q_init)


    def get_value(self, state):
        q_table = [self.get_qvalue(state, action) for action in range(self.action_space.n)]
        return max(q_table)


    def get_policy(self, state):
        q_table = [self.get_qvalue(state, action) for action in range(self.action_space.n)]
        maxQ = max(q_table)
        best_actions = [action for action in range(self.action_space.n) if q_table[action] == maxQ] 
        return random.choice(best_actions)
    

    def get_action(self, state):
        return self.get_policy(state)
    

    def learn(self, state, action, reward, next_state, done):
        q_table = self.get_qvalue(state, action)
        if done:
            value = reward
        else: 
            value = reward + self.gamma * self.get_value(next_state)
        self.q_table[(state, action)] = q_table + self.lr * (value - q_table)
