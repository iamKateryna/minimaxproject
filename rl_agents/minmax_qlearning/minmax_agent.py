import random

class MinMaxQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_init=0.02) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}
        self.q_init = q_init # initial q-value for unseen states
        
        self.own_action_space = action_space
        self.opponent_action_space = action_space


    def get_qvalue(self, state, own_action, opponent_action):
        joint_action = (own_action, opponent_action) 
        state_key = (tuple(state), joint_action)

        return self.q_table.get((state_key), self.q_init)

    
    def get_min_qvalue(self, state):

        # Find minimum q-value over ALL joint actions
        min_q = float("inf")
        
        for own_action in range(self.own_action_space.n):
            for opp_action in range(self.opponent_action_space.n):
                
                q = self.get_qvalue(state, own_action, opp_action)
                min_q = min(min_q, q)
                    
        return min_q
    

    def get_max_qvalue(self, state):

        # Find maximum q-value over ALL joint actions 
        max_q = float("-inf")
        
        for own_action in range(self.own_action_space.n):
            for opp_action in range(self.opponent_action_space.n):
                
                q = self.get_qvalue(state, own_action, opp_action)
                max_q = max(max_q, q)
                
        return max_q
    

    def get_action(self, state):

        # Exploration vs exploitation tradeof
        if random.random() < self.epsilon:
            return random.choice(range(self.own_action_space.n))
        
        best_actions = []
        max_q = self.get_max_qvalue(state)
        for own_action in range(self.own_action_space.n):
            for opp_action in range(self.opponent_action_space.n):
                q = self.get_qvalue(state, own_action, opp_action)
                if q == max_q:
                    best_actions.append(own_action)
        return random.choice(best_actions) if best_actions else random.choice(range(self.own_action_space.n))
    

    def learn(self, experience):

        for state, own_action, opp_action, reward, next_state, done in experience:
        
            if done:
                new_q = reward  
            else:
                min_q = self.get_min_qvalue(next_state)
                new_q = reward + (self.gamma * min_q)
                
            curr_q = self.get_qvalue(state, own_action, opp_action)  
            self.q_table[(tuple(state), own_action, opp_action)] = curr_q + (self.lr * (new_q - curr_q))

    # def learn(self, experience):
    #     for state, own_action, opp_action, reward, next_state, done in experience:
    #         if done:
    #             new_q = reward  
    #         else:
    #             min_q = self.get_min_qvalue(next_state)
    #             new_q = reward + (self.gamma * min_q)

    #         curr_q = self.get_qvalue(state, own_action, opp_action)
    #         self.q_table[(tuple(state), own_action, opp_action)] = curr_q + (self.lr * (new_q - curr_q))

