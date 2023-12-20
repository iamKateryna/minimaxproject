import random
from math import isclose
import numpy as np

class MinMaxQLearningAgent:
    def __init__(self, action_space, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.1, q_init=0.5) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}
        self.q_init = q_init # initial q-value for unseen states
        
        self.own_action_space = action_space
        self.opponent_action_space = action_space
        self.current_episode = None


    def get_qvalue(self, state, own_action, opponent_action):
        joint_action = (own_action, opponent_action) 
        state_key = (tuple(state), joint_action)

        return self.q_table.get((state_key), self.q_init)

    
    def get_min_qvalue(self, state):
        # Find minimum q-value over ALL joint actions
        min_q = float("inf")
        min_q_own_actions = set()
        
        for own_action in range(self.own_action_space.n):
            for opp_action in range(self.opponent_action_space.n):
                q = self.get_qvalue(state, own_action, opp_action)
                
                if isclose(q, min_q, abs_tol=1e-8):
                    min_q_own_actions.add(own_action)
                elif q < min_q:
                    min_q = q
                    min_q_own_actions = {own_action}
                    
        return min_q, min_q_own_actions
    

    def get_max_qvalue(self, state):
        # Find maximum q-value over ALL joint actions 
        max_q = float("-inf")
        max_q_own_actions = set()
        
        for own_action in range(self.own_action_space.n):
            for opp_action in range(self.opponent_action_space.n):
                q = self.get_qvalue(state, own_action, opp_action)
                
                if isclose(q, max_q, abs_tol=1e-8):
                    max_q_own_actions.add(own_action)
                elif q > max_q:
                    max_q = q
                    max_q_own_actions = {own_action}
                
        return max_q, max_q_own_actions

    def get_action(self, state, episode = None):
        # Exploration vs exploitation tradeof
        if random.random() < self.epsilon:
            return random.choice(range(self.own_action_space.n))
        
        # _, max_q_actions = self.get_max_qvalue(state)
        
        # Minimax Strategy
        # minimax_action = None
        # minimax_value = float('inf')
        # tie_actions = []

        # for own_action in range(self.own_action_space.n):
        #     worst_case = float('-inf')  # Initialize to a very low value
        #     for opp_action in range(self.opponent_action_space.n):
        #         q_value = self.get_qvalue(state, own_action, opp_action)
        #         worst_case = max(worst_case, q_value)  # Find the worst-case scenario for each action

        #     # Check for ties and update minimax action
        #     if isclose(worst_case, minimax_value, abs_tol=1e-8):
        #         tie_actions.append(own_action)
        #     elif worst_case < minimax_value:
        #         minimax_value = worst_case
        #         minimax_action = own_action
        #         tie_actions = [own_action]

        # # Handle tie cases by randomly choosing among them
        # if tie_actions:
        #     minimax_action = random.choice(tie_actions)

        # if episode > 0:
        #     print(f"minimax_action -> {minimax_action}")

        # return minimax_action if minimax_action is not None else random.choice(range(self.own_action_space.n))
        minimax_action = None
        minimax_value = float("inf")

        for a in range(self.own_action_space.n):
            worst_case = self.get_min_qvalue(state)[0] 
            if worst_case < minimax_value:
                minimax_action = a
                minimax_value = worst_case

        return minimax_action
    

    def learn(self, experience):
        for state, own_action, opp_action, reward, next_state, done in experience:
        
            if done:
                new_q = reward  
            else:
                min_q, _ = self.get_min_qvalue(next_state)
                max_q, _ = self.get_max_qvalue(next_state)
                new_q = reward + (self.gamma * min_q*max_q)
                
            curr_q = self.get_qvalue(state, own_action, opp_action)
            
            q_table_key = (tuple(state), (own_action, opp_action))
            self.q_table[q_table_key] = curr_q + (self.lr * (new_q - curr_q))

    # def learn(self, experience):
    #     for state, own_action, opp_action, reward, next_state, done in experience:
    #         if done:
    #             new_q = reward  
    #         else:
    #             min_q = self.get_min_qvalue(next_state)
    #             new_q = reward + (self.gamma * min_q)

    #         curr_q = self.get_qvalue(state, own_action, opp_action)
    #         self.q_table[(tuple(state), own_action, opp_action)] = curr_q + (self.lr * (new_q - curr_q))
# def get_action(self, state):
        
#         # Minimax Strategy
#         minimax_values = []
#         for own_action in range(self.own_action_space.n):
#             worst_case = float('inf')  # Initialize to a very high value
#             for opp_action in range(self.opponent_action_space.n):
#                 q_value = self.get_qvalue(state, own_action, opp_action)
#                 worst_case = min(worst_case, q_value)  # Find the worst-case scenario for each action
#             minimax_values.append(worst_case)

#         # Choose the action with the maximum of the worst-case values (Minimax action)
#         minmax_action = np.argmin(minimax_values)
#         return minmax_action