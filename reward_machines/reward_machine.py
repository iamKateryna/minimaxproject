from .reward_machine_utils import evaluate_dnf
from .reward_functions import *
import time


class RewardMachine:
    def __init__(self, file):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_r    = {} # reward-transition function
        self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*
        self._load_reward_machine(file)
        self.known_transitions = {} # Auxiliary variable to speed up computation of the next RM state


    def reset(self):
        # Returns the initial state
        return self.u0

    def _compute_next_state(self, u, true_propositions):
        for next_u in self.delta_u[u]:
            if evaluate_dnf(self.delta_u[u][next_u], true_propositions):
                return next_u
        return self.terminal_u # if no transition is defined for true_propositions
    

    def get_next_state(self, u, true_propositions):
        if (u,true_propositions) not in self.known_transitions:
            next_u = self._compute_next_state(u, true_propositions)
            self.known_transitions[(u,true_propositions)] = next_u
        return self.known_transitions[(u,true_propositions)]


    def step(self, u, true_propositions, state_info, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u != self.terminal_u, "the RM was set to a terminal state!"
        next_u = self.get_next_state(u, true_propositions)
        done = (next_u == self.terminal_u)
        # Getting the reward
        reward = self._get_reward(u,next_u,state_info, env_done)

        return next_u, reward, done
    

    def get_states(self):
        return self.U
    
    # figure out if we need env_done here
    def _get_reward(self, u, next_u, s_info, env_done): 
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        if u in self.delta_r and next_u in self.delta_r[u]:
            reward += self.delta_r[u][next_u].get_reward(s_info)
        # Returning final reward
        return reward
    

    def _load_reward_machine(self, file):
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = [line.rstrip() for line in f]
        f.close()
        # setting the initial state
        self.u0 = eval(lines[0])
        # setting the terminal state
        terminal_states = eval(lines[1])

        # adding transitions
        for transition in lines[2:]:
            # Reading the transition
            u, next_u, dnf_formula, reward_function = eval(transition)
            # terminal states

            if u in terminal_states:
                continue

            if next_u in terminal_states:
                next_u  = self.terminal_u

            # Adding machine state
            self._add_state([u, next_u])
            # Adding state-transition to delta_u

            if u not in self.delta_u:
                self.delta_u[u] = {}

            self.delta_u[u][next_u] = dnf_formula

            # Adding reward-transition to delta_r
            if u not in self.delta_r:
                self.delta_r[u] = {}

            self.delta_r[u][next_u] = reward_function
            
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)
