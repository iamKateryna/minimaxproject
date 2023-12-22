from pettingzoo.utils import BaseParallelWrapper
from .reward_machine import RewardMachine
from envs.officeWorld.value_iteration import value_iteration

from gymnasium import spaces
import numpy as np


class RewardMachineEnv(BaseParallelWrapper):
    def __init__(self, env, reward_machine_files):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve an RM task
            - This code keeps track of the current state on the RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of 2 strings with paths to RM files, rm_files[0] - agent_1 RM, rm_files[1] - agent_2 RM 
        """
        super().__init__(env)

        self.id_to_reward_machine = {
            self.env.PRIMARY_AGENT_ID: RewardMachine(reward_machine_files[0]),
            self.env.SECOND_AGENT_ID: RewardMachine(reward_machine_files[1]),
        }

        self.reward_machines = list(self.id_to_reward_machine.values())
        # self.num_rm_states = len(self.reward_machines[0].get_states()) * 2 # agents are identical, their RMs have the same number of states
        self.num_rm_states = len(self.reward_machines[0].get_states())

        self.observation_dict = spaces.Dict({
                                            'features': spaces.Dict({
                                                'primary_agent': spaces.Dict({
                                                    'observation': env.observation_space(self.env.PRIMARY_AGENT_ID),
                                                    'action_mask': spaces.MultiBinary(4)
                                                    }),
                                                'second_agent': spaces.Dict({
                                                    'observation': env.observation_space(self.env.SECOND_AGENT_ID),
                                                    'action_mask': spaces.MultiBinary(4)
                                                    })
                                                                    }),
                                            'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        flatdim = spaces.flatdim(self.observation_dict)
        space_low = float(env.observation_space(self.env.PRIMARY_AGENT_ID).low[0])
        space_high = float(env.observation_space(self.env.PRIMARY_AGENT_ID).high[0])
        self.observation_space = spaces.Box(low=space_low, high=space_high, shape=(flatdim,), dtype=np.float32)

        # Computing one-hot encodings for the non-terminal RM states
        self.reward_machine_state_features = {agent_id: {} for agent_id in self.id_to_reward_machine}

        for agent_id, agent_rm in self.id_to_reward_machine.items():
            for rm_state_id in agent_rm.get_states():
                u_features = np.zeros(self.num_rm_states)
                u_features[len(self.reward_machine_state_features[agent_id])] = 1
                self.reward_machine_state_features[agent_id][rm_state_id] = u_features
        # for terminal RM states, we give as features an array of zeros, same for both RMs
        self.reward_machine_done_features = np.zeros(self.num_rm_states)
        self.crm_params = {}

    @property
    def _primary_agent_rm(self):
        return self.id_to_reward_machine[self.env.PRIMARY_AGENT_ID]

    @property
    def _second_agent_rm(self):
        return self.id_to_reward_machine[self.env.SECOND_AGENT_ID]
    
    # Reset the environment, add the RMs state to the observation
    def reset(self):
        self.observation, _ = self.env.reset()
        self.current_rm_state_ids =  {self.env.PRIMARY_AGENT_ID: self._primary_agent_rm.reset(),
                                      self.env.SECOND_AGENT_ID: self._second_agent_rm.reset()}
        
        reward_machine_dones = {agent_id: False for agent_id, _ in self.id_to_reward_machine.items()}

        reward_machines_observation = {}

        for agent_id, _ in self.id_to_reward_machine.items():
            reward_machines_observation[agent_id] = self.get_observation(self.observation, agent_id,  self.current_rm_state_ids[agent_id], reward_machine_dones[agent_id])

        return reward_machines_observation
    

    def step(self, actions, agent_type, episode = None):
        print(f"actions -> {actions}, agent_type -> {agent_type}, episode -> {episode}")
        next_observation, _, env_done, _, info = self.env.step(actions)

        # getting the output of the detectors
        true_propositions = self.env._get_events()

        # init reward_machine_rewards, reward_machine_dones
        reward_machine_rewards = {agent_id: 0 for agent_id, _ in self.id_to_reward_machine.items()}
        reward_machine_dones = {agent_id: False for agent_id, _ in self.id_to_reward_machine.items()}
        # print(f'RM STATE IDs {self.current_rm_state_ids}')
        # print(f"episode -> {episode}")

        if agent_type == 'qlearning':

            for agent_id, agent_rm in self.id_to_reward_machine.items():
                # update RMs state
                self.current_rm_state_ids[agent_id], reward_machine_rewards[agent_id], reward_machine_dones[agent_id] = agent_rm.step(self.current_rm_state_ids[agent_id],
                                                                                                                                    true_propositions)
                # print(f"agent_id: {agent_id}, true_propositions: {true_propositions}, rm_state_id: {self.current_rm_state_ids} ")
                
                # saving information for generating counterfactual experiences
                self.crm_params[agent_id] = self.observation, actions[agent_id], next_observation, reward_machine_dones[agent_id], true_propositions

        elif agent_type == 'minmax':

            for agent_id, agent_rm in self.id_to_reward_machine.items():

                # define other_agent_id to keep track of other agent actions
                other_agent_id = "primary_agent" if "second_agent" == agent_id else "second_agent"

                # update RMs state
                self.current_rm_state_ids[agent_id], reward_machine_rewards[agent_id], reward_machine_dones[agent_id] = agent_rm.step(self.current_rm_state_ids[agent_id],
                                                                                                                                    true_propositions)
                # print(f"agent_id: {agent_id}, true_propositions: {true_propositions}, rm_state_id: {self.current_rm_state_ids} ")
                
                # saving information for generating counterfactual experiences
                self.crm_params[agent_id] = self.observation, actions[agent_id], actions[other_agent_id], next_observation, reward_machine_dones[agent_id], true_propositions

        else: 
            raise NotImplementedError(f"CRM updates for {agent_type} are not implemented, available options -> 'minmax' or 'qlearning' ")

                
        # print(f'RM STATE IDs {self.current_rm_state_ids}')
        self.observation = next_observation
        reward_machine_observations = {}    
        done = reward_machine_dones 
        for agent_id, agent_rm in self.id_to_reward_machine.items():
            reward_machine_observations[agent_id] = self.get_observation(next_observation, agent_id,  self.current_rm_state_ids[agent_id],  done[agent_id])
        
        # return true_propositions and self.current_rm_state_ids for tracking progress
        return reward_machine_observations, reward_machine_rewards, done, info, true_propositions, self.current_rm_state_ids
    
    
    # add RM state to the observation
    def get_observation(self, observation, agent_id, rm_state_id, reward_machine_done):

        if reward_machine_done:
            reward_machine_features = self.reward_machine_done_features 
        else:
            # reward_machine_features = self.reward_machine_state_features[(agent_id, self.current_rm_state_ids[agent_id])]
            reward_machine_features = self.reward_machine_state_features[agent_id][rm_state_id]

        reward_machine_observations = {'features': observation, 
                                       'rm-state': reward_machine_features}
        
        return spaces.flatten(self.observation_dict, reward_machine_observations)
    

    def render(self, mode="human"):
        if mode == "human":
            # commands
            str_to_action = {"w": 0, "d": 1, "s": 2, "a": 3}

            # play the game!
            done = True

            while True:
                if done:
                    print("-------------------------------- New episode --------------------------------")
                    observations = self.reset()
                    # print("Current task:", self.rm_files[self.current_rm_id])
                    self.env.show()
                    print("Features:", observations)
                    print("Events:", self.env._get_events())
                    print("RM state:", self.current_rm_state_ids)
                    done = False

                print(
                    "\nSelect action for the primary agent (WASD keys or q to quit): ",
                    end="",
                )
                action1 = input()

                if action1 == "q":
                    break

                print(
                    "\nSelect action for the second agent (WASD keys or q to quit): ",
                    end="",
                )
                action2 = input()
                print()

                if action2 == "q":
                    break

                # Executing action
                if action1 in str_to_action and action2 in str_to_action:
                    actions_to_execute = {
                        self.env.PRIMARY_AGENT_ID: str_to_action[action1],
                        self.env.SECOND_AGENT_ID: str_to_action[action2],
                    }

                    _, rewards, done, _ = self.step(actions_to_execute)
                    done = any(done.values())

                    self.env.show()
                    print("Features:", observations)
                    print("Events:", self.env._get_events())
                    print("RM state:", self.current_rm_state_ids)
                    print("Reward:", rewards)
                else:
                    print("Forbidden action")

                if done:
                    if rewards[self.env.PRIMARY_AGENT_ID] == 2 or rewards[self.env.SECOND_AGENT_ID] == 2:
                        print('\nThe supreme art of war is to subdue the enemy without fighting\n')
                        break
                    elif rewards[self.env.PRIMARY_AGENT_ID] == 1 or rewards[self.env.SECOND_AGENT_ID] == 1:
                        print('\nYou were born to win\n')
                        break
                    else:
                        print("\nSometimes the prize is not worth the costs\n")
                        break
        else:
            raise NotImplementedError
        
    # def test_optimal_policies(self, num_episodes, epsilon, gamma):
    #     """
    #     This code computes optimal policies for each reward machine and evaluates them using epsilon-greedy exploration

    #     PARAMS
    #     ----------
    #     num_episodes(int): Number of evaluation episodes
    #     epsilon(float):    Epsilon constant for exploring the environment
    #     gamma(float):      Discount factor

    #     RETURNS
    #     ----------
    #     List with the optimal average-reward-per-step per reward machine
    #     """
    #     S,A,L,T = self.env.get_model()
    #     print("\nComputing optimal policies... ", end='', flush=True)
    #     optimal_policies = [value_iteration(S,A,L,T,reward_machine,gamma) for reward_machine in self.reward_machines]
    #     print("Done!")
    #     optimal_ARPS = [[] for _ in range(len(optimal_policies))]
    #     print("\nEvaluating optimal policies.")
    #     for ep in range(num_episodes):
    #         if ep % 100 == 0 and ep > 0:
    #             print("%d/%d"%(ep,num_episodes))
    #         self.reset()
    #         s = tuple(self.obs)
    #         u = self.current_u_id
    #         rm_id = self.current_rm_id
    #         rewards = []
    #         done = False
    #         while not done:
    #             a = random.choice(A) if random.random() < epsilon else optimal_policies[rm_id][(s,u)]
    #             _, r, done, _ = self.step(a)
    #             rewards.append(r)
    #             s = tuple(self.obs)
    #             u = self.current_u_id
    #         optimal_ARPS[rm_id].append(sum(rewards)/len(rewards))
    #     print("Done!\n")

        return [sum(arps)/len(arps) for arps in optimal_ARPS]
 