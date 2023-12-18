from pettingzoo.utils import BaseParallelWrapper
import numpy as np

    
class RewardMachineWrapper(BaseParallelWrapper):
    def __init__(self, env, add_crm, gamma):
        """
        RM wrapper
        --------------------
        It adds crm (counterfactual experience) to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_crm(bool):   if True, it will add a set of counterfactual experiences to info
            - gamma(float):    Discount factor for the environment
        """

        super().__init__(env)
        self.add_crm = add_crm
        self.valid_states = {agent_id: None for agent_id in self.env.id_to_reward_machine}

    def get_num_rm_states(self):
        return self.env.num_rm_states
    

    def reset(self):
        # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!) 
        for agent_id in self.valid_states:
            self.valid_states[agent_id] = None

        return self.env.reset()
    

    def step(self, actions):
        # TODO: add RMs and RM states before executing the action
        id_to_reward_machine = self.env.id_to_reward_machine
        # reward_machines = list(id_to_reward_machine.values())
        # current_rm_state_ids = self.env.current_rm_state_ids
        # executing the action in the environment
        reward_machine_observations, reward_machine_rewards, done, info = self.env.step(actions)

        # print(f'\nINFO: {info}\n')

        # print('UPDATING CRM EXPERIENCE\n')
        # print(f"CRM PARAMS: {self.env.crm_params}")

        # example of self.crm_params[agent_id]
        # self.crm_params[agent_id] = self.observations, actions[agent_id], next_observation, env_done, true_propositions
        if self.add_crm:
            for agent_id, agent_rm in id_to_reward_machine.items():
                # print(f"CRM PARAMS for {agent_id}: {self.env.crm_params[agent_id]}")

                crm_experience = self._get_crm_experience(agent_id, agent_rm, *self.env.crm_params[agent_id])
                info[agent_id]["crm-experience"] = crm_experience

        # print(f'INFO: {info}\n')

        return reward_machine_observations, reward_machine_rewards, done, info
    

    def _get_rm_experience(self, agent_id, reward_machine, rm_state_id,
                            observation, action, next_observation, 
                            done,true_propositions):
        
        reward_machine_observation = self.env.get_observation(observation, agent_id, done)
        # print(f'RM OBSERVATION: {reward_machine_observation}')
        next_rm_state_id, reward_machine_reward, reward_machine_done  = reward_machine.step(rm_state_id, true_propositions)
        # self.env.observation = next_observation
        done = reward_machine_done or done
        next_reward_machine_observation = self.env.get_observation(next_observation, agent_id, done)
        # print(f'NEXT RM OBSERVATION: {next_reward_machine_observation}')
        # print(np.array_equal(reward_machine_observation,next_reward_machine_observation))
        return (reward_machine_observation, action, reward_machine_reward, next_reward_machine_observation, done), next_rm_state_id
    

    def _get_crm_experience(self, agent_id, reward_machine, observation, action, next_observation, done, true_propositions):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (observation, action, reward, next_observation, done), ...]
        """
        reachable_states, experiences = set(), []
        valid_states = self.valid_states[agent_id]

        for rm_state_id in reward_machine.get_states():
            experience, next_rm_state = self._get_rm_experience(agent_id, reward_machine, rm_state_id, 
                                                                observation, action, next_observation, 
                                                                done, true_propositions)
            reachable_states.add((reward_machine, next_rm_state))
            if valid_states is None or (reward_machine, next_rm_state) in valid_states:
                experiences.append(experience)

        self.valid_states[agent_id] = reachable_states

        return experiences
