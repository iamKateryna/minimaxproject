from pettingzoo.utils import BaseParallelWrapper
import numpy as np

    
class RewardMachineWrapper(BaseParallelWrapper):
    def __init__(self, env, add_crms):
        """
        RM wrapper
        --------------------
        It adds crm (counterfactual experience) to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_crm(bool):   if True, it will add a set of counterfactual experiences to info
        """

        super().__init__(env)
        self.add_crms = add_crms
        self.valid_states = {agent_id: None for agent_id in self.env.id_to_reward_machine}
        self.agent_ids = self.env.id_to_reward_machine

    def get_num_rm_states(self):
        return self.env.num_rm_states
    

    def reset(self):
        # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!) 
        for agent_id in self.valid_states:
            self.valid_states[agent_id] = None

        return self.env.reset()
    


    def step(self, actions, agent_type, episode = None):
        # print(f"WRAPPER: actions -> {actions}, agent_type -> {agent_type}, episode -> {episode}")
        # executing the action in the environment
        # true_propositions and rm_state are for progress tracking purposes
        reward_machine_observations, reward_machine_rewards, done, info, true_propositions, rm_state = self.env.step(actions, agent_type, episode)


        for id, agent_id in enumerate(self.agent_ids):
            print(f"id: {id}, agent_id -> {agent_id}, add_crms -> {self.add_crms}")

            if self.add_crms[id]:
                crm_experience = self._get_crm_experience(agent_id, *self.env.crm_params[agent_id])
                info[agent_id]["crm-experience"] = crm_experience

        return reward_machine_observations, reward_machine_rewards, done, info, true_propositions, rm_state
    

    def _get_crm_experience(self, agent_id, observation, actions, next_observation, done, true_propositions):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (observation, action, reward, next_observation, done), ...]
        """
        reachable_states, experiences = set(), []

        agents_valid_state = self.valid_states[agent_id]
        reward_machine = self.env.id_to_reward_machine[agent_id]


        for rm_state_id in reward_machine.get_states():
            experience, next_rm_state = self._get_rm_experience(agent_id, reward_machine, rm_state_id, 
                                                                observation, actions, next_observation, 
                                                                done, true_propositions)
            
            reachable_states.add((reward_machine, next_rm_state))

            if agents_valid_state is None or (reward_machine, rm_state_id) in agents_valid_state:
                experiences.append(experience)

        self.valid_states[agent_id] = reachable_states

        return experiences
    

    def _get_rm_experience(self, agent_id, reward_machine, rm_state_id,
                            observation, actions, next_observation, 
                            done,true_propositions):
        

        reward_machine_observation = self.env.get_observation(observation, agent_id, rm_state_id, done)

        next_rm_state_id, reward_machine_reward, reward_machine_done  = reward_machine.step(rm_state_id, true_propositions)
        done = reward_machine_done or done

        next_reward_machine_observation = self.env.get_observation(next_observation, agent_id, next_rm_state_id, done)

        return (reward_machine_observation, actions, reward_machine_reward, next_reward_machine_observation, done), next_rm_state_id
    

    # def _get_rm_experience(self, agent_id, reward_machine, rm_state_id,
    #                         observation, action, next_observation, 
    #                         done,true_propositions):
        

    #     reward_machine_observation = self.env.get_observation(observation, agent_id, rm_state_id, done)

    #     next_rm_state_id, reward_machine_reward, reward_machine_done  = reward_machine.step(rm_state_id, true_propositions)
    #     done = reward_machine_done or done

    #     next_reward_machine_observation = self.env.get_observation(next_observation, agent_id, next_rm_state_id, done)

    #     return (reward_machine_observation, action, reward_machine_reward, next_reward_machine_observation, done), next_rm_state_id
    

    # def _get_rm_experience_minmax(self, agent_id, reward_machine, rm_state_id,
    #                         observation, action, other_agent_action, next_observation, 
    #                         done,true_propositions):
        
    #     reward_machine_observation = self.env.get_observation(observation, agent_id, rm_state_id, done)
    #     next_rm_state_id, reward_machine_reward, reward_machine_done  = reward_machine.step(rm_state_id, true_propositions)
    #     done = reward_machine_done or done
    #     next_reward_machine_observation = self.env.get_observation(next_observation, agent_id, next_rm_state_id, done)
    #     return (reward_machine_observation, action, other_agent_action, reward_machine_reward, next_reward_machine_observation, done), next_rm_state_id
    
    
    # def _get_crm_experience(self, agent_id, observation, action, next_observation, done, true_propositions):
    #     """
    #     Returns a list of counterfactual experiences generated per each RM state.
    #     Format: [..., (observation, action, reward, next_observation, done), ...]
    #     """
    #     reachable_states, experiences = set(), []

    #     agents_valid_state = self.valid_states[agent_id]
    #     reward_machine = self.env.id_to_reward_machine[agent_id]


    #     for rm_state_id in reward_machine.get_states():
    #         experience, next_rm_state = self._get_rm_experience(agent_id, reward_machine, rm_state_id, 
    #                                                             observation, action, next_observation, 
    #                                                             done, true_propositions)
            
    #         reachable_states.add((reward_machine, next_rm_state))

    #         if agents_valid_state is None or (reward_machine, rm_state_id) in agents_valid_state:
    #             experiences.append(experience)

    #     self.valid_states[agent_id] = reachable_states

    #     return experiences
    

    # def _get_crm_experience_minmax(self, agent_id, observation, action, other_agent_action, next_observation, done, true_propositions):
    #     """
    #     Returns a list of counterfactual experiences generated per each RM state.
    #     Format: [..., (observation, action, reward, next_observation, done), ...]
    #     """
    #     reachable_states, experiences = set(), []

    #     agents_valid_state = self.valid_states[agent_id]
    #     reward_machine = self.env.id_to_reward_machine[agent_id]

    #     for rm_state_id in reward_machine.get_states():
    #         experience, next_rm_state = self._get_rm_experience_minmax(agent_id, reward_machine, rm_state_id, 
    #                                                             observation, action, other_agent_action, next_observation, 
    #                                                             done, true_propositions)
    #         reachable_states.add((reward_machine, next_rm_state))

    #         # print(f"Current rm state -> true_propositions -> new rm state: {rm_state_id} -> {true_propositions} -> {next_rm_state} ")

    #         # print(f"Adding crm to experience -> {agents_valid_state is None or (reward_machine, next_rm_state) in agents_valid_state}")

    #         if agents_valid_state is None or (reward_machine, rm_state_id) in agents_valid_state:
    #             experiences.append(experience)

    #     self.valid_states[agent_id] = reachable_states

    #     return experiences
