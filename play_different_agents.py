import logging
from tqdm import tqdm
from datetime import datetime
import os
import pickle

from rl_agents.random.random_agent import RandomAgent
from rl_agents.minmax_qlearning.minmax_agent import MinMaxQLearningAgent
from rl_agents.qlearning.qlearning_agent import QLearningAgent
from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

from envs.officeWorld.map_collection import MapCollection
from utils import setup_logger

import rm_constants


def main(filename, q_init, learning_rate, discount_factor, use_crm, map_object, map_type, reward_machine_files, map_number, total_timesteps=100000, max_episode_length=1000, print_freq=0):
    setup_logger(filename)

    policy_directory = 'policies'

    if not os.path.exists(f'{policy_directory}/{map_type}-{map_number}'):
        os.makedirs(f'{policy_directory}/{map_type}-{map_number}')

    office_env = OfficeWorldEnv(map_object=map_object, map_type=map_type)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crm=use_crm, gamma = 0.5)

    # print(office_env.possible_agents)
     
    action_space = office_env.primary_agent.action_space
    
    # learning_agents = {agent_id: MinMaxQLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor) 
    #           for agent_id in office_env.possible_agents}

    learning_agents = {office_env.possible_agents[0]: MinMaxQLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor),
                       office_env.possible_agents[1]: QLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor)}

    logging.info(f'USE CRM: {use_crm}, MAP: {map_object}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}')
    logging.info(f"lr -> {learning_rate}, q_init -> {q_init}, discount_factor -> {discount_factor}")

    reward_total = {agent_id: 0 for agent_id in office_env.all_agents}
    wins_total = {agent_id: 0 for agent_id in office_env.all_agents}
    broken_decorations_score = {agent_id: 0 for agent_id in office_env.all_agents}

    num_steps = 0
    num_episodes = 0

    while num_steps < total_timesteps:
        num_steps_per_episode = 0
        
        # reward_total = {agent_id: 0 for agent_id in office_env.all_agents}
        # wins_total = {agent_id: 0 for agent_id in office_env.all_agents}
        # broken_decorations_score = {agent_id: 0 for agent_id in office_env.all_agents}
        # print(f'Start Episode {num_episodes +1}')

        reset_env_state = env.reset()
        state = {agent_id: tuple(reset_env_state[agent_id]) for agent_id in office_env.all_agents}
        done = False
        num_episodes+=1

        while not done and num_steps_per_episode < max_episode_length:
            num_steps_per_episode += 1
            # print(" ------------- NEW STEP ------------- ")
            actions_to_execute = {}

            # Selecting and executing the action
            for agent_id in office_env.all_agents:
                agent, agent_state = learning_agents[agent_id], state[agent_id]

                if agent_state not in agent.q_table:
                    # print(f"creating new agent_state -> {agent_state}")
                    agent.init_q_values(agent_state)

                action = agent.get_action(agent_state)
                actions_to_execute[agent_id] = action

            # print(f"actions_to_execute -> {actions_to_execute}")

            next_state, rewards, done, info, true_propositions, rm_state = env.step(actions_to_execute, agent_type = 'minmax', episode = num_episodes)
            done = any(done.values())

            # Updating the q-values
            # print(" --- UPDARING Q-VALUES --- ")
            for agent_id in office_env.all_agents:

                other_agent_id = office_env.all_agents[0] if office_env.all_agents[1] == agent_id else office_env.all_agents[1]

                if isinstance(learning_agents[agent_id], MinMaxQLearningAgent):
                    if use_crm:
                        experiences = []
                        for _state, _action, _other_agent_action, _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                            experiences.append((tuple(_state), _action, _other_agent_action, _reward, tuple(_next_state), _done))
                    else:
                        experiences = [(state[agent_id], actions_to_execute[agent_id], actions_to_execute[other_agent_id], rewards[agent_id], tuple(next_state[agent_id]), done)]
                        
                    learning_agents[agent_id].learn(experiences)

                elif isinstance(learning_agents[agent_id], QLearningAgent): 
                    if use_crm:
                        experiences = []
                        for _state, _action, _, _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                            experiences.append((tuple(_state), _action, _reward, tuple(_next_state), _done))
                    else:
                        experiences = [(state[agent_id], actions_to_execute[agent_id], rewards[agent_id], tuple(next_state[agent_id]), done)]
                
                    learning_agents[agent_id].learn(experiences)

                elif isinstance(learning_agents[agent_id], RandomAgent):
                    pass
                else: 
                    raise NotImplementedError


            # set state <- next_state and update reward_total
            for agent_id in office_env.all_agents:
                state[agent_id] = tuple(next_state[agent_id])

                reward_total[agent_id] += rewards[agent_id]
            

            num_steps += 1

            # tracking the training details for logging
            agent1, agent2 = office_env.all_agents[0], office_env.all_agents[1]
            if 'n1' in true_propositions and rm_state[agent1] == 4:
                broken_decorations_score[agent1] +=1
            if 'n2' in true_propositions and rm_state[agent2] == 4:
                broken_decorations_score[agent2] +=1
            if 'g1' in true_propositions and rm_state[agent1] == 4:
                wins_total[agent1] += 1
            if 'g2' in true_propositions and rm_state[agent2] == 4:
                wins_total[agent2] += 1

            if print_freq and num_steps % print_freq == 0:
                logging.info(f"True props -> {true_propositions}")
                # print(f"Steps: {num_steps}, Episodes: {num_episodes}, Reward: {rewards}, Total reward: {reward_total}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}")
                logging.info(f"Steps: {num_steps}, Episodes: {num_episodes}, Reward: {rewards}, Total reward: {reward_total}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}")

            if print_freq and num_steps%print_freq == 0:
                reward_total = {agent_id: 0 for agent_id in office_env.all_agents}
                wins_total = {agent_id: 0 for agent_id in office_env.all_agents}
                broken_decorations_score = {agent_id: 0 for agent_id in office_env.all_agents}
                num_episodes = 0

                # save policies
                logging.info('policy saving start')

                for agent_id in office_env.all_agents:
                    other_agent_id = office_env.all_agents[0] if office_env.all_agents[1] == agent_id else office_env.all_agents[1]
                    checkpoint = num_steps/100000
                    policy_type = learning_agents[agent_id].name()[0] + learning_agents[other_agent_id].name()[0]
                    
                    with open(f'{policy_directory}/{map_type}-{map_number}/{checkpoint}-{policy_type}-{algorithm}-policy.pkl', 'wb') as file:
                        pickle.dump(learning_agents[agent_id].q_table, file)

                logging.info('policy saving end')



if __name__ == '__main__':
    timestamp = datetime.now().strftime("%m%d_%H%M")

    use_crm = True 
    map_type = 'simplified' # or 'base'

    q_init=2 # do not change
    learning_rate = 0.5
    discount_factor = 0.9

    if use_crm:
        algorithm = 'qrm'
    else:
        algorithm = 'qlearning'
    
    rm_list = rm_constants.MAP_2_RM_ORIGINAL_REWARDS
    map_number = rm_list[0]
    map_object = MapCollection.MAP_4_OBJECTS
    reward_machine_files = rm_list[1:]

    filename = f'logs/map_{map_number}_mixed_agents_{algorithm}_test_v2.log'
    # filename = f'logs/map_{map_number}_HIGHER_PENALTY_DECORATION_minmax_{algorithm}.log'

    main(filename, q_init, learning_rate, discount_factor,use_crm, map_object, map_type, reward_machine_files, map_number, total_timesteps=10e7, max_episode_length=1000, print_freq = 100000)
