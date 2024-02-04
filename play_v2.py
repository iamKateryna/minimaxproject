import logging
from tqdm import tqdm
from datetime import datetime

from rl_agents.qlearning.qlearning_agent import QLearningAgent
from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

from envs.officeWorld.map_collection import MapCollection
from utils import setup_logger

import rm_constants


def main(filename, q_init, learning_rate, discount_factor, use_crm, map_number, reward_machine_files, total_timesteps=100000, max_episode_length=1000, print_freq=0):
    setup_logger(filename)

    office_env = OfficeWorldEnv(map_number)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crm=use_crm, gamma = 0.5)

    action_space = office_env.primary_agent.action_space
    
    learning_agents = {agent_id: QLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor) 
              for agent_id in office_env.possible_agents}

    logging.info(f'USE CRM: {use_crm}, MAP: {map_number}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}')
    logging.info(f"lr -> {learning_rate}, q_init -> {q_init}, discount_factor -> {discount_factor}")

    reward_total = {agent_id: 0 for agent_id in office_env.all_agents}
    wins_total = {agent_id: 0 for agent_id in office_env.all_agents}
    broken_decorations_score = {agent_id: 0 for agent_id in office_env.all_agents}

    num_steps = 0
    num_episodes = 0

    while num_steps < total_timesteps:
        num_steps_per_episode = 0
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
                # print(f"AGENT -> {agent_id}")
                agent, agent_state = learning_agents[agent_id], state[agent_id]

                # print(f"agent_state -> {agent_state}")
                # print(f"agent_state not in agent.q_table -> {agent_state not in agent.q_table}")

                if agent_state not in agent.q_table:
                    # print(f"creating new agent_state -> {agent_state}")
                    agent.init_q_values(agent_state)

                action = agent.get_action(agent_state)
                actions_to_execute[agent_id] = action

            # print(f"actions_to_execute -> {actions_to_execute}")

            next_state, rewards, done, info, true_propositions, rm_state = env.step(actions_to_execute,  agent_type = 'qlearning', episode = num_episodes)
            done = any(done.values())

            # Updating the q-values
            # print(" --- UPDARING Q-VALUES --- ")
            for agent_id in office_env.all_agents:



                if use_crm:
                    experiences = []
                    for _state, _action, _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                        experiences.append((tuple(_state), _action, _reward, tuple(_next_state), _done))
                else:
                    experiences = [(state[agent_id], actions_to_execute[agent_id], rewards[agent_id], tuple(next_state[agent_id]), done)]
                
                learning_agents[agent_id].learn(experiences)

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
                print(f"Steps: {num_steps}, Episodes: {num_episodes}, Reward: {rewards}, Total reward: {reward_total}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}")
                logging.info(f"Steps: {num_steps}, Episodes: {num_episodes}, Reward: {rewards}, Total reward: {reward_total}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%m%d_%H%M")

    use_crm = False 

    q_init=2 # do not change
    learning_rate = 0.5
    discount_factor = 0.9

    if use_crm:
        algorithm = 'qrm'
    else:
        algorithm = 'qlearning'
    
    rm_list = rm_constants.MAP_2_RM_HIGHER_PENALTY_LOSS
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]

    filename = f'logs/{map_number}_higher_penalty_loss_{algorithm}.log'

    main(filename, q_init, learning_rate, discount_factor,use_crm, map_number, reward_machine_files, total_timesteps=1000000, max_episode_length=1000, print_freq = 100000)
