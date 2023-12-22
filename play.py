import logging
from tqdm import tqdm

from rl_agents.qlearning.qlearning_agent import QLearningAgent
from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

import rm_constants

def grid_environment():
    num_episodes = 100000
    total_timesteps = 2500
    print_freq = 0
    q_init=2.0
    
    return num_episodes, total_timesteps, print_freq, q_init

def setup_logger(filename):
    # Configure logger
    logging.basicConfig(filename=filename,  filemode='w', level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Started Logging")


def main(filename, use_crm, map_number, reward_machine_files):
    setup_logger(filename)

    office_env = OfficeWorldEnv(map_number)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crm=use_crm, gamma = 0.5)

    action_space = office_env.primary_agent.action_space
    
    learning_agents = {agent_id: QLearningAgent(action_space) 
              for agent_id in office_env.possible_agents}

    num_episodes, total_timesteps, print_freq, q_init = grid_environment()

    logging.info(f'USE CRM: {use_crm}, MAP: {map_number}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}')

    for episode in tqdm(range(num_episodes)):
        logging.info(f'Start Episode {episode +1}')
        # print('Start episode')
        
        state = {agent_id: tuple(env.reset()[agent_id]) for agent_id in office_env.all_agents}
        # print(f'state: {state}')

        num_steps = 0 

        done = False
        while not done and num_steps < total_timesteps:

            actions_to_execute = {}

            for agent_id in office_env.agents:
                if state[agent_id] not in learning_agents[agent_id].q_table:
                    learning_agents[agent_id].q_table[state[agent_id]] = {action: q_init for action in range(action_space.n)}

                action = learning_agents[agent_id].get_action(state[agent_id])
                actions_to_execute[agent_id] = action
            num_steps += 1
            # print(f'Choose actions: {actions_to_execute} ')
            # print(f'MAKING A STEP')

            next_state, rewards, done, info = env.step(actions_to_execute)
            
            done = any(done.values())

            # Updating the q-values
            for agent_id in office_env.all_agents:
                if use_crm:
                    experiences = []
                    for _state, _action, _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                        experiences.append((tuple(_state), _action, _reward, tuple(_next_state), _done))
                        # experiences.append((_state, _action, _reward, _next_state, _done))
                else:
                    experiences = [(state[agent_id], actions_to_execute[agent_id], rewards[agent_id], next_state[agent_id], done)]
                
                learning_agents[agent_id].learn(experiences)

            for agent_id in office_env.all_agents:    
                state[agent_id] = tuple(next_state[agent_id])

            if print_freq:
                if num_steps % print_freq == 0:
                    logging.info(f"Episode: {episode}, Num steps: {num_steps}, Reward: {rewards}")

        logging.info(f"Episode {episode+1}/{num_episodes} complete with Total Reward: {rewards}, and steps: {num_steps}")


if __name__ == '__main__':

    use_crm = True
    map_number = 3
    reward_machine_files = rm_constants.MAP_3_RMS

    if use_crm:
        main(f'logs/training_log_qrm_new_locations_{map_number}.log', use_crm, map_number, reward_machine_files)
    else:
        main(f'logs/training_log_qlearning_new_locations_{map_number}.log', use_crm, map_number, reward_machine_files)