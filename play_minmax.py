import logging
from tqdm import tqdm

from rl_agents.minmax_qlearning.minmax_agent import MinMaxQLearningAgent
from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

import rm_constants
from defaults import grid_environment

def setup_logger(filename):
    # Configure logger
    logging.basicConfig(filename=filename,  filemode='w', level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Started Logging")


def main(filename, use_crm, map_number, reward_machine_files):
    use_crm = False
    setup_logger(filename)

    office_env = OfficeWorldEnv(map_number)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crm=use_crm, gamma = 0.5)

    action_space = office_env.primary_agent.action_space

    learning_agents = {agent_id: MinMaxQLearningAgent(action_space) 
              for agent_id in office_env.possible_agents}
    
    num_episodes, total_timesteps, print_freq, q_init = grid_environment()

    logging.info(f'USE CRM: {use_crm}, MAP: {map_number}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}')

    for episode in tqdm(range(num_episodes)):
        logging.info(f'Start Episode {episode +1}')

        state = {agent_id: tuple(env.reset()[agent_id]) for agent_id in office_env.all_agents}

        num_steps = 0

        done = False
        # print('Start epoch')
        while not done and num_steps < total_timesteps:

            actions_to_execute = {}
            # print(f'Actions to exec: {actions_to_execute}')

            for agent_id in office_env.agents:

                if state[agent_id] not in learning_agents[agent_id].q_table:
                    learning_agents[agent_id].q_table[state[agent_id]] = {(action_1, action_2): q_init 
                                                                          for action_1 in range(action_space.n)
                                                                          for action_2 in range(action_space.n)}
                action = learning_agents[agent_id].get_action(state[agent_id])
                actions_to_execute[agent_id] = action
            
            # print(f'Actions to exec: {actions_to_execute}')
            num_steps +=1

            # print(f'STATE: {state}')

            next_state, rewards, done, info = env.step(actions_to_execute)

            # print(f'NEXT STATE: {next_state}')

            done = any(done.values())

            # Updating the q-values
            for agent_id in office_env.all_agents:
                other_agent_id = office_env.all_agents[0] if office_env.all_agents[1] == agent_id else office_env.all_agents[1]
                # print(agent_id, other_agent_id)
                if use_crm:
                    # to be implemented
                    raise NotImplementedError(" CRM with MinMax is not implemented yet")
                else:
                    experiences = [(state[agent_id], actions_to_execute[agent_id], actions_to_execute[other_agent_id], rewards[agent_id], next_state[agent_id], done)]
                learning_agents[agent_id].learn(experiences)

            for agent_id in office_env.all_agents:    
                state[agent_id] = tuple(next_state[agent_id])

            if print_freq:
                if num_steps % print_freq == 0:
                    logging.info(f"Episode: {episode}, Num steps: {num_steps}, Reward: {rewards}")

        logging.info(f"Episode {episode+1}/{num_episodes} complete with Total Reward: {rewards}, and steps: {num_steps}")

if __name__ == '__main__':

    use_crm = False
    map_number=2
    reward_machine_files = rm_constants.MAP_2_RMS
    
    main(f'logs/training_log_minmax_{map_number}.log', use_crm, map_number, reward_machine_files)