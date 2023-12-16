import logging
from tqdm import tqdm

from rl_agents.qlearning.qlearning_agent import QLearningAgent
from envs.grids.office_world_env import OfficeWorldEnv
from reward_machines.rm_environment import RewardMachineEnv

def setup_logger():
    # Configure logger
    logging.basicConfig(filename='training_log.log',  filemode='w', level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Started Logging")

def main():
    setup_logger()
    office_env = OfficeWorldEnv()
    reward_machine_files = ['envs/grids/reward_machines_m2_a1.txt',
                            'envs/grids/reward_machines_m2_a2.txt']
    env = RewardMachineEnv(office_env, reward_machine_files)

    action_space = office_env.primary_agent.action_space
    
    learning_agents = {agent_id: QLearningAgent(action_space) 
              for agent_id in office_env.possible_agents}

    num_episodes = 100000
    total_timesteps = 1000 # per episode
    print_freq = 100
    q_init = 0.02

    for episode in tqdm(range(num_episodes)):
        logging.info(f'Start Episode {episode +1}')
        state = tuple(env.reset())
        num_steps = 0 

        done = False
        while not done and num_steps < total_timesteps:

            actions_to_execute = {}

            for agent_id in office_env.agents:

                if state not in learning_agents[agent_id].q_table:
                    learning_agents[agent_id].q_table[state] = {action: q_init for action in range(action_space.n)}

                action = learning_agents[agent_id].get_action(state)
                actions_to_execute[agent_id] = action
            num_steps += 1

            next_state, rewards, done, _ = env.step(actions_to_execute)
            
            done = any(done.values())

            for agent_id in office_env.agents:
                learning_agents[agent_id].learn(state, actions_to_execute[agent_id], rewards[agent_id], next_state, done) 

            state = tuple(next_state)
        
            # if num_steps % print_freq == 0:
            #     logging.info(f"Episode: {episode}, Num steps: {num_steps}, Reward: {total_reward}")

        logging.info(f"Episode {episode+1}/{num_episodes} complete with Total Reward: {rewards}, and steps: {num_steps}")



if __name__ == '__main__':
    main()