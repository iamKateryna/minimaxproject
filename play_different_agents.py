import logging
from tqdm import tqdm
from datetime import datetime
import os
import pickle
import wandb

from rl_agents.random.random_agent import RandomAgent
from rl_agents.minmax_qlearning.minmax_agent import MinMaxQLearningAgent
from rl_agents.qlearning.qlearning_agent import QLearningAgent

from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

from envs.officeWorld.map_collection import MapCollection
from utils import setup_logger

import rm_constants


def main(filename, agent_0_type, agent_1_type, q_init, learning_rate, discount_factor, use_crm, map_object, map_type, coffee_type, agents_can_be_in_same_cell, reward_machine_files, map_number, total_timesteps, max_episode_length, print_freq):
    wandb.init(
        name=filename,
        project="minimaxQRM",

        config={
            "agent_0_type": agent_0_type,
            "agent_1_type": agent_1_type,
            "use_crm": use_crm,
            "map_type": map_type,
            "coffee_type": coffee_type,
            "agents_can_be_in_same_cell": agents_can_be_in_same_cell,
            "total_timesteps": total_timesteps, 
            "max_episode_length": max_episode_length
        }
    )


    setup_logger(filename)

    policy_directory = "policies"

    if not os.path.exists(f"{policy_directory}/{map_type}-{map_number}"):
        os.makedirs(f"{policy_directory}/{map_type}-{map_number}")

    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    logging.info(f"job ID -> {job_id}")

    office_env = OfficeWorldEnv(map_object=map_object, map_type=map_type, coffee_type=coffee_type, agents_can_be_in_same_cell=agents_can_be_in_same_cell)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crm=use_crm, gamma = 0.5)

     
    action_space = office_env.primary_agent.action_space
    
    agent_types = [agent_0_type, agent_1_type]
    agent_ids = office_env.all_agents
    learning_agents = {}

    for agent_type, agent_id in zip(agent_types, agent_ids):
        if agent_type == "minmax":
            learning_agents[agent_id] = MinMaxQLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor)
        elif agent_type == "qlearning":
            learning_agents[agent_id] = QLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor)
        elif agent_type == "random":
            learning_agents[agent_id] = RandomAgent(action_space)
    
    logging.info(f"Agents -> {agent_types}")

    # learning_agents = {office_env.possible_agents[0]: QLearningAgent(action_space, q_init=q_init, learning_rate=learning_rate, discount_factor=discount_factor),
    #                    office_env.possible_agents[1]: RandomAgent(action_space)}

    logging.info(f"USE CRM: {use_crm}, MAP: {map_object}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}")
    logging.info(f"lr -> {learning_rate}, q_init -> {q_init}, discount_factor -> {discount_factor}")
    print(f"USE CRM: {use_crm}, MAP: {map_object}, TOTAL TIMESTEPS/EPISODE: {total_timesteps}")
    print(f"lr -> {learning_rate}, q_init -> {q_init}, discount_factor -> {discount_factor}")

    reward_total = {agent_id: 0 for agent_id in agent_ids}
    wins_total = {agent_id: 0 for agent_id in agent_ids}
    broken_decorations_score = {agent_id: 0 for agent_id in agent_ids}
    picked_coffees = {agent_id: 0 for agent_id in agent_ids}

    num_steps = 0
    num_episodes = 0

    while num_steps < total_timesteps:
        num_steps_per_episode = 0
        
        # reward_total = {agent_id: 0 for agent_id in agent_ids}
        # wins_total = {agent_id: 0 for agent_id in agent_ids}
        # broken_decorations_score = {agent_id: 0 for agent_id in agent_ids}
        # print(f"Start Episode {num_episodes +1}")

        reset_env_state = env.reset()
        state = {agent_id: tuple(reset_env_state[agent_id]) for agent_id in agent_ids}
        done = False
        num_episodes+=1

        while not done and num_steps_per_episode < max_episode_length:
            num_steps_per_episode += 1
            actions_to_execute = {}

            # Selecting and executing the action
            for agent_id in agent_ids:
                if isinstance(learning_agents[agent_id], MinMaxQLearningAgent) or isinstance(learning_agents[agent_id], QLearningAgent):
                    agent, agent_state = learning_agents[agent_id], state[agent_id]

                    if agent_state not in agent.q_table:
                        # print(f"creating new agent_state -> {agent_state}")
                        agent.init_q_values(agent_state)

                action = agent.get_action(agent_state)
                actions_to_execute[agent_id] = action


            next_state, rewards, done, info, true_propositions, rm_state = env.step(actions_to_execute, agent_type = "minmax", episode = num_episodes)
            done = any(done.values())

            # Updating the q-values
            for agent_id in agent_ids:

                other_agent_id = agent_ids[0] if agent_ids[1] == agent_id else agent_ids[1]

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
            for agent_id in agent_ids:
                state[agent_id] = tuple(next_state[agent_id])

                reward_total[agent_id] += rewards[agent_id]
            

            num_steps += 1
            # tracking the training details for logging
            agent1, agent2 = agent_ids[0], agent_ids[1]
            if "n1" in true_propositions and rm_state[agent1] == 4:
                broken_decorations_score[agent1] +=1
            if "n2" in true_propositions and rm_state[agent2] == 4:
                broken_decorations_score[agent2] +=1
            if "g1" in true_propositions and rm_state[agent1] == 4:
                wins_total[agent1] += 1
            if "g2" in true_propositions and rm_state[agent2] == 4:
                wins_total[agent2] += 1
            if ("f1" or "h1") in true_propositions:
                picked_coffees[agent1] += 1
            if ("f2" or "h2") in true_propositions:
                picked_coffees[agent2] += 1

            if print_freq and num_steps % print_freq == 0:
                logging.info(f"True props -> {true_propositions}")
                logging.info(f"Steps: {num_steps}, Episodes: {num_episodes}, Reward: {rewards}, Total reward: {reward_total}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}, Picked coffees: {picked_coffees}")

                wandb.log({
                    "Steps": num_steps, 
                    "Episodes": num_episodes,

                    "rewards_a1": reward_total[agent_ids[0]],
                    "wins_a1": wins_total[agent_ids[0]],
                    "picked_coffees_a1": picked_coffees[agent_ids[0]],
                    "broken_decorations_a1": broken_decorations_score[agent_ids[0]],

                    "rewards_a2": reward_total[agent_ids[1]],
                    "wins_a2": wins_total[agent_ids[1]],
                    "picked_coffees_a2": picked_coffees[agent_ids[1]],
                    "broken_decorations_a2": broken_decorations_score[agent_ids[1]],

                })

                reward_total = {agent_id: 0 for agent_id in agent_ids}
                wins_total = {agent_id: 0 for agent_id in agent_ids}
                broken_decorations_score = {agent_id: 0 for agent_id in agent_ids}
                picked_coffees = {agent_id: 0 for agent_id in agent_ids}
                num_episodes = 0
                
                if num_steps%(100*print_freq)==0:
                    # save policies
                    for agent_id in agent_ids:
                        checkpoint = num_steps/(100*print_freq)
                        policy_type = agent_types[0] + agent_types[1]
                        
                        with open(f"{policy_directory}/{map_type}-{map_number}/{checkpoint}-{policy_type}-{algorithm}-policy.pkl", "wb") as file:
                            pickle.dump(learning_agents[agent_id].q_table, file)



if __name__ == '__main__':
    timestamp = datetime.now().strftime("%m%d_%H%M")

    use_crm = True 
    map_type = "simplified" # or "base"
    can_be_in_same_cell = False
    coffee_type = "single" # or "single", pay attention at the rm_files

    # agent_types = ["minmax", "qlearning"] # "minmax" or "qlearning" or "random"
    # agent_types = ["minmax", "minmax"]
    # agent_types = ["minmax", "random"]
    # agent_types = ["qlearning", "random"]
    agent_types = ["qlearning", "qlearning"]

    total_timesteps=1000000
    max_episode_length=1000
    print_freq = 10000

    q_init = 2 # do not change
    learning_rate = 0.5
    discount_factor = 0.9
 
    if use_crm:
        algorithm = "qrm"
    else:
        algorithm = "qlearning"

    details = "-original-rewards"
    
    rm_list = rm_constants.MAP_2_RM_ORIGINAL_REWARDS_2_DIFFERENT_COFFEES 
    map_number = rm_list[0]
    map_object = MapCollection.MAP_4_OBJECTS
    reward_machine_files = rm_list[1:]

    filename = f"logs/0902/map{map_number}-{agent_types[0]}-vs-{agent_types[1]}-agents-{algorithm}-{coffee_type}-{can_be_in_same_cell}{details}.log"
    # filename = f"logs/0502_2/a_test_coffee_type.log"

    main(filename, agent_types[0], agent_types[1], q_init, learning_rate, discount_factor,use_crm, map_object, map_type, coffee_type, can_be_in_same_cell, reward_machine_files, map_number, total_timesteps, max_episode_length, print_freq)
