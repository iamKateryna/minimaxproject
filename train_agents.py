import logging
import os
import wandb

from rl_agents.random.random_agent import RandomAgent
from rl_agents.minmax_qlearning.minmax_agent import MinMaxQLearningAgent
from rl_agents.qlearning.qlearning_agent import QLearningAgent

from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

from training_configurations import TrainingConfig

from utils import setup_logger


def setup_environment(config):
    office_env = OfficeWorldEnv(map_object=config.map_object, 
                                map_type=config.map_type.value, 
                                coffee_type=config.coffee_type.value, 
                                predator_prey=config.predator_prey, 
                                allow_stealing=config.allow_stealing, 
                                agents_can_be_in_same_cell=config.can_be_in_same_cell)
    rm_env = RewardMachineEnv(office_env, config.reward_machine_files)
    env = RewardMachineWrapper(rm_env, add_crms=config.use_crms)
    return office_env, env


def initialize_agents(agent_types, agent_ids, action_space, config):
    agents = {}
    for agent_type, agent_id in zip(agent_types, agent_ids):
        if agent_type.value == "minmax":
            agents[agent_id] = MinMaxQLearningAgent(action_space, q_init=config.q_init, learning_rate=config.learning_rate, discount_factor=config.discount_factor, exploration_rate=config.exploration_rate)
        elif agent_type.value == "qlearning":
            agents[agent_id] = QLearningAgent(action_space, q_init=config.q_init, learning_rate=config.learning_rate, discount_factor=config.discount_factor, exploration_rate=config.exploration_rate)
        elif agent_type.value == "random":
            agents[agent_id] = RandomAgent(action_space)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    return agents


def initialize_experiment(config, filename, my_group):
    wandb.init(
        name=filename[10:-4],
        project="minimaxQRM",
        group= my_group,

        config={
            "agent_0": f"{config.agent_types[0]}_{config.use_crms[0]}",
            "agent_1": f"{config.agent_types[1]}_{config.use_crms[1]}",
            "map_type": config.map_type,
            "exploration_decay_after": config.exploration_decay_after,
            "coffee_type": config.coffee_type,
            "agents_can_be_in_same_cell": config.can_be_in_same_cell,
            "total_timesteps": config.total_timesteps, 
            "max_episode_length": config.max_episode_length,
            "predator_prey": config.predator_prey,
            "kind": config.kind,
            "map_number": config.map_number,
        }
    )

    policy_path = f"new_policies/{config.map_type}-{config.map_number}"
    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    setup_logger(filename)

    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    logging.info(f"job ID -> {job_id}")
    logging.info(f"Policies directory -> {policy_path}")
    logging.info(f"Agents -> {config.agent_types}")
    logging.info(f"USE CRM: {config.use_crms}, MAP: {config.map_object}, TOTAL TIMESTEPS/EPISODE: {config.total_timesteps}")
    logging.info(f"lr -> {config.learning_rate}, q_init -> {config.q_init}, discount_factor -> {config.discount_factor}")
    
    return policy_path

def main(filename, my_group, details, config):

    policy_path = initialize_experiment(config, filename, my_group)

    office_env, env = setup_environment(config)
    action_space = office_env.primary_agent.action_space
    agent_ids = office_env.all_agents
    learning_agents = initialize_agents(config.agent_types, agent_ids, action_space, config)

    reward_total = {agent_id: 0 for agent_id in agent_ids}
    wins_total = {agent_id: 0 for agent_id in agent_ids}
    broken_decorations_score = {agent_id: 0 for agent_id in agent_ids}
    picked_coffees = {agent_id: 0 for agent_id in agent_ids}

    reward_per_print = {agent_id: 0 for agent_id in agent_ids}
    wins_per_print = {agent_id: 0 for agent_id in agent_ids}
    decorations_per_print = {agent_id: 0 for agent_id in agent_ids}
    coffees_per_print = {agent_id: 0 for agent_id in agent_ids}

    stolen_coffees = 0

    num_steps = 0
    num_episodes = 0

    while num_steps < config.total_timesteps:
        num_steps_per_episode = 0

        reset_env_state = env.reset()
        state = {agent_id: tuple(reset_env_state[agent_id]) for agent_id in agent_ids}
        done = False
        num_episodes+=1

        if config.exploration_decay_after.value == 'episode':
            for agent_id in agent_ids:
                learning_agents[agent_id].decay_epsilon(config.n_episodes_for_decay)

        while not done and num_steps_per_episode < config.max_episode_length:
            num_steps_per_episode += 1
            actions_to_execute = {}

            # Selecting and executing the action
            for agent_id in agent_ids:
                agent, agent_state = learning_agents[agent_id], state[agent_id]
                if isinstance(agent, MinMaxQLearningAgent) or isinstance(agent, QLearningAgent):
                    if agent_state not in agent.q_table:
                        agent.init_q_values(agent_state)

                    if config.exploration_decay_after.value == 'step':
                        agent.decay_epsilon(config.total_timesteps*0.7)

                    epsilon = agent.epsilon
                    learning_rate = agent.lr
                
                action = agent.get_action(agent_state)
                actions_to_execute[agent_id] = action

            next_state, rewards, done, info, true_propositions, rm_state = env.step(actions_to_execute, agent_type = "minmax", episode = num_episodes)
            done = any(done.values())

            # Updating the q-values
            for id, agent_id in enumerate(agent_ids):

                other_agent_id = agent_ids[0] if agent_ids[1] == agent_id else agent_ids[1]

                if isinstance(learning_agents[agent_id], MinMaxQLearningAgent):
                    if config.use_crms[id]:
                        experiences = []
                        for _state, (_action, _other_agent_action), _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                            experiences.append((tuple(_state), (_action, _other_agent_action), _reward, tuple(_next_state), _done))
                    else:
                        experiences = [(state[agent_id], (actions_to_execute[agent_id], actions_to_execute[other_agent_id]), rewards[agent_id], tuple(next_state[agent_id]), done)]
                        
                    learning_agents[agent_id].learn(experiences)

                elif isinstance(learning_agents[agent_id], QLearningAgent): 
                    if config.use_crms[id]:
                        experiences = []
                        for _state, (_action, _), _reward, _next_state, _done in info[agent_id]["crm-experience"]:
                            experiences.append((tuple(_state), (_action, ), _reward, tuple(_next_state), _done))
                    else:
                        experiences = [(state[agent_id], (actions_to_execute[agent_id],), rewards[agent_id], tuple(next_state[agent_id]), done)]
                
                    learning_agents[agent_id].learn(experiences)

                elif isinstance(learning_agents[agent_id], RandomAgent):
                    pass
                else: 
                    raise NotImplementedError
            

            logging.info(f"Step: {num_steps}, state -> {list(state[agent_ids[0]])}, actions -> {actions_to_execute[agent_ids[0]]}, {actions_to_execute[agent_ids[1]]}, props -> {true_propositions}, rm_state: ({rm_state[agent_ids[0]]}, {rm_state[agent_ids[1]]})rew: ({rewards[agent_ids[0]]}, {rewards[agent_ids[1]]}), next state -> {next_state[agent_ids[0]]}")

            num_steps += 1

            # set state <- next_state ansssd update reward_total
            for agent_id in agent_ids:
                state[agent_id] = tuple(next_state[agent_id])
                reward_total[agent_id] += rewards[agent_id]
                reward_per_print[agent_id] += rewards[agent_id]

            # tracking the training details for logging
            agent1, agent2 = agent_ids[0], agent_ids[1]
            if "n1" in true_propositions and rm_state[agent1] == 4:
                broken_decorations_score[agent1] +=1
                decorations_per_print[agent1] +=1
            elif (("g1" in true_propositions) and rm_state[agent1] == 4) and ("n2" not in true_propositions):
                wins_total[agent1] += 1
                wins_per_print[agent1] +=1
                logging.info('a win is a win -> g1')
            elif "f1" in true_propositions or "h1" in true_propositions:
                picked_coffees[agent1] += 1
                coffees_per_print[agent1] += 1
                logging.info('("f1" or "h1") in true_propositions')

            if "n2" in true_propositions and rm_state[agent2] == 4:
                broken_decorations_score[agent2] +=1
            elif "f2" in true_propositions or "h2" in true_propositions:
                picked_coffees[agent2] += 1
                coffees_per_print[agent2] += 1
                logging.info('("f2" or "h2") in true_propositions')
            if config.predator_prey:
                if "t" in true_propositions:
                    wins_total[agent2] += 1
                    wins_per_print[agent2] += 1
                    logging.info('a win is a win -> t')
            else:
                if (("g2" in true_propositions) and rm_state[agent2] == 4) and ("n1" not in true_propositions):
                    wins_total[agent2] += 1
                    wins_per_print[agent2] += 1
                    logging.info('a win is a win -> g2')

            if config.allow_stealing:
                if "t" in true_propositions:
                    stolen_coffees += 0 
                    logging.info("he's stealing cheese!!!")


            if config.print_freq and num_steps % config.print_freq == 0:

                logging.info(f"Steps: {num_steps}, Episodes: {num_episodes}, Total reward: {reward_total}, Picked coffees: {picked_coffees}, Total wins: {wins_total}, Broken decorations: {broken_decorations_score}")

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

                    "stolen_coffees": stolen_coffees,

                    # per print tracking

                    "rew_pp_a1": reward_per_print[agent_ids[0]],
                    "win_pp_a1": wins_per_print[agent_ids[0]],
                    "coffee_pp_a1": coffees_per_print[agent_ids[0]],
                    "decoration_pp_a1": decorations_per_print[agent_ids[0]],

                    "rew_pp_a2": reward_per_print[agent_ids[1]],
                    "win_pp_a2": wins_per_print[agent_ids[1]],
                    "coffee_pp_a2": coffees_per_print[agent_ids[1]],
                    "decoration_pp_a2": decorations_per_print[agent_ids[1]],

                    "exploration_rate": epsilon,
                    "details": details,
                    "learning_rate": learning_rate,
                    "discount_factor": config.discount_factor,

                })

                reward_per_print = {agent_id: 0 for agent_id in agent_ids}
                wins_per_print = {agent_id: 0 for agent_id in agent_ids}
                decorations_per_print = {agent_id: 0 for agent_id in agent_ids}
                coffees_per_print = {agent_id: 0 for agent_id in agent_ids}
                # num_episodes = 0
    
    # save policies
    if details == '-record-policy-':
        for agent_id in agent_ids:
            if isinstance(learning_agents[agent_id], MinMaxQLearningAgent) or isinstance(learning_agents[agent_id], QLearningAgent):
                filename = f"{policy_path}/{filename[10:-4]}-{agent_id}-policy.pkl"
                learning_agents[agent_id].save_policy(filename)
     
    # logging.info(f"q_table counts A1-> {learning_agents[agent_ids[0]].update_counts}")
    # with open(f"{filename[:-4]}-a1.pkl", "wb") as file:
    #                         pickle.dump(learning_agents[agent_id].update_counts, file)
    # logging.info(f"q_table counts A2 -> {learning_agents[agent_ids[1]].update_counts}")
    # with open(f"{filename[:-4]}-a2.pkl", "wb") as file:
    #                         pickle.dump(learning_agents[agent_id].update_counts, file)

if __name__ == '__main__':

    config = TrainingConfig()
    # details of the training to capture in .log file name
    my_group = f"new_loc-150k-mmqrm-vs-mmqrm-map9-longer"
    # my_group = ""
    details = "-record-policy-"
    # details = "-1-"
    # name of the .log file
    name = f"{config.kind}-{my_group}{details}map{config.map_number}-{config.agent_types[0]}-{config.algorithms[0]}-vs-{config.agent_types[1]}-{config.algorithms[1]}-{config.exploration_decay_after}-{config.coffee_type}-same_cell-{config.can_be_in_same_cell}"
    filename = f"logs/0603/{name}.log"

    main(filename, my_group, details, config)
