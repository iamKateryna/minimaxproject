from envs.office_world.office_world_enums import MapType, CoffeeType
from envs.office_world.map_collection import MapCollection

import reward_machines.rm_constants as rm_constants

from rl_agents.enums import AgentType, ExplorationDecay, ExplorationPolicy

class TrainingConfig:
    # RM files and map configurations
    predator_prey = False
    allow_stealing = False 
    map_object = MapCollection.MAP_11_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_2_DELIVERY_TASK
    # training info
    total_timesteps = 400_000_000
    max_episode_length = 1_000
    print_freq = 1_000

    q_init = 0
    learning_rate = 0.2
    discount_factor = 0.95

    policy = ExplorationPolicy.BOLTZMANN
    exploration_rate = 0.8 # if boltzmann exploration policy, treat exploration_rate as temperature
    min_exploration_rate = 0.1
    exploration_decay_after = ExplorationDecay.EPISODE
    n_episodes_for_decay = 1_000

    # training algorithms
    # use_crms = (False, False)
    use_crms = (True, True)
    agent_types = (AgentType.MINMAX, AgentType.MINMAX)
    # agent_types = (AgentType.QLEARNING, AgentType.QLEARNING)

    #log file details
    my_group = f"meeting-0605" # wandb group
    save_policy = True
    policies_path = "meeting-0605"
    # details = "-400k-ep-day-random-init-" 
    details = '-rm-wrapper-debug-'
    n_run = ""

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("crm" if use_crm else "q" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"

    # name of the .log file
    name = f"{kind}-{my_group}{details}{n_run}map{map_number}-{agent_types[0]}-{algorithms[0]}-vs-{agent_types[1]}-{algorithms[1]}"
    filename = f"logs/0605/{name}.log"