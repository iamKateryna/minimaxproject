from envs.office_world.office_world_enums import MapType, CoffeeType
from envs.office_world.map_collection import MapCollection

import reward_machines.rm_constants as rm_constants

from rl_agents.enums import AgentType, ExplorationDecay

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
    total_timesteps = 10_000_000
    max_episode_length = 1_000
    print_freq = 1_000

    q_init = 0
    learning_rate = 0.2
    discount_factor = 0.95
    exploration_rate = 0.5

    exploration_decay_after = ExplorationDecay.EPISODE
    n_episodes_for_decay = 150_000

    # training algorithms
    use_crms = (False, False)
    # use_crms = (True, True)
    agent_types = (AgentType.MINMAX, AgentType.MINMAX)
    # agent_types = (AgentType.QLEARNING, AgentType.QLEARNING)

    #log file details
    my_group = f"big-rew-real-150ep-map11-m" # wandb group
    details = "-record-policy-" # if detasils = "-record-policy-", policy (q-table) will be saved
    # details = "-1-"

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("qrm" if use_crm else "qlearning" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"

    # name of the .log file
    name = f"{kind}-{my_group}{details}map{map_number}-{agent_types[0]}-{algorithms[0]}-vs-{agent_types[1]}-{algorithms[1]}"
    # name = f"{config.kind}-{my_group}{details}map{config.map_number}-{config.agent_types[0]}-{config.algorithms[0]}-vs-{config.agent_types[1]}-{config.algorithms[1]}-{config.exploration_decay_after}-{config.coffee_type}-same_cell-{config.can_be_in_same_cell}"
    filename = f"logs/2503/{name}.log"