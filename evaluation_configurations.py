import pickle

from envs.office_world.office_world_enums import MapType, CoffeeType
from envs.office_world.map_collection import MapCollection

import reward_machines.rm_constants as rm_constants

from rl_agents.enums import AgentType

class EvaluateConfig:
    # RM files and map configurations
    predator_prey = False
    allow_stealing = False 
    map_object = MapCollection.MAP_11_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_2_DELIVERY_TASK
    # training info
    total_timesteps = 100_000
    max_episode_length = 1000
    print_freq = 1_000

    q_init = 0

    # training algorithms details
    use_crms = (False, True)
    # use_crms = (True, True)
    # use_crms = (True, False)
    # use_crms = (False, True)
    # agent_types = (AgentType.MINMAX, AgentType.QLEARNING)
    agent_types = (AgentType.MINMAX, AgentType.MINMAX)
    # agent_types = (AgentType.QLEARNING, AgentType.QLEARNING)
    # agent_types = (AgentType.QLEARNING, AgentType.MINMAX)

    path_a1 = "meeting-0105/MapType.SIMPLIFIED-2/same_goal-meeting-01052-5mil-boltzmann-1-map2-AgentType.MINMAX-qrm-vs-AgentType.MINMAX-qrm-primary_agent-policy.pkl"
    path_a2 = "meeting-0105/MapType.SIMPLIFIED-2/same_goal-meeting-01052-5mil-boltzmann-1-map2-AgentType.MINMAX-qrm-vs-AgentType.MINMAX-qrm-second_agent-policy.pkl"

    # log file details
    my_group = f"eval-meeting-0105"
    details = "-boltzmann-"

    # do not change
    q_table_paths = {
        "primary_agent": path_a1,
        "second_agent": path_a2
    }
    
    exploration_rate = 0
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("crm" if use_crm else "q" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"
    q_tables = {agent: pickle.load(open(path, "rb")) for agent, path in q_table_paths.items()}

    # name of the .log file
    name = f"{kind}-{my_group}{details}map{map_number}-{agent_types[0]}-{algorithms[0]}-vs-{agent_types[1]}-{algorithms[1]}"
    filename = f"logs/0605/{name}.log"