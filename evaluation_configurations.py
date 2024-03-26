import pickle

from envs.office_world.office_world_enums import MapType, CoffeeType
from envs.office_world.map_collection import MapCollection

import reward_machines.rm_constants as rm_constants

from rl_agents.enums import AgentType

class EvaluateConfig:
    # RM files and map configurations
    predator_prey = True
    allow_stealing = False 
    map_object = MapCollection.MAP_10_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_3_PREDATOR_PREY

    # training info
    total_timesteps = 100_000
    max_episode_length = 1_000
    print_freq = 1_000

    q_init = 0
    exploration_rate = 0

    # training algorithms details
    use_crms = (True, True)
    # use_crms = (False, True)
    # agent_types = (AgentType.MINMAX, AgentType.QLEARNING)
    agent_types = (AgentType.QLEARNING, AgentType.MINMAX)

    path_a2 = "report_policies/MapType.SIMPLIFIED-3/predator_prey-map10-mqrm-mqrm-record-policy-map3-AgentType.MINMAX-qrm-vs-AgentType.MINMAX-qrm-second_agent-policy.pkl"
    path_a1 = "report_policies/MapType.SIMPLIFIED-3/predator_prey-map10-qrm-qrm-record-policy-map3-AgentType.QLEARNING-qrm-vs-AgentType.QLEARNING-qrm-primary_agent-policy.pkl"

    # Define the paths in a dictionary for clarity and easy management
    q_table_paths = {
        "primary_agent": path_a1,
        "second_agent": path_a2
    }

    # log file details
    my_group = f"eval-qrm-vs-mqrm-n"
    details = "-test-"

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("qrm" if use_crm else "qlearning" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"
    q_tables = {agent: pickle.load(open(path, "rb")) for agent, path in q_table_paths.items()}

    # name of the .log file
    name = f"{my_group}{details}map{map_number}-{kind}-{coffee_type}-{can_be_in_same_cell}"
    filename = f"logs/2503/{name}.log"