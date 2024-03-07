import pickle

from envs.officeWorld.enums import MapType, CoffeeType, AgentType
from envs.officeWorld.map_collection import MapCollection
import reward_machines.rm_constants as rm_constants

class EvaluateConfig:
    # RM files and map configurations
    predator_prey = False
    allow_stealing = False 
    map_object = MapCollection.MAP_7_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_2_RM_ORIGINAL_REWARDS_2_DIFFERENT_COFFEES

    # training info
    total_timesteps = 100_000
    max_episode_length = 1_000
    print_freq = 1_000

    q_init = 0
    exploration_rate = 0

    # training algorithms details
    use_crms = (True, True)
    agent_types = (AgentType.QLEARNING, AgentType.QLEARNING)

    # Define the paths in a dictionary for clarity and easy management
    q_table_paths = {
        "second_agent": "policies/simplified-2/same_goal-150k-ep-decay-track-qrm-vs-qrm-record-policy-map2-qlearning-qrm-vs-qlearning-qrm-explor_decay_after-step-coffee-single-same_cell-False-second_agent-policy.pkl",
        "primary_agent": "policies/simplified-2/same_goal-150k-ep-decay-track-qrm-vs-qrm-record-policy-map2-qlearning-qrm-vs-qlearning-qrm-explor_decay_after-episode-coffee-single-same_cell-False-primary_agent-policy.pkl"
    }

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("qrm" if use_crm else "qlearning" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"
    q_tables = {agent: pickle.load(open(path, "rb")) for agent, path in q_table_paths.items()}

