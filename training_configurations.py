from envs.officeWorld.enums import MapType, CoffeeType, ExplorationDecay, AgentType
from envs.officeWorld.map_collection import MapCollection
import reward_machines.rm_constants as rm_constants

class TrainingConfig:
    # RM files and map configurations
    predator_prey = False
    allow_stealing = False 
    map_object = MapCollection.MAP_7_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_2_RM_ORIGINAL_REWARDS_2_DIFFERENT_COFFEES

    # training info
    total_timesteps = 2_000_000
    max_episode_length = 1_000
    print_freq = 1_000

    q_init = 0
    learning_rate = 0.2
    discount_factor = 0.95
    exploration_rate = 0.501

    exploration_decay_after = ExplorationDecay.EPISODE
    n_episodes_for_decay = 150_000

    # training algorithms
    use_crms = (True, True)
    agent_types = (AgentType.MINMAX, AgentType.MINMAX)

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]
    algorithms = tuple("qrm" if use_crm else "qlearning" for use_crm in use_crms)
    kind = "predator_prey" if predator_prey else "same_goal_allow_stealing" if allow_stealing else "same_goal"