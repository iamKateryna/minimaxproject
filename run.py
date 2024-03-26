import reward_machines.rm_constants as rm_constants
from envs.office_world.map_collection import MapCollection
from envs.office_world.office_world_enums import MapType, CoffeeType
from rl_agents.enums import AgentType

from envs.office_world.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv

class HumanModeConfig:
    predator_prey = True
    allow_stealing = False 
    map_object = MapCollection.MAP_10_OBJECTS
    map_type = MapType.SIMPLIFIED
    can_be_in_same_cell = False # if predator_prey == True, can_be_in_same_cell always true, even if you choose here False
    coffee_type = CoffeeType.SINGLE 
    rm_list = rm_constants.MAP_3_PREDATOR_PREY

    # do not change
    map_number = rm_list[0]
    reward_machine_files = rm_list[1:]


# play OfficeWorldRMsEnv
def main(config):
    office_env = OfficeWorldEnv(map_object=config.map_object, 
                                map_type=config.map_type.value, 
                                coffee_type=config.coffee_type.value, 
                                predator_prey=config.predator_prey, 
                                allow_stealing=config.allow_stealing, 
                                agents_can_be_in_same_cell=config.can_be_in_same_cell)
    rm_env = RewardMachineEnv(office_env, config.reward_machine_files)
   
    rm_env.render()


if __name__ == '__main__':
    config = HumanModeConfig()
    main()