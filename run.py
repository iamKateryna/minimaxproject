import rm_constants
from envs.officeWorld.map_collection import MapCollection

from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv

# play OfficeWorldRMsEnv
def main():

    # options
    map_type = 'simplified' # or 'base'
    rm_list = rm_constants.MAP_2_PREDATOR_PREY # see rm_constants for options
    map_object = MapCollection.MAP_4_OBJECTS # see map_collection for options

    # do not change
    reward_machine_files = rm_list[1:]
    office_env = OfficeWorldEnv(map_object=map_object, map_type=map_type, coffee_type="single", predator_prey=True, agents_can_be_in_same_cell=True)
    rm_env = RewardMachineEnv(office_env, reward_machine_files)
    rm_env.render()


if __name__ == '__main__':
    main()