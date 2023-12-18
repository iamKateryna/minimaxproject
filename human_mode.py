from envs.officeWorld.office_world_env import OfficeWorldEnv
from reward_machines.reward_machine_environment import RewardMachineEnv

import rm_constants

def play_officeWorldRMEnv():
    office_env = OfficeWorldEnv(map_number=2)
    reward_machine_files = rm_constants.MAP_2_RMS
    reward_machine_env = RewardMachineEnv(office_env, reward_machine_files)
    reward_machine_env.render()


if __name__ == '__main__':

    play_officeWorldRMEnv()
