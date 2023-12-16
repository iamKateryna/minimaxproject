from envs.grids.office_world_env import OfficeWorldEnv
from reward_machines.rm_environment import RewardMachineEnv

import rm_constants

SEED = 42

def play_officeWorldRMEnv():
    office_env = OfficeWorldEnv(map_number=2)
    reward_machine_files = rm_constants.MAP_2_RMS
    reward_machine_env = RewardMachineEnv(office_env, reward_machine_files)
    reward_machine_env.render()


# def train(env, total_timesteps):

#     model = learn(
#         env=env,
#         seed=SEED,
#         total_timesteps=total_timesteps,
#         **alg_kwargs
#     )
    



# def main():
#     total_timesteps = 10

#     office_env = OfficeWorldEnv()
#     reward_machine_files = ['envs/grids/reward_machines_m2_a1.txt','envs/grids/reward_machines_m2_a2.txt']
#     reward_machine_env = RewardMachineEnv(office_env, reward_machine_files)

#     train(reward_machine_env, total_timesteps)
#     pass


if __name__ == '__main__':

    play_officeWorldRMEnv()
