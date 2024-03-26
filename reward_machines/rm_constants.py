RM_PATH_PREFIX_OFFICE_WORLD = "reward_machines/office_world/"
RM_PATH_PREFIX_MPE_SIMPLE_ADVERSARY = "reward_machines/mpe/simple_adversary/"
RM_PATH_PREFIX_MPE_SIMPLE_TAG = "reward_machines/mpe/simple_tag/"
A1_SUFFIX = '_a1'
A2_SUFFIX = '_a2'

# RM files paths
#mpe simple tag
ST = ["", f"{RM_PATH_PREFIX_MPE_SIMPLE_TAG}st{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_MPE_SIMPLE_TAG}st{A2_SUFFIX}.txt"]

# predator-prey
MAP_2_PREDATOR_PREY = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_predator_prey{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_predator_prey{A2_SUFFIX}.txt"]
MAP_3_PREDATOR_PREY = [3, f"{RM_PATH_PREFIX_OFFICE_WORLD}m3_predator_prey{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m3_predator_prey{A2_SUFFIX}.txt"]

# same task
MAP_3_DELIVERY_TASK = [3, f"{RM_PATH_PREFIX_OFFICE_WORLD}m3_2_different_coffees{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m3_2_different_coffees{A2_SUFFIX}.txt"]
MAP_2_DELIVERY_TASK = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_original_rewards_2_different_coffees{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_original_rewards_2_different_coffees{A2_SUFFIX}.txt"]

# MAP_2_RM_COMPLETED_TASK_HIGHER_REWARD_2_DIFFERENT_COFFEES = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_completed_task_higher_reward_2_different_coffees{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_completed_task_higher_reward_2_different_coffees{A2_SUFFIX}.txt"] # add second coffee sign
# MAP_2_RM_BROKEN_DEC_HIGHER_PENALTY_2_DIFFERENT_COFFEES = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_broken_dec_higher_penalty_2_different_coffees{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_broken_dec_higher_penalty_2_different_coffees{A2_SUFFIX}.txt"] # add second coffee sign

# allow stealing coffee
# MAP_2_STEALING = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_stealing_mode_2_different_coffees{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_stealing_mode_2_different_coffees{A2_SUFFIX}.txt"]

# when one agent dies, another continues the game
# MAP_2_SEPARATE_DEATHS = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_separate_deaths{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_separate_deaths{A2_SUFFIX}.txt"]

# RMs for maps with no distinction between coffee machines, were used before "single" coffee per machine option, 
# not used anymore, since "...different_coffees" RMs are suitable for both options

# MAP_3_RM = [3, f"{RM_PATH_PREFIX_OFFICE_WORLD}m3{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m3{A2_SUFFIX}.txt"]
# MAP_2_RM_HIGHER_PENALTY_DECORATION = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_higher_penalty_decoration{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_higher_penalty_decoration{A2_SUFFIX}.txt"]
# MAP_2_RM_HIGHER_PENALTY_LOSS = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_higher_penalty_loss{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_higher_penalty_loss{A2_SUFFIX}.txt"]
# MAP_2_RM_NO_PENALTY_DECORATION = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_no_penalty_decoration{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_no_penalty_decoration{A2_SUFFIX}.txt"]
# MAP_2_RM_ORIGINAL_REWARDS = [2, f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_original_rewards{A1_SUFFIX}.txt", f"{RM_PATH_PREFIX_OFFICE_WORLD}m2_original_rewards{A2_SUFFIX}.txt"]