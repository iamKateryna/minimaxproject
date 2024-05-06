import math

class RewardFunction:
    def __init__(self):
        pass


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def get_type(self):
        return "constant"

    def get_reward(self):
        return self.constant
    

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        return s_info['reward_ctrl']