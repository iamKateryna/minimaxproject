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