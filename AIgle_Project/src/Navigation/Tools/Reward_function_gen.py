
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import math

# Libs
from keras.models import Sequential


# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Reward_function(object):
    def __init__(self):
        self.goal_dict = {"0": {"x": 0,
                                "y": -28.5,
                                "z": -2,
                                "turn": 0}
                          }
        return

    def get_reward(self, state, goal, collision, age):
        if collision is True:
            # --> Negative reward if a collision occurred
            reward = -1000 * age

        else:
            # --> Set default reward to 0
            reward = 0

            # --> Check x position
            if -2.65 <= state[0][0] <= 3.27 and -28 <= state[0][1] <= -29 and -4.93 <= state[0][2] <= 0.7596:
                pass
            else:
                reward = -(math.sqrt((self.goal_dict[str(goal)]["x"] - state[0][0])**2
                                     + (self.goal_dict[str(goal)]["y"] - state[0][1])**2
                                     + (self.goal_dict[str(goal)]["z"] - state[0][2])**2)

                           / math.sqrt(self.goal_dict[str(goal)]["x"]**2
                                       + self.goal_dict[str(goal)]["y"]**2
                                       + self.goal_dict[str(goal)]["z"]**2)) * age

        return reward

    def check_if_done(self, state, goal, collision, age, max_age):
        if age >= max_age:
            # --> End run if max age reached
            return True

        if collision is True:
            # --> End run if collision occurred
            return True

        else:
            # TODO: universalise reward function of gate passing (better value than 1?)
            # --> Check x position
            if self.goal_dict[str(goal)]["x"]-1 <= state[0][0] <= self.goal_dict[str(goal)]["x"] + 1:
                pass
            else:
                return False

            # --> Check y position
            if self.goal_dict[str(goal)]["y"] - 1 <= state[0][1] <= self.goal_dict[str(goal)]["y"] + 1:
                pass
            else:
                return False

            # --> Check z position
            if self.goal_dict[str(goal)]["z"] - 1 <= state[0][2] <= self.goal_dict[str(goal)]["z"] + 1:
                pass
            else:
                return False

            return True
