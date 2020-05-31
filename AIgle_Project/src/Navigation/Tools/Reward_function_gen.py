
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
        distance_from_goal = math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2)

        if collision is True:
            # --> Negative reward if a collision occurred (not included for now)
            reward = -100
            # reward = -(math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2))

        else:
            if distance_from_goal < 2:
                reward = 200

            else:
                reward = -(math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2))
                #  (math.sqrt((self.goal_dict[str(goal)]["x"])**2 +
                #                     (self.goal_dict[str(goal)]["y"])**2 +
                #                     (self.goal_dict[str(goal)]["z"])**2))

        return reward

    def check_if_done(self, state, goal, collision, age, max_age):

        distance_from_goal = math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2)

        if age >= max_age:
            # --> End run if max age reached
            return True

        if collision is True:
            # --> End run if collision occurred
            return True

        # --> Check x position
        if distance_from_goal < 2:
            return True

        return False
