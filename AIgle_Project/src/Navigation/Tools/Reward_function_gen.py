
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
        return

    def get_reward(self, state, collision, age):
        if collision is True:
            # --> Negative reward if a collision occurred
            reward = -10

        else:
            # --> Set default reward to 0
            reward = 1

            # --> Check x position
            if -2.65 <= state[0][0] <= 3.27 and -28 <= state[0][1] <= -29 and -4.93 <= state[0][2] <= 0.7596:
                pass
            else:
                reward = -(math.sqrt((0 - state[0][0])**2 + (-28.5 - state[0][1])**2 + (-2 - state[0][2])**2)/math.sqrt((0)**2 + (-28.5)**2 + (-2)**2))

        return reward

    def check_if_done(self, state, collision, age):
        # TODO: Connect age to settings
        if age >= 10:
            return True

        if collision is True:
            # --> Kill run if collision occurred
            return True

        else:
            # --> Check x position
            if -2.65 <= state[0][0] <= 3.27:
                pass
            else:
                return False

            # --> Check y position
            if -28 <= state[0][1] <= -29:
                pass
            else:
                return False

            # --> Check z position
            if -4.93 <= state[0][2] <= 0.7596:
                pass
            else:
                return False

            return True
