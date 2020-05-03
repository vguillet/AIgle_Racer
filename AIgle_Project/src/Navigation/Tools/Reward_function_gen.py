
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Reward_function:
    def __init__(self):
        return

    def get_reward(self, state, collision, age):
        if collision is True:
            # --> Negative reward if a collision occurred
            reward = -10

        else:
            # --> Set default reward to 0
            reward = 1000

            # --> Check x position
            if -2.65 <= state[0][0] <= 3.27:
                pass
            else:
                reward = -1

            # --> Check y position
            if -28 <= state[0][0] <= -29:
                pass
            else:
                reward = -1

            # --> Check z position
            if -4.93 <= state[0][2] <= 0.7596:
                pass
            else:
                reward = -1

        return reward

    def check_if_done(self, state, collision, age):
        # TODO: Connect age to settings
        if age >= 10:
            return True

        if collision is True:
            # --> Kill run if collision occurred
            return True

        else:
            # --> Set default reward to 0
            reward = 0

            # --> Check x position
            if -2.65 <= state[0][0] <= 3.27:
                pass
            else:
                return False

            # --> Check y position
            if -28 <= state[0][0] <= -29:
                pass
            else:
                return False

            # --> Check z position
            if -4.93 <= state[0][2] <= 0.7596:
                pass
            else:
                return False

            return True
