
################################################################################################################
"""

"""

# Built-in/Generic Imports
import pprint

# Libs
import airsim

# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '30/04/2020'

################################################################################################################

# TODO: Implement name-based agent functions


class Agent:
    def __init__(self,
                 client,
                 name: "Bot name"):

        # ----- Setup settings
        self.settings = SETTINGS()
        self.settings.agent_settings.gen_agent_settings()

        # --> Setup Agent properties
        self.client = client
        self.name = name

        # --> Reset Drone to starting position
        self.client.reset()

        # --> Enable API control and take off
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(0, 0, -2, 3).join()

        return

    def move(self, x, y, z, v):
        # --> Move drone to specified position
        self.client.moveToPositionAsync(x, y, z, v,
                                        150,
                                        airsim.DrivetrainType.ForwardOnly,
                                        airsim.YawMode(False, 0), 20, 1).join()
        return

    def get_state(self, print_state=False):
        # --> Fetch rotor state
        state = self.client.getMultirotorState()
        if print_state:
            s = pprint.pformat(state.kinematics_estimated.position)
            print("state: %s \n" % s)

        return state

    def reset_agent(self, random_starting_pos=False):
        # TODO: Implement random offset starting point
        # --> Reset Drone to starting position
        self.client.reset()
        return

    def __str__(self):
        return self.name + " (Bot)"

    def __repr__(self):
        self.__repr__()
