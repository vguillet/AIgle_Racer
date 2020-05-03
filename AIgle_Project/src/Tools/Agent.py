
################################################################################################################
"""

"""

# Built-in/Generic Imports
import pprint
import time

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

        return

    @property
    def state(self):
        return self.client.getMultirotorState()

    @property
    def collision(self):
        return self.state.collision.has_collided

    def move(self, new_state):
        print("=======================> Move", new_state[0][0],
                                        new_state[0][1],
                                        new_state[0][2],
                                        new_state[1])

        # # --> Move drone to specified position
        self.client.moveToPositionAsync(new_state[0][0],
                                        new_state[0][1],
                                        new_state[0][2],
                                        new_state[1]
                                        )

        time.sleep(1.5)

        # result = self.client.moveOnPathAsync([airsim.Vector3r(new_state[0][0],
        #                                                       new_state[0][1],
        #                                                       new_state[0][2])],
        #                                      new_state[1], 150,
        #                                      airsim.DrivetrainType.ForwardOnly,
        #                                      airsim.YawMode(False, 0), 20, 1).join()

        return

    def reset(self, random_starting_pos=False):
        print("=======================> RESET")

        # TODO: Implement random offset starting point
        # --> Reset Drone to starting position
        self.client.reset()

        # --> Enable API control and take off
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(0, 0, -2, 3).join()

        return

    def __str__(self):
        return self.name + " (Bot)"

    def __repr__(self):
        self.__repr__()
