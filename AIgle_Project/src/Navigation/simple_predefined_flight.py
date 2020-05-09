
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys
import time

# Libs
import airsim
import cv2
import numpy as np

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class flight_navigation():
    def __init__(self, client):
        # --> Navigation_airsim_vanilla to client
        self.client = client

    def run(self):
        # --> Reset Drone to starting position
        self.client.reset()

        # --> Enable API control and take off
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(0, 0, -2, 3).join()

        # --> Set path
        result = self.client.moveOnPathAsync([airsim.Vector3r(0, -40, 2),
                                               airsim.Vector3r(-10, -72, 0),
                                               airsim.Vector3r(-25, -65, 4),
                                               airsim.Vector3r(-40, -65, 0),
                                               airsim.Vector3r(-62, -55, 0),
                                               airsim.Vector3r(-65, 30, 0),
                                               airsim.Vector3r(-50, 45, 4),
                                               airsim.Vector3r(-35, 55, 5),
                                               airsim.Vector3r(-10, 40, 5),
                                               airsim.Vector3r(-0, 25, 4),
                                               airsim.Vector3r(0, 0, -2),
                                               airsim.Vector3r(0, 0, -2),
                                               airsim.Vector3r(0, 0, -2),
                                               ],
                                              16, 150,
                                              airsim.DrivetrainType.ForwardOnly,
                                              airsim.YawMode(False, 0), 20, 1).join()

        self.client.moveToPositionAsync(0, -20, -2, 10).join()
        time.sleep(2)

        self.client.moveToPositionAsync(0, -20, 4, 10).join()
        # time.sleep(2)
        # self.client.moveToPositionAsync(0, -28.5, -2, 1).join()
        # time.sleep(2)
        # self.client.moveToPositionAsync(0, -28.5, -6, 1)

        # self.client.moveToPositionAsync(0, -28, -2, 1).join()
        # time.sleep(2)
        # self.client.moveToPositionAsync(0, -28, -2, 1).join()
        # self.client.moveToPositionAsync(-2, -26.5, -2, 2).join()

        # print("1")
        # self.client.moveToPositionAsync(-40, -60, -2, 16, drivetrain=airsim.DrivetrainType.ForwardOnly).join()
        print("2")
