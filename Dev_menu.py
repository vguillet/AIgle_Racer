
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs
import airsim

# Own modules
from AIgle_Project.src.Navigation.RL_algorithm import RL_navigation

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

# ---- Initial setup of simulation
# --> Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.simPause(False)
# client.simSetTraceLine(color="")

# ---- Creation of various code components
# --> Setup navigation

rl_navigation = RL_navigation(client)

#
# client.reset()
# client.simFlushPersistentMarkers()
#
# client.enableApiControl(True)
# client.armDisarm(True)
#
# # Goal 0
# client.moveOnPathAsync([airsim.Vector3r(0, -28.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 1
# client.moveOnPathAsync([airsim.Vector3r(-6, -53.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 2
# client.moveOnPathAsync([airsim.Vector3r(-54, -57.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 3
# client.moveOnPathAsync([airsim.Vector3r(-65, -14.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 4
# client.moveOnPathAsync([airsim.Vector3r(-53.5, 28.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 5
# client.moveOnPathAsync([airsim.Vector3r(-12, 32.5, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# # Goal 6
# client.moveOnPathAsync([airsim.Vector3r(0, 0, -2)],
#                         12, 150,
#                         airsim.DrivetrainType.ForwardOnly,
#                         airsim.YawMode(False, 0), 20, 1).join()
#
# client.simPause(True)
#
# state = client.getMultirotorState()
# s = pprint.pformat(state.kinematics_estimated.position)
# print("state: %s \n" % s)
# time.sleep(1)
#
# client.simPause(False)
