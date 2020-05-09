
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import time
import pprint

# Libs
import airsim

# Own modules
from AIgle_Project.src.Vision.Camera import Camera
from AIgle_Project.src.Vision.Postprocessor import Postprocessor
from AIgle_Project.src.Navigation.simple_predefined_flight import flight_navigation
from AIgle_Project.src.Navigation.Image_DQL.DQL_algorithm import DQL_image_based_navigation
from AIgle_Project.src.Navigation.Vector_DDQL.DDQL_algorithm import DQL_vector_based_navigation

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# ----- Initial setup of simulation
# --> Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.simPause(False)

# --> Setup simulator settings

# --> Run image based Image_DQL navigation
# dql_image_based = DQL_image_based_navigation(client)
dql_vector_based = DQL_vector_based_navigation(client)


# # ----- Creation of various code components
# --> Setup navigation
navigation = flight_navigation(client)

# --> Setup vision
center_cam = Camera(client, "center", "0")
postprocessor_1 = Postprocessor(average_smooth=0,
                                blur_gauss=0,
                                blur_median=0,
                                simple_colour_filter=1,
                                filter_laplacian=0,
                                grayscale=0,
                                threshold=0,
                                adaptive_threshold_gauss=0,
                                adaptive_threshold_otsu=0,
                                edge_detection=6,
                                corner_detection=5)

while True:

    navigation.run()

    while True:
        state = client.getMultirotorState()
        s = pprint.pformat(state.kinematics_estimated.position)
        print("state: %s \n" % s)
        time.sleep(1)

    center_cam.display_camera_view()
    center_cam.run(record=False,
                   postprocessor=postprocessor_1.constructed_postprocessor)
