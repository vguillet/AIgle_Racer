
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import time

# Libs
import airsim

# Own modules
from AIgle_Vision.Vision.Camera import Camera
from AIgle_Vision.Vision.Postprocessor import Postprocessor
from AIgle_Vision.Navigation.simple_predefined_flight import flight_navigation


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

# ----- Initial setup of simulation
# --> Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.simPause(False)

# --> Setup simulator settings
# client.simSetTraceLine((255, 0, 0, 1), thickness=3)

# ----- Creation of various code components
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
    # center_cam.display_camera_view()
    center_cam.run(record=False,
                   postprocessor=postprocessor_1.constructed_postprocessor)
