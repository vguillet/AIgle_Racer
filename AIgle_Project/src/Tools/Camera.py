
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys
from threading import Thread

# Libs
import airsim
import cv2
import numpy as np

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Camera:
    def __init__(self, client, camera_name, camera_ref,
                 camera_type="scene", fps=30):
        # --> Connect camera to client
        self.client = client

        # --> Define camera types available
        cameraTypeMap = {
                        "depth": airsim.ImageType.DepthVis,
                        "segmentation": airsim.ImageType.Segmentation,
                        "seg": airsim.ImageType.Segmentation,
                        "scene": airsim.ImageType.Scene,
                        "disparity": airsim.ImageType.DisparityNormalized,
                        "normals": airsim.ImageType.SurfaceNormals
                        }

        # --> Checking whether camera type selected exists
        if camera_type not in cameraTypeMap:
            print("!!!!! Invalid camera type setting")
            sys.exit(0)

        # --> Define attributes
        self.shutter_speed = int(1/fps * 1000)
        self.camera_name = camera_name
        self.camera_ref = camera_ref
        self.camera_type = cameraTypeMap[camera_type]

        self.camera_memory = []

    def run(self, postprocessor=None, record=False):
        counter = -1

        while True:
            # --> Fetch drone camera image
            image = self.fetch_single_img()

            # --> Encode image to png
            image = cv2.imdecode(airsim.string_to_uint8_array(image), cv2.IMREAD_UNCHANGED)

            # --> Run image through supplied pre-processor
            if postprocessor is not None:
                image = postprocessor(image)

            # --> Display using open cv
            self.camera_memory.append(image)
            cv2.imshow("Camera " + self.camera_name, image)

            # --> Wait for a key to be pressed to close window and resume simulation
            cv2.waitKey(self.shutter_speed)

            # --> Record image is record is enabled
            if record:
                counter += 1
                # --> Save png
                cv2.imwrite(os.path.normpath("image_" + str(counter) + '.png'), image)

    def fetch_single_img(self):
        # --> Fetch image from simulation using correct camera
        response = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_ref, self.camera_type, False, False)])[0]
        # image = self.client.simGetImage(self.camera_ref, self.camera_type)

        # --> Checking whether obtained image is not none
        if response is None:
            print("!!!!! Camera is not returning image, check airsim for error messages !!!!!")
            sys.exit(0)

        return response

    def fetch_and_record_single_image(self, file_name="Cam_shot"):
        # --> Fetch drone camera image
        image = self.fetch_single_img()

        # --> Encode image to png
        png = cv2.imdecode(airsim.string_to_uint8_array(image), cv2.IMREAD_UNCHANGED)

        # --> Save png
        cv2.imwrite(os.path.normpath(file_name + '.png'), png)
        return

    def display_camera_view(self):
        # --> Pause simulation
        self.client.simPause(True)

        # --> Fetch drone camera image
        image = self.fetch_single_img()

        # --> Encode image to png
        png = cv2.imdecode(airsim.string_to_uint8_array(image), cv2.IMREAD_UNCHANGED)

        # --> Display using open cv
        cv2.imshow("Camera " + self.camera_name, png)

        # --> Wait for a key to be pressed to close window and resume simulation
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.client.simPause(False)

        return
