
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


class Postprocessor:
    def __init__(self,
                 average_smooth=0,
                 blur_gauss=0,
                 blur_median=0,
                 simple_colour_filter=0, filter_laplacian=0,
                 grayscale=1,
                 threshold=0, adaptive_threshold_gauss=0, adaptive_threshold_otsu=0,
                 edge_detection=0,
                 corner_detection=0):

        # ----- Setting up filters
        self.average_smooth = average_smooth

        self.blur_gauss = blur_gauss
        self.blur_median = blur_median

        self.simple_colour_filter = simple_colour_filter
        self.filter_laplacian = filter_laplacian

        self.grayscale = grayscale

        # ----- Setting up thresholds
        self.threshold = threshold
        self.adaptive_threshold_gauss = adaptive_threshold_gauss
        self.adaptive_threshold_otsu = adaptive_threshold_otsu

        # ----- Setting up other
        self.edge_detection_canny = edge_detection

        self.corner_detection = corner_detection
        return

    def constructed_postprocessor(self, image):
        """
        Note: Gauss and otsu thresholds cannot be used in succession
        """
        sorted_processors = sorted(self.__dict__.items(), key=lambda x: x[1])
        for i in sorted_processors:
            if i[1] == 0:
                pass
            else:
                print("run_" + i[0])
                processor = getattr(self, "run_" + i[0])
                image = processor(image)

        return image

    @staticmethod
    def run_average_smooth(image):
        kernel = np.ones((15,15), np.float32)/255
        image = cv2.filter2D(image, -1, kernel)
        return image

    @staticmethod
    def run_blur_gauss(image):
        image = cv2.GaussianBlur(image, (15, 15), 0)
        return image

    @staticmethod
    def run_blur_median(image):
        image = cv2.medianBlur(image, 15)
        return image

    @staticmethod
    def run_simple_colour_filter(image):
        # --> Convert image to hsv color field
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --> Setting upper/lower colors
        # (hue, saturation, value)
        # -> Hue = color
        # -> Saturation = How bright/dull a colour is
        # -> Value = How light/dark

        lower_color = np.array([5, 50, 50])
        upper_color = np.array([15, 255, 255])

        # --> Build mask
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # --> Build result
        image = cv2.bitwise_and(image, image, mask=mask)
        return image

    @staticmethod
    def run_filter_laplacian(image):
        image = cv2.Laplacian(image, cv2.CV_64F)
        return image

    @staticmethod
    def run_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def run_threshold(image):
        _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        return image

    def run_adaptive_threshold_gauss(self, image):
        # --> First grayscale image (required)
        image = self.run_grayscale(image)

        # --> Apply Gauss adaptive threshold
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        return image

    def run_adaptive_threshold_otsu(self, image):
        # --> First grayscale image (required)
        image = self.run_grayscale(image)

        # --> Apply otsu adaptive threshold
        _, image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return image

    @staticmethod
    def run_edge_detection_canny(image):
        image = cv2.Canny(image, 100, 200)
        return image

    @staticmethod
    def run_corner_detection(image):
        # --> Convert image to gray (required)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # --> Detect corners
        # (image, max corners to detect, quality, and minimum distance between corners)
        corners = cv2.goodFeaturesToTrack(gray, 8, 0.01, 25)
        corners = np.int0(corners)

        # --. Plot corners on image
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, 255, -1)

        return image

    @staticmethod
    def run_optical_flow_lucas_kanade(image):
        return