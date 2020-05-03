
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os

# Libs
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQL_models:
    @staticmethod
    def model_1(input_shape, action_space):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=input_shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(action_space, activation='linear'))  # how many choices
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model
