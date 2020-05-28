
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys
import random
import time
import os

# Libs
import numpy as np

from keras.initializers import RandomUniform as RU
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam

import tensorflow as tf

# Own modules
from AIgle_Project.src.Navigation.Models.DDQL_model import DDQL_model

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Actor_DDQL(DDQL_model):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory="Data/ddpg/actor", model_ref=None):
        super().__init__(name, input_dims, nb_actions, checkpoint_directory, model_ref)

    def create_network(self):
        # TODO: Review model

        observation = Input(shape=self.input_dims, dtype='float32')
        x = Dense(400, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                        1 / np.sqrt(self.input_dims)))(observation)

        x = Dense(300, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(400), 1 / np.sqrt(400)))(x)

        out = Dense(self.action_dims,
                    activation='tanh',
                    kernel_initializer=RU(-0.003, 0.003))(x)

        model = Model(inputs=observation, outputs=out)
        return model
