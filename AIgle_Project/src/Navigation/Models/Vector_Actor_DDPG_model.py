
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
from keras.optimizers import Adam, RMSprop

import tensorflow as tf

# Own modules
from AIgle_Project.src.Navigation.Models.DDQL_model import DDQL_model

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Vector_Actor_DDPG_model(DDQL_model):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory="Data/ddpg/actor",
                 model_ref=None):
        super().__init__(name, input_dims, nb_actions, checkpoint_directory, model_ref)

    def create_network(self):
        # TODO: Review model

        x_input = Input(shape=self.input_dims)
        x = Dense(400, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                        1 / np.sqrt(self.input_dims)))(x_input)

        x = Dense(300, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(400), 1 / np.sqrt(400)))(x)

        x = Dense(self.nb_action,
                  activation='tanh',
                  kernel_initializer=RU(-0.003, 0.003))(x)

        model = Model(inputs=x_input, outputs=x, name='AIgle_Racer_DDPG_actor_model')
        # model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model
