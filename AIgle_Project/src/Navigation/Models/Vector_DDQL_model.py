
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


class Vector_DDQL_model(DDQL_model):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory="AIgle_Project/src/Navigation/Saved_models/Vector_ddpg/",
                 model_ref=None):
        super().__init__(name, input_dims, nb_actions, checkpoint_directory, model_ref)

    def create_network(self):
        # TODO: Review model

        print("\n self.input_dims:", self.input_dims)

        X_input = Input(self.input_dims)

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, input_shape=self.input_dims, activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(self.nb_action, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='CartPole DQN model')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model
