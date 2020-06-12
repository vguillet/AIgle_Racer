
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

from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

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
        # --> Setup input
        input = Input(shape=self.input_dims)

        # --> Setup network hidden structure
        x = Dense(400, activation='relu')(input)
        x = Dense(300, activation='relu')(x)

        # --> Setup output
        output = Dense(self.nb_action, activation='tanh')(x)

        # --> Build model
        model = Model(inputs=input, outputs=output, name='AIgle_Racer_DDPG_actor_model')
        # model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model

        # state = Input(shape=state_shape)
        # x = Dense(units[0], name="L0", activation='relu')(state)
        # for index in range(1, len(units)):
        #     x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
        #
        # unscaled_output = Dense(action_dim, name="Out", activation='tanh')(x)
        # scalar = action_bound * np.ones(action_dim)
        # output = Lambda(lambda op: op * scalar)(unscaled_output)
        # if np.sum(action_shift) != 0:
        #     output = Lambda(lambda op: op + action_shift)(output)  # for action range not centered at zero
        #
        # model = Model(inputs=state, outputs=output)
        #
        # return model