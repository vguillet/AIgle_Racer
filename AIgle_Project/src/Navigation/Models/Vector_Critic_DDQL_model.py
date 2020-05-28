
##################################################################################################################
"""
https://www.youtube.com/watch?v=6Yd5WnYls_Y
"""

# Built-in/Generic Imports
import os

# Libs
import numpy as np

from keras.initializers import RandomUniform as RU
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate

# Own modules
from AIgle_Project.src.Navigation.Models.DDQL_model import DDQL_model

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Critic_DDQL(DDQL_model):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory="Data/ddpg/critic", model_ref=None):
        super().__init__(name, input_dims, nb_actions, checkpoint_directory, model_ref)

    def create_network(self):
        # TODO: Review model
        state = Input(shape=self.input_dims, name='state_input', dtype='float32')
        state_i = Dense(400,
                        activation='relu',
                        kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims)))(state)

        action = Input(shape=(self.action_dims,), name='action_input')

        x = Concatenate([state_i, action])

        x = Dense(300,
                  activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(401),
                                        1 / np.sqrt(401)))(x)

        out = Dense(4, activation='linear')(x)

        model = Model(inputs=[state, action], outputs=out)
        return model
