
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
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam, RMSprop

# Own modules
from AIgle_Project.src.Navigation.Models.DDQL_model import DDQL_model

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Vector_Critic_DDQL_model(DDQL_model):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory="Data/ddpg/critic",
                 model_ref=None):
        super().__init__(name, input_dims, nb_actions, checkpoint_directory, model_ref)

    def create_network(self):
        # TODO: Review model
        state = Input(shape=self.input_dims, name='state_input')
        state_i = Dense(400,
                        activation='relu',
                        kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims)))(state)

        action = Input(shape=(self.nb_action,), name='action_input')

        x = concatenate([state_i, action])

        x = Dense(300,
                  activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(401),
                                        1 / np.sqrt(401)))(x)

        x = Dense(self.nb_action, activation='linear')(x)

        model = Model(inputs=[state, action], outputs=x, name='AIgle_Racer_DDPG_critic_model')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        return model
