
##################################################################################################################
"""
https://www.youtube.com/watch?v=6Yd5WnYls_Y
"""

# Built-in/Generic Imports
import os

# Libs
import numpy as np

from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop

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
        # --> Setup input
        input = [Input(shape=self.input_dims),
                 Input(shape=(self.nb_action,))]

        concat = Concatenate(axis=-1)(input)

        # --> Setup network hidden structure
        x = Dense(48, activation='relu')(concat)
        x = Dense(24, activation='relu')(x)

        # --> Setup output
        output = Dense(1)(x)

        # --> Build model
        model = Model(inputs=input, outputs=output, name='AIgle_Racer_DDPG_critic_model')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model
