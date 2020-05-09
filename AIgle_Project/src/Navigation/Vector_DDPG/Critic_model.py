
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
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Critic_model:
    def __init__(self, name, learning_rate, tau,
                 input_dims, nb_actions, checkpoint_directory="tmp/ddpg/critic"):
        # ---- Initiate model parameters
        self.name = name
        self.learning_rate = learning_rate

        self.tau = tau

        self.input_dims = input_dims
        self.action_dims = nb_actions

        # --> Set checkpoint file directory
        self.checkpoint_file = os.path.join(checkpoint_directory, name+"_ddpg")

        # --> Create main model
        self.main_network, self.main_action, self.main_state = self.create_network()

        # --> Create target network
        self.target_network, self.target_action, self.target_state = self.create_network()

        # --> Set target network weights equal to main model weights
        self.target_network.set_weights(self.main_network.get_weights())

    def train_target(self):
        main_weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(main_weights)):
            target_weights[i] = self.tau * main_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_network.set_weights(target_weights)
        return

    def create_network(self):
        # TODO: Review model
        state = Input(shape=self.input_dims, name='state_input', dtype='float32')
        state_i = Dense(300,
                        activation='relu',
                        kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims),
                                              1 / np.sqrt(self.input_dims)))(state)

        action = Input(shape=(self.action_dims,), name='action_input')

        x = Concatenate([state_i, action])

        x = Dense(600,
                  activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(401),
                                        1 / np.sqrt(401)))(x)

        out = Dense(4, activation='linear')(x)

        return Model(inputs=[state, action], outputs=out)

    # def create_network(self):
    #     S = Input(shape=[self.input_dims])
    #     A = Input(shape=[self.action_dims], name='action2')
    #     w1 = Dense(300, activation='relu')(S)
    #     a1 = Dense(600, activation='linear')(A)
    #     h1 = Dense(600, activation='linear')(w1)
    #     h2 = Add()([h1, a1])
    #     h3 = Dense(600, activation='relu')(h2)
    #
    #     V = Dense(self.action_dims, activation='linear')(h3)
    #
    #     model = Model(input=[S, A], output=V)
    #     adam = Adam(lr=self.learning_rate)
    #     model.compile(loss='mse', optimizer=adam)
    #     return model, A, S

    def save_checkpoint(self):
        print(".. saving checkpoint ...")
        # TODO: Add save network
        return

    def load_checkpoint(self):
        print(".. loading checkpoint ...")
        # TODO: Add load network
        return