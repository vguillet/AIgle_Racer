
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

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Actor_model(object):
    def __init__(self, sess, name, learning_rate, tau, mu,
                 input_dims, nb_actions, action_bound, checkpoint_directory="tmp/ddpg/actor"):
        # ---- Initiate model parameters
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate

        self.tau = tau

        self.input_dims = input_dims
        self.action_dims = nb_actions

        self.action_bound = action_bound

        self.params = tf.keras.Input(shape=(), dtype=tf.dtypes.float32)

        # --> Set checkpoint file directory
        self.checkpoint_file = os.path.join(checkpoint_directory, name + "_ddpg")

        # --> Create main model
        self.main_network = self.create_network()

        # --> Create target network
        self.target_network = self.create_network()

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
        state = Input(shape=self.input_dims, dtype='float32')
        x = Dense(400, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(self.input_dims),
                                        1 / np.sqrt(self.input_dims)))(state)

        x = Dense(300, activation='relu',
                  kernel_initializer=RU(-1 / np.sqrt(400), 1 / np.sqrt(400)))(x)

        out = Dense(self.action_dims,
                    activation='tanh',
                    kernel_initializer=RU(-0.003, 0.003))(x)

        return Model(inputs=state, outputs=out)
