
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys

from datetime import datetime

# Libs
import tensorflow as tf
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam, RMSprop

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DDQL_model(object):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory,
                 model_ref=None):

        # --> Seed tensorflow
        tf.random.set_seed(10)

        # ---- Initiate model parameters
        self.name = name
        self.type = "ddql"
        self.input_dims = input_dims
        self.nb_action = nb_actions

        self.params = tf.keras.Input(shape=(), dtype=tf.dtypes.float32)

        # --> Set checkpoint file directory
        self.checkpoint_path = checkpoint_directory

        if model_ref is None:
            # --> Create main model
            self.main_network = self.create_network()

            # --> Create target network and network weights equal to main model weights
            self.target_network = self.create_network()
            self.target_network.set_weights(self.main_network.get_weights())

        else:
            self.load_checkpoint(model_ref)

        # plot_model(self.main_network,
        #            to_file=os.path.join(self.checkpoint_path, self.name + "_" + self.type + ".png"),
        #            show_shapes=True,
        #            show_layer_names=True,
        #            rankdir='TB')

        self.main_network.summary()

    def soft_update_target(self, tau):
        # --> Soft update using tau
        main_weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(main_weights)):
            target_weights[i] = tau * main_weights[i] + (1 - tau) * target_weights[i]
        self.target_network.set_weights(target_weights)
        return

    def hard_update_target(self):
        # --> Hard update
        main_weights = self.main_network.get_weights()
        self.target_network.set_weights(main_weights)
        return

    def create_network(self):
        print("!!!!! No network specified !!!!!")
        sys.exit()

    def save_checkpoint(self, ref):
        print("\n... saving checkpoint ...\n")
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.target_network.save(os.path.join(self.checkpoint_path, self.name + "_" + self.type + "_" + str(ref) + ".h5"))

        return

    def load_checkpoint(self, ref, mode="simple"):
        if mode == "simple":
            print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("\n... loading checkpoint ...\n")

            # --> Load main model
            self.main_network = load_model(ref, compile=True)

            # --> Load target network
            self.target_network = load_model(ref, compile=True)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        elif mode == "transfer":
            print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("\n... loading checkpoint, swapping last layer for transfer learning ...\n")

            # ----- Load main network
            loaded_main_network = load_model(ref, compile=True)

            # --> removing last layer (load outputs before last layer)
            x = loaded_main_network.layers[-1].output

            # --> Add hidden layer
            x = Dense(32, activation="relu", kernel_initializer='he_uniform')(x)

            # Output Layer with # of actions
            x = Dense(self.nb_action, activation="linear", kernel_initializer='he_uniform')(x)
            self.main_network = Model(inputs=loaded_main_network.input, outputs=x)

            # ----- Load target network
            loaded_target_network = load_model(ref, compile=True)

            # --> removing last layer (load outputs before last layer)
            x = loaded_target_network.layers[-1].output

            # --> Add hidden layer
            x = Dense(32, activation="relu", kernel_initializer='he_uniform')(x)

            # Output Layer with # of actions
            x = Dense(self.nb_action, activation="linear", kernel_initializer='he_uniform')(x)
            self.main_network = Model(inputs=loaded_main_network.input, outputs=x)


            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        return
