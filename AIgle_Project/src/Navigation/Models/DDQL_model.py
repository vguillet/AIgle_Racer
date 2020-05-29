
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys

# Libs
import tensorflow as tf

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DDQL_model(object):
    def __init__(self, name,
                 input_dims, nb_actions,
                 checkpoint_directory, model_ref=None):

        # ---- Initiate model parameters
        self.name = name
        self.input_dims = input_dims
        self.nb_action = nb_actions

        self.params = tf.keras.Input(shape=(), dtype=tf.dtypes.float32)

        # --> Set checkpoint file directory
        self.checkpoint_path = os.path.join(checkpoint_directory, self.name + "_ddpg")

        if model_ref is None:
            # --> Create main model
            self.main_network = self.create_network()
            self.main_network.summary()

            # --> Create target network
            self.target_network = self.create_network()

            # --> Set target network weights equal to main model weights
            self.target_network.set_weights(self.main_network.get_weights())
        else:
            # --> Load main model
            self.main_network = self.load_checkpoint(model_ref)

            # --> Load target network
            self.target_network = self.load_checkpoint(model_ref)

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
        print(".. saving checkpoint ...")
        self.main_network.save(os.path.join(self.checkpoint_path, ref + ".h5"))
        return

    def load_checkpoint(self, model_ref):
        print(".. loading checkpoint ...")
        # TODO: Add load network
        return
