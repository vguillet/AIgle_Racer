
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '7/02/2020'

##################################################################################################################


class RL_behavior_settings:
    def gen_dql_settings(self):
        # ___________________________ Print/plot parameters ______________________
        # ---- General run settings
        self.print_action_process = False

        # ---- Stats settings
        self.plot_best_agent_stats = True

        # ---- Cycle settings
        self.plot_best_agent_reward_timeline = False
        self.plot_best_agent_inventory = True

        self.show_best_agent_visualiser = False

        # ___________________________ Image_DQL main parameters __________________
        # ---- Stats settings
        self.stats_sampling_rate = 100

        # ---- Agent properties
        self.memory_size = 5_000
        self.min_replay_memory_size = 250

        # ---- Run settings
        self.episodes = 20_000

        # ---- Cycle settings
        self.cyclic_training = False

        # ---- Learning settings
        self.learning_rate = 0.3        # learn nothing (privilege long term) 0 <-- x --> 1 only consider recent info
        self.discount = 0.75            #                   short-term reward 0 <-- x --> 1 long-term reward

        self.minibatch_size = 20
        self.update_target_every = 5

        # ---- Exploration settings
        self.epsilon = 25               # Probability (percent) of taking random action
        self.random_starting_pos = False

        # ---- Decay settings
        self.decay_functions = ["Fixed value", "Linear decay", "Exponential decay", "Logarithmic decay"]

        self.learning_rate_decay = 1
        self.discount_decay = 1
        self.epsilon_decay = 1

        return

    def gen_ddql_settings(self):
        # ___________________________ Print/plot parameters ______________________

        # ___________________________ Image_DQL main parameters __________________
        # ---- Agent properties
        self.memory_size = 5_000
        self.min_replay_memory_size = 250

        # ---- Run settings
        self.episodes = 2_000

        self.critic_learning_rate = 0.0001
        self.tau = 0.001                    # Rate at which target weights change




        return
