
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

        # ---- Episode batch settings
        self.batch_episode_size = 50
        self.plot_episode_batch_reward = True


        # ___________________________ Main parameters ____________________________
        # ---- Agent ref
        self.model_ref = None

        self.memory_ref = None

        # ---- Stats settings
        self.stats_sampling_rate = 100

        # ---- Agent properties
        # ["simple, "prioritized"]
        self.memory_type = "prioritized"

        self.memory_size = 5_000
        self.min_replay_memory_size = 250

        # ---- Run settings
        self.episodes = 10_000

        # ---- Cycle settings
        self.cyclic_training = False

        # ---- Learning settings
        # self.learning_rate = 0.3        # learn nothing (privilege long term) 0 <-- x --> 1 only consider recent info
        self.discount = 0.75            #                   short-term reward 0 <-- x --> 1 long-term reward

        self.minibatch_size = 20
        self.tau = 0.001                # Rate at which target weights change
        self.update_target_every = 100000000

        # ---- Exploration settings
        self.epsilon = 35               # Probability (percent) of taking random action
        self.random_starting_pos = False

        # ---- Decay settings
        self.decay_functions = ["Fixed value", "Linear decay", "Exponential decay", "Logarithmic decay"]

        self.tau_decay = 0
        self.discount_decay = 1
        self.epsilon_decay = 1

        return

    def gen_ddpg_settings(self):
        # ___________________________ Print/plot parameters ______________________

        # ___________________________ Main parameters ____________________________
        # ---- Agent ref
        self.actor_ref = None
        self.critic_ref = None

        self.memory_ref = None

        # ---- Agent properties
        # ["simple, "prioritized"]
        self.memory_type = "simple"
        self.memory_size = 5_000
        self.min_replay_memory_size = 250

        # ---- Run settings
        self.episodes = 2_000

        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.0001

        self.minibatch_size = 64
        self.tau = 0.001                    # Rate at which target weights change

        self.discount = 0.98            #                   short-term reward 0.9 <-- x --> 0.99 long-term reward



        return
