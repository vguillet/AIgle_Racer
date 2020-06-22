
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
    run_mode = "Train"
    # run_mode = "Test"

    run_algorithm = "DDQL"
    # run_algorithm = "DDPG"

    def gen_ddql_settings(self):
        # ---- General run settings
        self.run_name = "Run_FSFG"
        self.training_type = "Track"     # Can be Door or Track
        self.algorithm = "DDQL"
        # ___________________________ Print/plot parameters ______________________
        # self.print_action_process = False

        # ---- Stats settings
        # self.plot_best_agent_stats = True

        # ---- Epoque batch settings (for visualisation/stats purposes only)
        self.batch_epoque_size = 100
        self.plot_epoque_batch_reward = True

        # ---- Simulation settings
        # ["all", "batch", "individual", "step", "none"]
        self.show_tracelines = "batch"

        # ---- Progress management
        self.save_model_on_batch = True

        self.model_ref = None
        # self.model_ref = "AIgle_Project/src/Navigation/Saved_models/Vector_ddql/Track/Run_9/Vector_ddql_10000.h5"

        # ___________________________ Main parameters ____________________________
        # ---- Run settings
        self.epoques = 10_000

        # ---- Stats settings
        # self.stats_sampling_rate = 100

        # ---- Memory settings
        # ["simple, "prioritized"]
        self.memory_type = "prioritized"

        self.memory_size = 5_000
        self.min_replay_memory_size = 250
        self.minibatch_size = 32

        self.memory_ref = None

        # ---- Learning settings
        # self.learning_rate = 0.3      # learn nothing (privilege long term) 0 <-- x --> 1 only consider recent info
        self.discount = 0.75            #  gamma     75            short-term reward 0 <-- x --> 1 long-term reward
        self.tau = 0.001                # Rate at which target weights change

        self.hard_update_target_every = None

        # ---- Exploration settings
        self.epsilon = 20               # Probability (percent) of taking random action

        # ---- Decay settings
        self.decay_functions = ["Fixed value", "Linear decay", "Exponential decay", "Logarithmic decay"]

        self.tau_decay = 0
        self.discount_decay = 0
        self.epsilon_decay = 1

        return

    def gen_ddpg_settings(self):
        # ---- General run settings
        self.run_name = "Run_1"
        self.training_type = "Door"     # Can be Door or Track
        self.algorithm = "DDPG"
        # ___________________________ Print/plot parameters ______________________
        # self.print_action_process = False

        # ---- Stats settings
        # self.plot_best_agent_stats = True

        # ---- Epoque batch settings (for visualisation/stats purposes only)
        self.batch_epoque_size = 100
        self.plot_epoque_batch_reward = True

        # ---- Simulation settings
        # ["all", "batch", "individual", "step", "none"]
        self.show_tracelines = "batch"

        # ---- Progress management
        self.save_model_on_batch = True

        self.actor_ref = None
        self.critic_ref = None

        # ___________________________ Main parameters ____________________________
        # ---- Run settings
        self.epoques = 5_000

        # ---- Stats settings
        # self.stats_sampling_rate = 100

        # ---- Memory settings
        # ["simple, "prioritized"]
        self.memory_type = "prioritized"

        self.memory_size = 5_000
        self.min_replay_memory_size = 250
        self.minibatch_size = 64

        self.memory_ref = None

        # ---- Learning settings
        self.actor_learning_rate = 0.001    # learn nothing (privilege long term) 0 <-- x --> 1 only consider recent info
        self.critic_learning_rate = 0.0001
        self.discount = 0.75            #                   short-term reward 0 <-- x --> 1 long-term reward
        self.tau = 0.001                # Rate at which target weights change

        self.hard_update_target_every = None

        # ---- Exploration settings
        self.epsilon = 25               # Probability (percent) of taking random action

        # ---- Decay settings
        self.decay_functions = ["Fixed value", "Linear decay", "Exponential decay", "Logarithmic decay"]

        self.tau_decay = 0
        self.discount_decay = 0
        self.epsilon_decay = 1

        return
