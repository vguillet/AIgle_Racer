
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs

# Own modules
from AIgle_Project.src.Navigation.Tools.ML_tools import ML_tools

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class RL_tools:
    def __init__(self):
        return

    @staticmethod
    def get_episode_parameters(episode, settings):
        ml_tools = ML_tools()

        # ----- Throttle learning rate
        # Throttle (decrease) learning rate according to episode
        learning_rate = ml_tools.throttle(episode, settings.rl_behavior_settings.episodes,
                                          settings.rl_behavior_settings.learning_rate, 0.99, 0.1,
                                          settings.rl_behavior_settings.learning_rate_decay,
                                          inverse=True, start_from_setting_value=True)
        # ----- Throttle discount
        # Throttle (increase) discount according to episode
        discount = ml_tools.throttle(episode, settings.rl_behavior_settings.episodes,
                                     settings.rl_behavior_settings.discount, 0.99, 0.1,
                                     settings.rl_behavior_settings.discount_decay,
                                     inverse=False, start_from_setting_value=True)

        # ----- Throttle epsilon
        # Throttle (decrease) epsilon according to episode
        epsilon = ml_tools.throttle(episode, settings.rl_behavior_settings.episodes,
                                    settings.rl_behavior_settings.epsilon, 100, 0,
                                    settings.rl_behavior_settings.epsilon_decay,
                                    inverse=True, start_from_setting_value=True)

        return learning_rate, discount, epsilon
