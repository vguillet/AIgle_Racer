
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
    def get_epoque_parameters(epoque, settings):
        ml_tools = ML_tools()

        # ----- Throttle tau
        # Throttle (decrease) tau according to epoque
        tau = ml_tools.throttle(epoque, settings.rl_behavior_settings.epoques,
                                settings.rl_behavior_settings.tau, 0.001, 0.0008,
                                settings.rl_behavior_settings.tau_decay,
                                inverse=True, start_from_setting_value=True)

        # ----- Throttle discount
        # Throttle (increase) discount according to epoque
        discount = ml_tools.throttle(epoque, settings.rl_behavior_settings.epoques,
                                     settings.rl_behavior_settings.discount, 0.99, 0.1,
                                     settings.rl_behavior_settings.discount_decay,
                                     inverse=False, start_from_setting_value=True)

        # ----- Throttle epsilon
        # Throttle (decrease) epsilon according to epoque
        epsilon = ml_tools.throttle(epoque, settings.rl_behavior_settings.epoques-500,
                                    settings.rl_behavior_settings.epsilon, 100, 5,
                                    settings.rl_behavior_settings.epsilon_decay,
                                    inverse=True, start_from_setting_value=True)

        return round(tau, 4), round(discount, 4), round(epsilon, 4)
