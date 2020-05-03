
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys

# Libs
from keras.models import Model, load_model

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class ML_tools:
    def __init__(self):
        return

    @staticmethod
    def load(name):
        model = load_model(name)

        return model

    @staticmethod
    def save(model, name):
        model.save(name)

        return

    @staticmethod
    def throttle(current_step: int, nb_steps: int,
                 setting_value, max_value, min_value,
                 decay_function: int = 0,
                 inverse: bool = False, start_from_setting_value: bool = True):
        """
        Throttle a value according to the instance in the run time.

        The following decay functions settings can be used:
                0 - Fixed value (no decay, returns max value)

                1 - Linear decay

                2 - Logarithmic decay (in development)

        :param current_step: Current step
        :param nb_steps: Total number of step in the run
        :param setting_value: Setting value
        :param max_value: Max allowed value
        :param min_value: Min allowed value
        :param decay_function: Decay function setting
        :param inverse: Decrease throttled value instead of increasing it
        :param start_from_setting_value: Set to decide whether to throttle from setting value
        :return: Throttled value
        """
        from math import log10

        if start_from_setting_value:
            if not inverse:
                min_value = setting_value
            else:
                max_value = setting_value

        if current_step <= nb_steps:
            if decay_function == 0:  # Fixed value
                return setting_value

            elif decay_function == 1:  # Linear decay
                if not inverse:
                    throttled_value = setting_value + (max_value - min_value) / nb_steps * current_step
                    if throttled_value >= max_value:
                        throttled_value = max_value
                else:
                    throttled_value = setting_value - (max_value - min_value) / nb_steps * current_step
                    if throttled_value <= min_value:
                        throttled_value = min_value

            # TODO: Complete log decay
            elif decay_function == 2:  # Logarithmic decay
                throttled_value = setting_value + log10(-(current_step - nb_steps))

            # TODO: add decay functions (log/exponential etc...)
            else:
                # -- Exit program if incorrect settings used
                raise Exception("Invalid throttle function setting")

        else:
            if not inverse:
                throttled_value = max_value
            else:
                throttled_value = min_value

        return throttled_value

