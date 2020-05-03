
##################################################################################################################
"""
Used to combine all setting classes
"""

# Built-in/Generic Imports

# Libs

# Own modules
from Settings.Print_plot_settings import Print_plot_settings
from Settings.RL_behavior_settings import RL_behavior_settings
from Settings.Environment_settings import Environment_settings
from Settings.Agent_settings import Agent_settings

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '7/02/2020'

##################################################################################################################


class SETTINGS:
    def __init__(self):
        self.print_plot_settings = Print_plot_settings()
        self.rl_behavior_settings = RL_behavior_settings()
        self.environment_settings = Environment_settings()
        self.agent_settings = Agent_settings()

    def cache_settings(self):
        with open("Settings/Settings_cache/Print_settings", "w") as cache:
            cache.write("Environment_prints " + str(self.print_plot_settings.environment_prints) + "\n")
            cache.write("Environment_plots " + str(self.print_plot_settings.environment_plots) + "\n")
