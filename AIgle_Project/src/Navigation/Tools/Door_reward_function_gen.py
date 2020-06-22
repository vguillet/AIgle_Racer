
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import math

# Libs
from keras.models import Sequential


# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Door_reward_function(object):
    def __init__(self):
        self.direction = 1
        self.goal_dict = {"0": {"x": 0,
                                "y": -28.5,
                                "z": -2,
                                "turn": 0},
                          "1": {"x": -6,
                                "y": -53.5,
                                "z": -2,
                                "turn": 0},
                          # "2": {"x": -24.5,
                          #       "y": -71.5,
                          #       "z": -2,
                          #       "turn": 0},
                          "3": {"x": -54,
                                "y": -57.5,
                                "z": -2,
                                "turn": 0},
                          "4": {"x": -65,
                                "y": -14.5,
                                "z": -2,
                                "turn": 0},
                          "5": {"x": -54.5,
                                "y": 39.5,
                                "z": -2,
                                "turn": 0},
                          "6": {"x": -11,
                                "y": 40.5,
                                "z": -2,
                                "turn": 0},
                          "7": {"x": 0,
                                "y": 0,
                                "z": -2,
                                "turn": 0},
                          }
        return

    def get_distance_from_goal(self, state):
        return math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2)

    def get_reward(self, state, goal, collision, age, max_age):
        distance_from_goal = self.get_distance_from_goal(state)

        if collision is True:
            # --> Negative reward if a collision occurred (not included for now)
            reward = -200
            # reward = -(math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2))

        else:
            if distance_from_goal < 2:
                reward = 300

            elif age >= max_age:
                reward = -100

            else:
                reward = -(math.sqrt((state[0])**2 + (state[1])**2 + (state[2])**2))

        return reward

    def check_if_done(self, state, goal, collision, age, max_age):
        distance_from_goal = self.get_distance_from_goal(state)

        if age >= max_age:
            # --> End run if max age reached
            return True, goal, age

        if collision is True:
            # --> End run if collision occurred
            return True, goal, age

        # --> Check x position
        if distance_from_goal < 2:
            return True, goal, age

        return False, goal, age

    def flip_track(self):
        if self.direction == 2:
            self.goal_dict = {"0": {"x": 0,
                                    "y": -28.5,
                                    "z": -2,
                                    "turn": 0},
                              "1": {"x": -6,
                                    "y": -53.5,
                                    "z": -2,
                                    "turn": 0},
                              "2": {"x": -54,
                                    "y": -57.5,
                                    "z": -2,
                                    "turn": 0},
                              "3": {"x": -65,
                                    "y": -14.5,
                                    "z": -2,
                                    "turn": 0},
                              "4": {"x": -53.5,
                                    "y": 28.5,
                                    "z": -2,
                                    "turn": 0},
                              "5": {"x": -12,
                                    "y": 32.5,
                                    "z": -2,
                                    "turn": 0},
                              "6": {"x": 0,
                                    "y": 0,
                                    "z": -2,
                                    "turn": 0},
                              }
            self.direction = 1

        else:
            self.goal_dict = {"6": {"x": 0,
                                    "y": 0,
                                    "z": -2,
                                    "turn": 0},
                              "5": {"x": 0,
                                    "y": -28.5,
                                    "z": -2,
                                    "turn": 0},
                              "4": {"x": -6,
                                    "y": -53.5,
                                    "z": -2,
                                    "turn": 0},
                              "3": {"x": -54,
                                    "y": -57.5,
                                    "z": -2,
                                    "turn": 0},
                              "2": {"x": -65,
                                    "y": -14.5,
                                    "z": -2,
                                    "turn": 0},
                              "1": {"x": -53.5,
                                    "y": 28.5,
                                    "z": -2,
                                    "turn": 0},
                              "0": {"x": -12,
                                    "y": 32.5,
                                    "z": -2,
                                    "turn": 0},

                              }
            self.direction = 2

        return