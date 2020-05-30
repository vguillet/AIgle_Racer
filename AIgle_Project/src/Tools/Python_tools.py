
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import random
from random import random
from collections import deque
import pickle

# Libs
import numpy as np

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def weighted_choice(objects, weights, batch_size=1):
    """ returns randomly an element from the sequence of 'objects',
        the likelihood of the objects is weighted according
        to the sequence of 'weights', i.e. percentages."""

    # assert(isinstance(objects, list))
    # assert(isinstance(weights, list))
    selection = []

    # print(weights)

    for i in range(batch_size):

        current_weights = np.array(weights[i], dtype=np.float64)
        sum_of_weights = current_weights.sum()

        # standardization:
        np.multiply(current_weights, 1 / sum_of_weights, current_weights)
        current_weights = current_weights.cumsum()

        x = random()
        for j in range(len(current_weights)):
            if x < current_weights[j]:
                selection.append(objects[j])

    if batch_size == 1:
        return selection[0]

    else:
        return selection