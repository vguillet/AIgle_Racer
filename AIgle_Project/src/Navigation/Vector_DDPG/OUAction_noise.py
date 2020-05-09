
##################################################################################################################
"""
Used to generate Ornstein Uhlenbeck noise that models temporal correlations of brownian motion (centered around 0)
Used in DDPG to introduce noise and handle exploration
"""

# Built-in/Generic Imports

# Libs
import numpy as np

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class OUAction_noise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma= sigma
        self.dt = dt
        self. x0 = x0

        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
