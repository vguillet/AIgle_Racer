
################################################################################################################
"""
Contain math tools
"""

# Libs
import numpy as np

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '10/09/2019'

################################################################################################################


class math_tools:
    @staticmethod
    def alignator_minus_one_one(signal, signal_max=100, signal_min=-100):
        signal_normalised = np.zeros(len(signal))

        for i in range(len(signal)):
            signal_normalised[i] = 2*(signal[i] - signal_min) / ((signal_max - signal_min) or 1)-1

        for i in range(len(signal_normalised)):
            if signal_normalised[i] > 1:
                signal_normalised[i] = 1

            elif signal_normalised[i] < -1:
                signal_normalised[i] = -1

        return signal_normalised

    @staticmethod
    def normalise_zero_one(signal):
        signal_normalised = np.zeros(len(signal))

        signal_min = min(signal)
        signal_max = max(signal)
        
        for i in range(len(signal)):
            signal_normalised[i] = (signal[i]-signal_min)/((signal_max-signal_min) or 1)
        
        return signal_normalised
    
    @staticmethod
    def normalise_minus_one_one(signal):
        signal_normalised = np.zeros(len(signal))

        signal_min = min(signal)
        signal_max = max(signal)

        for i in range(len(signal)):
            signal_normalised[i] = 2*(signal[i] - signal_min) / ((signal_max - signal_min) or 1)-1

        return signal_normalised

    @staticmethod
    def normalise_minus_x_x(signal, x):
        signal_normalised = np.zeros(len(signal))

        signal_min = min(signal)
        signal_max = max(signal)

        for i in range(len(signal)):
            signal_normalised[i] = x*(signal[i] - signal_min) / ((signal_max - signal_min) or 1)-x/2

        return signal_normalised

    @staticmethod
    def amplify(signal, amplification_factor):

        signal_amplified = np.zeros(len(signal))

        for i in range(len(signal)):
            signal_amplified[i] = signal[i]*amplification_factor

        return signal_amplified

    @staticmethod
    def best_fit(X, Y):
        xbar = sum(X) / len(X)
        ybar = sum(Y) / len(Y)
        n = len(X)  # or len(Y)

        numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
        denum = sum([xi ** 2 for xi in X]) - n * xbar ** 2

        b = numer / denum
        a = ybar - b * xbar

        # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

        return a, b
