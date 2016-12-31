import numpy as np
import matplotlib
from scipy import stats
matplotlib.use('TkAgg')
import bisect
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from os.path import exists
from collections import OrderedDict
import sys
import time
import glob
from sklearn import mixture
import cv2


"""
This class is used for online vehicle detection.
"""


class TrafficSensorAlg:
    """
    This class contains all the functions needed for detecting vehicles using streaming data.
    """
    def __init__(self, data_source=None, output=None, res=(4,32)):
        """
        Initialize the function with the data source and output options
        :return:
        """

        # The noise distribution
        self.noise_mu = np.empty(res)
        self.noise_sigma = np.empty(res)

        # initialize the background noise
        self.init_noise_dist()

        # perform nonlinear transoform ahead
        self.nonlinear_transform()


    def run_det(self):
        """
        This function activates the online detection algorithm and saves outputs to the corresponding file
        :return:
        """

        while True:

            meas = self.grab_new_data()

            # for each pixel, subtract noise
            norm_meas = self.subtract_noise(meas=meas)

            # update noise distribution, and push to buffer for speed estimation
            self.update_noise(norm_meas=norm_meas, prob_veh=0.5)

            # if buffer ready for speed estimation

            # run hough transform after nonlinear transform

            # run clustering, and return speed; clear buffer data.


    # all below functions are supposed to be private functions
    def grab_new_data(self):
        """
        This function grabs the latest measurements from the data source
        :return:
        """
        return np.empty((4,32))


    def init_noise_dist(self):
        """
        This function initialize the background noise distribution
        :return:
        """
        pass


    def nonlinear_transform(self, fov=120):
        """
        This function performs nonlinear mapping of the pixel direction to distance
        :return:
        """
        pass


    def subtract_noise(self, meas=None):
        """
        This function subtracts the noise mean and normalize by the standard deviation for each pixel
        :param meas: the measurement. np.array, 4x16 or 4x32
        :return: the normalized frame
        """
        return (meas-self.noise_mu)/self.noise_sigma


    def update_noise(self, norm_meas=None, prob_veh=0.05):
        """
        This function classifies the normalized measurement data for each pixel.
        And update the noise if the measurement is noise.
        :param norm_meas: the normalized measurement array
        :param prob_veh: if prob < prob_veh, then the meas is classified as veh
        :return: np.array with the same dimension of norm_meas, true if is veh
        """

        is_veh = mlab.normpdf(norm_meas, 0, 1) <= prob_veh

        pass


    def push_to_buffer(self, norm_meas):
        """
        This function should determine if the measurement should be pushed to buffer.
        :param norm_meas:
        :return:
        """
        pass


    def speed_est(self):
        """
        This function runs the speed estimation on the buffer using clustering in the hough space
        :return:
        """

        # grab
















