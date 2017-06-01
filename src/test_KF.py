import cv2
print('Imported OpenCV version {0}'.format(cv2.__version__))
import sys, os
from contextlib import contextmanager
import time, glob
from copy import deepcopy
from os.path import exists
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from matplotlib import gridspec
import matplotlib.patches as patches
from datetime import datetime
from datetime import timedelta
import numpy as np
import scipy
import itertools
from scipy import stats
from scipy import odr
from scipy.signal import argrelextrema
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
import pandas as pd
import imutils


"""
This script tests the KF which will be used for estimating the background noise mean
"""


def main():

    num_meas = 100

    # one dimensional
    # a sin wave
    t_a = np.sin(np.linspace(0,3.14,num_meas))
    t_a = np.linspace(0, 1,num_meas)
    # t_a = np.random.normal(0.0,2.0,10000)

    # noise data
    _mu, n_sigma = 0.0, 0.05
    t_n = np.random.normal(_mu, n_sigma, num_meas)

    # temperature data
    t = t_a + t_n

    # =================================================
    # use Kalman filter to track the mean
    # kf = KF(1, 0.01**2, 0.5**2)
    # kf.initialize_states(0.0, 0.2**2)

    # all_mu = []
    # t1 = datetime.now()
    # for i in range(0, 10000):
    #     mu = kf.update_state(t[i])
    #     # print mu.squeeze()
    #     all_mu.append(mu)
    # t2 = datetime.now()
    # print('KF time: {0} s'.format((t2-t1).total_seconds()))

    # =================================================
    # use 1d Kalman filter to track the mean
    kf = KF_1d(1, 0.005**2, 0.05**2)
    kf.initialize_states(0.0, 0.01**2)

    all_mu = []
    t1 = datetime.now()
    for i in range(0, num_meas):
        mu = kf.update_state(t[i])
        # print mu.squeeze()
        all_mu.append(mu)
    t2 = datetime.now()
    print('KF_1d time: {0} s'.format((t2-t1).total_seconds()))

    # plot the result
    all_mu = np.array(all_mu).squeeze()
    plt.plot(t, color = 'b', linestyle='-', linewidth=2, label='measurement')
    plt.plot(t_a, color = 'r', linewidth=2, label='true')
    plt.plot(all_mu, color='g',linewidth=2, label='estimates')
    plt.legend(loc=2, fontsize=18)
    plt.ylabel('End of queue', fontsize=18)
    plt.xlabel('Time steps', fontsize=18)
    plt.show()
    #
    # # two dimensional
    # # a sin wave
    # t_a1 = np.sin(np.linspace(0,3.14*4,10000))
    # t_a2 = np.cos(np.linspace(0,3.14*4,10000))
    # # t_a = np.random.normal(0.0,2.0,10000)
    #
    # # noise data
    # _mu, n_sigma = 0.0, 0.5
    # t_n = np.random.normal(_mu, n_sigma, 10000)
    #
    # # temperature data
    # t_1 = t_a1 + t_n
    # t_2 = t_a2 + t_n
    #
    # t = np.vstack([t_1, t_2]).T
    #
    # # use Kalman filter to track the mean
    # Q = np.array([[0.005**2, 0.0], [0.0, 0.005**2]])
    # R = np.array([[0.5**2, 0.0], [0.0, 0.5**2]])
    # kf = KF(2, Q, R)
    #
    # x0 = np.array([0.0, 0.0])
    # P0 = np.array([[0.3**2, 0.0], [0.0, 0.3**2]])
    # kf.initialize_states(x0, P0)
    #
    # all_mu = []
    # for i in range(0, 10000):
    #     mu = kf.update_state(t[i])
    #     all_mu.append(mu)
    #
    # # plot the result
    # all_mu = np.array(all_mu).squeeze()
    #
    # # plot dim 1
    # plt.figure()
    # plt.plot(t_1, 'b')
    # plt.plot(all_mu[:,0], color='g',linewidth=2)
    # plt.title('State 1')
    #
    # plt.figure()
    # plt.plot(t_2, 'b')
    # plt.plot(all_mu[:,1], color='g',linewidth=2)
    # plt.title('State 2')
    #
    # plt.show()



class KF:
    """
    The system:
        x(k) = Ix(k-1) + w, where w ~ N(0, Q)
        z(k) = Ix(k) + v, where v ~ N(0, R)
    """

    def __init__(self, dim_state, Q, R):
        """
        Initialize the kalman filter,
        :param dim_state: the dimension of the state
        :param Q: the error covariance matrix of the model noise
        :param R: the error covariance matrix of the measurement noise
        :return:
        """
        self.dim = dim_state
        if type(Q) is float:
            self.Q = Q*np.ones((1,1))
        else:
            self.Q = Q

        if type(R) is float:
            self.R = R*np.ones((1,1))
        else:
            self.R = R

        # states. preallocate memory
        self.x = np.zeros(self.dim)
        self.P = np.zeros((self.dim, self.dim))

        # the predicted state and error
        self.x_f = np.zeros(self.dim)
        self.P_f = np.zeros((self.dim, self.dim))

    def initialize_states(self, x0, P0):
        """
        This function initializes the initial state
        :param x0: the initial state
        :param P0: the initial error covariance matrix
        :return: initialized into property
        """
        self.x = deepcopy(x0)
        self.P = deepcopy(P0)

    def update_state(self, z):
        """
        This function updates the current state given measurement z
        :param z: np array
        :return: the updated system state
        """

        # forward propagate the state
        self.x_f = deepcopy(self.x)
        self.P_f = self.P + self.Q

        # compute the innovation sequence and Kalman gain
        y = z - self.x_f
        S = self.P_f + self.R

        K = np.dot(self.P_f, np.linalg.inv(S))

        # update the state
        self.x = self.x_f + np.dot(K, y)
        self.P = np.dot(np.diag(np.ones(self.dim))-K ,self.P_f)

        return self.x


    def update_state_sequence(self, zs):
        """
        This function updates the current state given a sequence of measurements zs
        :param zs: a sequence of measurement z, num_meas x dim
        :return: the current state
        """

        for z in zs:
            self.update_state(z)

        return self.x




class KF_1d:
    """
    The system is a one-dimentional KF:
        x(k) = Ix(k-1) + w, where w ~ N(0, Q)
        z(k) = Ix(k) + v, where v ~ N(0, R)
    Update:
    The system is a one-dimentional KF:
        x(k) = x(k-1) + 1.1 + w, where w ~ N(0, Q)
        z(k) = x(k) + v, where v ~ N(0, R)
    """

    def __init__(self, dim_state, Q, R):
        """
        Initialize the kalman filter,
        :param dim_state: the dimension of the state
        :param Q: the error covariance matrix of the model noise
        :param R: the error covariance matrix of the measurement noise
        :return:
        """
        self.dim = dim_state
        self.Q = Q
        self.R = R

        # states. preallocate memory
        self.x = 0.0
        self.P = 0.0

        # the predicted state and error
        self.x_f = 0.0
        self.P_f = 0.0

    def initialize_states(self, x0, P0):
        """
        This function initializes the initial state
        :param x0: the initial state
        :param P0: the initial error covariance matrix
        :return: initialized into property
        """
        self.x = x0
        self.P = P0

    def update_state(self, z):
        """
        This function updates the current state given measurement z
        :param z: np array
        :return: the updated system state
        """

        # forward propagate the state
        self.x_f = self.x + 0.011
        self.P_f = self.P + self.Q

        # compute the innovation sequence and Kalman gain
        y = z - self.x_f
        S = self.P_f + self.R
        K = self.P_f/S

        # update the state
        self.x = self.x_f + K*y
        self.P = (1-K)*self.P_f

        return self.x


    def update_state_sequence(self, zs):
        """
        This function updates the current state given a sequence of measurements zs
        :param zs: a sequence of measurement z, num_meas x dim
        :return: the current state
        """

        for z in zs:
            self.update_state(z)

        return self.x




if __name__ == '__main__':
    main()











