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




def main():

    # a sin wave
    t_a = np.sin(np.linspace(0,3.14*4,10000))
    # t_a = np.random.normal(0.0,2.0,10000)

    # noise data
    _mu, n_sigma = 0.0, 0.5
    t_n = np.random.normal(_mu, n_sigma, 10000)

    # temperature data
    t = t_a + t_n

    # assume background piror
    hyper_mu = 0.0
    hyper_sig = 1000.0

    # update the ambient temperature every 100 data points
    all_mu = []
    n_mu = 0.0
    for i in range(1, 100):

        all_mu.append([(i-1)*100, n_mu])

        data = t[(i-1)*100:i*100]
        (post_n_mu, post_n_sig), (post_hyper_mu, post_hyper_sig) = _MAP_update(data, (n_mu, n_sigma), (hyper_mu, hyper_sig))

        hyper_mu = post_hyper_mu
        hyper_sig = post_hyper_sig
        n_mu = post_n_mu
        # n_sigma = post_n_sig

    # plot the result
    all_mu = np.array(all_mu)
    plt.plot(t, 'b')
    plt.plot(all_mu[:,0], all_mu[:,1],'g',linewidth=2)
    plt.show()


# @profile
def _MAP_update(data, paras, prior):
    """
    This function updates the noise distribution and the prior of the noise mean distribution
    :param data: the noise data point
    :param paras: (mu, sigma) the current noise distribution; Note, sigma is known
    :param prior: (prior_mu, prior_sigma) the prior distribution of mu
    :return: posterior (mu, sigma), posterior (prior_mu, prior_sigma)
    """
    mu, var = paras[0], paras[1]**2
    prior_mu, prior_var = prior[0], prior[1]**2
    if type(data) is int or type(data) is float:
        len_data = 1
    else:
        len_data = len(data)

    # self._debug_MAP_data_len.append(len_data)

    post_var = 1.0/(1.0/prior_var + len_data/var)
    data_sum = np.sum(data)
    post_mu = post_var*(data_sum/var + prior_mu/prior_var)

    return (post_mu, np.sqrt( var +  post_var)), (post_mu, np.sqrt(post_var))






if __name__ == '__main__':
    main()
