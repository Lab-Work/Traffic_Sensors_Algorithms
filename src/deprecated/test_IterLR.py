__author_='Yanning Li'
"""
This script contains the code for testing iterative robust linear regression.
"""



import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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
import random


def main():

    # load the data
    sample_veh = np.load('../workspace/20161013_213301_405012.npy')
    sample_veh = sample_veh[()]

    # get the space grid
    # 120 degree field of view
    ratio = 6.0
    _dup = sample_veh['data'].shape[0]/32
    d_theta = (60.0/16)*np.pi/180.0
    spaces = []
    for i in range(-16, 16):
        for d in range(0, _dup):
            # duplicate the nonlinear operator for vec
            spaces.append( np.tan(d_theta/2 + i*d_theta) )
    spaces = -np.asarray(spaces)/ratio

    # get the time grid
    _dt = sample_veh['time'] - sample_veh['time'][0]
    times = np.asarray([i.total_seconds() for i in _dt])

    # remove nan values
    data = []
    num_row, num_col = sample_veh['data'].shape
    for row in range(0, num_row):
        for col in range(0, num_col):
            if ~np.isnan(sample_veh['data'][row, col]):
                data.append([times[col], spaces[row]])
    data = np.asarray(data)

    print('Removed nan and got {0}'.format(data.shape))

    # plot the initial figure
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(data[:,0], data[:,1])
    ax.set_title('Initial', fontsize=20)
    plt.savefig('{0}.png'.format(0), bbox_inches='tight')
    plt.clf()
    plt.close()

    # run and plot the iterative robust linear regression
    regressor = IterRLR(X=data[:,0], y=data[:,1], plot=True)

    regressor.run_RLR(tol=0.03, init_num_inliers=100, stop_tol=(0.001, 0.001))


class IterRLR:
    """
    This is the iterative robust linear regression class.
    - It contains one parameter, tol, which gives the tolerance for selecting inliers
    - It iteratively run linear regression until convergence to a model
    """

    def __init__(self, X=None, y=None, plot=False):
        """
        Initialize with data
        :param X: n_samples x dim_inputs
        :param y: n_samples x 1, assuming only one output
        :return:
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.plot = plot

    def run_RLR(self, tol=0.1, init_num_inliers=100, stop_tol=(0.0001, 0.01)):
        """
        Run the iterative robust regression
        :param tol: tolerance, the pass of each vehicle takes about 0.2 s
        :param init_num_inliers: the initial number of inliers
        :param stop_tol: the stopping tolerance for k and c, where y = kx + c
        :return:
        """
        inlier_idx = random.sample(range(0, len(self.y)), init_num_inliers)

        # fit the initial model
        pre_mdl, res = self.linear_model(data=self.X[inlier_idx], y=self.y[inlier_idx])

        if self.plot is True:
            self.plot_progress(pre_mdl, inlier_idx, save_name='1', tol=tol)

        for i in range(2, 15):

            print('Iteration {0}'.format(i))
            # update inliers
            inlier_idx = self.update_inliers(mdl=pre_mdl, tol=tol)

            # fit the new model
            mdl, res = self.linear_model(data=self.X[inlier_idx], y=self.y[inlier_idx])

            if self.plot is True:
                self.plot_progress(mdl, inlier_idx, save_name=i, tol=tol)

            if all(np.asarray(mdl) - np.asarray(pre_mdl) <= np.asarray(stop_tol)):
                print('Found the optimal fitting: y = {0} x + {1}'.format(mdl[0], mdl[1]))
                break
            else:
                pre_mdl = mdl


    def resample(self, all_idx):
        """
        This function randomly selects max(2, len(all_idx)/4) points
        :param all_idx: a list of values
        :return: unique idx
        """
        all_idx = np.asarray(all_idx)
        num_inliers = len(all_idx)
        _idx = random.sample(range(0, num_inliers), np.max(2, int(num_inliers/4)))

        return all_idx[_idx]


    def plot_progress(self, mdl, inlier_idx, save_name=None, tol=0.01):
        """
        This function plots the progress of the fitting
        :param mdl: (k, c)
        :param inlier_idx: list of integers
        :return:
        """
        # plot the initial figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.X, self.y, c='k')
        ax.scatter(self.X[inlier_idx], self.y[inlier_idx], c='g')
        # plot the line
        x_line = np.asarray([np.min(self.X), np.max(self.X)])
        y_line = mdl[0]*x_line + mdl[1]
        ax.plot(x_line, y_line, linewidth=2, color='b')
        # plot the tolerance
        if mdl[0] != 0:
            c1 = mdl[1] + np.sqrt(tol**2+(tol*mdl[0])**2)
            c2 = mdl[1] - np.sqrt(tol**2+(tol*mdl[0])**2)
        else:
            c1 = mdl[1] + tol
            c2 = mdl[1] - tol

        y_line_1 = mdl[0]*x_line + c1
        ax.plot(x_line, y_line_1, linewidth=2, color='r', linestyle='--')

        y_line_2 = mdl[0]*x_line + c2
        ax.plot(x_line, y_line_2, linewidth=2, color='r', linestyle='--')

        ax.set_title('{0}'.format(save_name), fontsize=20)

        if save_name is not None:
            plt.savefig('{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()


    def linear_model(self, data=None, y=None):
        """
        This function returns the linear model using least squared fitting
        :param data: num_samples x dim_inputs
        :param y: num_samples x 1
        :return:
        """

        # add constant values
        num_samples = data.shape[0]
        X = np.vstack((data, np.ones(num_samples))).T

        coe, res, _, _ = np.linalg.lstsq(X, y)

        return coe, res



    def update_inliers(self, mdl=None, tol=0.1):
        """
        This function updates the inliers, i.e., get data points that lies within tol perpendicular distance to model
        :param mdl: tuple (k, c): y = kx + c
        :param tol: the tolerance for being considered as an inlier
        :return: index list
        """
        k = mdl[0]
        c = mdl[1]

        dist = np.abs(self.X*k - self.y + c)/np.sqrt(1 + k**2)

        idx = (dist <= tol)

        return idx






if __name__=='__main__':
    main()





