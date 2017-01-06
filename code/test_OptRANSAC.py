__author_='Yanning Li'
"""
This script contains the code for testing Optimal RANSAC
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
from sklearn.cluster import DBSCAN
import random
import scipy


def main():

    # load the data
    # sample_veh = np.load('../workspace/20161013_213309_727757.npy') # nice clean single vehicle
    # sample_veh = np.load('../workspace/20161013_213921_970537.npy') # nice clean four vehicles
    # sample_veh = np.load('../workspace/20161013_213458_177875.npy') # two vehicles: different signal variance
    sample_veh = np.load('../workspace/20161013_213301_405012.npy')   # five vehicles with two overlapping

    # sample_veh = np.load('../workspace/20161013_213255_441458.npy') # three: two overlapping, one weak signal
    # sample_veh = np.load('../workspace/20161013_213952_192220.npy') # four: very noisy upper half and weak signal

    # sample_veh = np.load('../workspace/20161013_213325_428855.npy') # two vehicle, opposite direction
    # sample_veh = np.load('../workspace/20161013_213526_373052.npy') # three vehicles: one opposite direction,
                                                                    # different signal variance
    # sample_veh = np.load('../workspace/20161013_213313_429930.npy') # no vehicle

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
    print('spaces: {0}'.format(spaces))

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
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sc = ax.scatter(data[:,0], data[:,1], color='0.6')
    # ax.set_title('Initial', fontsize=20)
    # plt.savefig('{0}.png'.format(0), bbox_inches='tight')
    # plt.clf()
    # plt.close()

    # run and plot the iterative robust linear regression
    regressor = OptRANSAC(X=data[:,0], y=data[:,1], plot=True)

    # regressor.get_clusters(db_radius=0.05, db_min_size=30)

    regressor.run_optransac(tol=0.1, init_num_inliers=10, stop_tol=(0.002, 0.05))


class OptRANSAC:
    """
    This is the optimal ransac algorithm
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

    def run_optransac(self, tol=0.1, init_num_inliers=100, stop_tol=(0.0001, 0.01)):
        """
        Run the iterative robust regression
        :param tol: threshold for determining inliers, the pass of each vehicle takes about 0.2 s
        :param init_num_inliers: the initial number of inliers
        :param stop_tol: the stopping tolerance for k and c, where y = kx + c
        :return:
        """

        # first use RANSAC to get the clusters
        clusters = self.get_clusters(db_radius=0.05, db_min_size=30)

        if len(clusters) == 0:
            print('Warning: failed to find any cluster using RANSAC')
            return -1
        else:

            for counter, inlier_idx in enumerate(clusters):

                print('\nCluster {0}'.format(counter))

                best_mdl = None
                best_inlier_idx = []

                # fit an initial model with all points
                pre_mdl, res = self.linear_model(data=self.X[inlier_idx], y=self.y[inlier_idx])
                # update the current maybe inliers
                inlier_idx = self.update_inliers(mdl=pre_mdl, tol=tol)

                if self.plot is True:
                    self.plot_progress(pre_mdl, inlier_idx, save_name='cluster_init_{0}'.format(counter), tol=tol)

                # For each cluster, randomly resample 10 times to get the best result;
                # Two points does not work well, they can lead to clustering to the horizontal line
                for i in range(0, 10):

                    print('-- Random subset {0}'.format(i))

                    try_idx = self.resample(inlier_idx)
                    # try_idx = inlier_idx    # try simply use all point and refine it

                    new_mdl, new_inlier_idx = self.rescore(pre_mdl=pre_mdl, inlier_idx=try_idx, tol=tol, stop_tol=stop_tol,
                                                           save_name='{0}_{1}'.format(counter, i))

                    if len(new_inlier_idx) >= len(best_inlier_idx):
                        best_mdl = new_mdl
                        best_inlier_idx = new_inlier_idx
                    else:
                        continue

                print('Best Model for cluster {0} with {1} inliers: y = {2} x + {3}'.format(counter,
                                                                                            len(best_inlier_idx),
                                                                                            best_mdl[0], best_mdl[1]))

                if self.plot is True:
                    self.plot_progress(mdl=best_mdl, inlier_idx=best_inlier_idx, save_name='{0}_best'.format(counter),
                                       title='Speed: {0} mph (d=3.5m)'.format(np.abs(best_mdl[0]*21*2.24)), tol=tol)


    def get_clusters(self, db_radius=0.05, db_min_size=30):
        """
        This function returns a list of candidate clusters using DBSCAN.
        :param db_radius: the radius of a point, 1/64 = 0.015625
        :param db_min_size: the number of point within the radius for a point being considered as core
        :return: [cluster_1, cluster_2], each cluster contains at least two indices (points)
        """
        clusters = []

        samples = np.vstack([self.X, self.y]).T
        y_pre = DBSCAN(eps=db_radius, min_samples=db_min_size).fit_predict(samples)
        num_clusters = len(set(y_pre)) - (1 if -1 in y_pre else 0)
        print('{0} clusters: {1}'.format(num_clusters, np.unique(y_pre)))
        y_pre = np.asarray(y_pre)

        # plot the clustering
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(samples[:,0], samples[:,1], color='0.6')

        colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w']
        for cluster_label in range(0, num_clusters):
            clus = (y_pre == cluster_label)
            ax.scatter(samples[clus,0], samples[clus,1], color=colors[cluster_label])
            clusters.append([i for i,x in enumerate(clus) if x])

        ax.set_title('DBSCAN clustering', fontsize=20)
        plt.savefig('DBSCAN_cluster.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        # return
        return clusters


    def rescore(self, pre_mdl=None, inlier_idx=None, tol=None, stop_tol=None, save_name=None, new_outlier_weight=0):
        """
        Iteratively re-estimate the model until convergence and return the model and number of inliers
        :param pre_mdl: the previous model
        :param inlier_idx: the idx of inliers within the pre_mdl line
        :return:
        """
        _pre_mdl = pre_mdl
        _inlier_idx = inlier_idx

        flag = False
        for i in range(0, 100):

            # fit the new model
            mdl, res = self.linear_model(data=self.X[_inlier_idx], y=self.y[_inlier_idx])

            if self.plot is True:
                if save_name is not None:
                    self.plot_progress(mdl, _inlier_idx, save_name=save_name+'_{0}'.format(i),
                                       title='Speed: {0} mph (d=3.5m)'.format(np.abs(mdl[0]*21*2.24)), tol=tol)
                else:
                    self.plot_progress(mdl, _inlier_idx, save_name=None, tol=tol)


            # check if should stop, at least two iterations
            if all(np.asarray(mdl) - np.asarray(_pre_mdl) <= np.asarray(stop_tol)) and i>1:
                flag = True
            else:
                _pre_mdl = mdl

            # update the inliers
            all_inlier_idx = self.update_inliers(mdl=_pre_mdl, tol=tol)
            new_inlier_idx = self.get_new_inliers(_inlier_idx, all_inlier_idx)

            print('---- Rescore {0} with {1} pts ({2} new): y = {3} x + {4}, r2: {5}'.format(i, len(np.unique(_inlier_idx)),
                                                                                              len(new_inlier_idx),
                                                                                              mdl[0], mdl[1],
                                                                                              res))
            # increase the weight of the new inliers
            _inlier_idx = all_inlier_idx + new_inlier_idx*new_outlier_weight

            if flag is True:
                break


        print('---- Rescore Done with {0} points: y = {1} x + {2}, r2: {3}'.format(len(_inlier_idx),
                                                                                            mdl[0], mdl[1], res))
        return mdl, _inlier_idx


    def get_new_inliers(self, old_inlier_idx, new_inlier_idx):
        """
        This function returns what new inliers has been added to new_inlier_idx
        :param old_inlier_idx: list of int
        :param new_inlier_idx: list of int
        :return: new_inliers = list of int.
        """
        new_inliers = []
        for idx in new_inlier_idx:
            if idx not in old_inlier_idx:
                new_inliers.append(idx)

        return new_inliers


    def resample(self, all_idx):
        """
        This function randomly selects max(2, len(all_idx)/4) points
        :param all_idx: a list of values
        :return: unique idx
        """
        all_idx = np.asarray(all_idx)
        num_inliers = len(all_idx)
        # _idx = random.sample(range(0, num_inliers), np.max([len(all_idx)/10, 2]))
        _idx = random.sample(range(0, num_inliers), 2)

        return all_idx[_idx]


    def plot_progress(self, mdl, inlier_idx, save_name=None, title=None, tol=0.01):
        """
        This function plots the progress of the fitting
        :param mdl: (k, c)
        :param inlier_idx: list of integers
        :return:
        """
        # plot the initial figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.X, self.y, color='0.6')
        ax.scatter(self.X[inlier_idx], self.y[inlier_idx], color='g')
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

        ax.set_title('{0}'.format(title), fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax.set_xlim([np.min(self.X), np.max(self.X)])
        ax.set_ylim([np.min(self.y), np.max(self.y)])

        if save_name is not None:
            plt.savefig('{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()


    def linear_model(self, data=None, y=None):
        """
        This function returns the linear model using least squared fitting
        :param data: num_samples x dim_inputs, the time positions
        :param y: num_samples x 1, the space positions
        :return:
        """

        # add constant values
        # num_samples = data.shape[0]

        # fitting as it is y:space, x:time
        # X = np.vstack((data, np.ones(num_samples))).T
        # coe, res, _, _ = np.linalg.lstsq(X, y)
        # print('========= space LR: {0}', coe)

        # fitting inverse
        # X = np.vstack((y, np.ones(num_samples))).T
        # _coe, _res, _, _ = np.linalg.lstsq(X, data)
        # y = coe[0]*x + coe[1]
        # x = y/coe[0] - coe[1]/coe[0]
        # print('========= time LR: {0}', [1/_coe[0], -_coe[1]/_coe[0]])

        # Use the time fitting one
        # coe = np.array([1/_coe[0], -_coe[1]/_coe[0]])

        # Use scipy
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y,data)

        # express in space = coe*time + intercept
        coe = np.array([1/slope, -intercept/slope])

        return coe, r_value**2



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

        return [i for i,x in enumerate(idx) if x]




if __name__=='__main__':
    main()





