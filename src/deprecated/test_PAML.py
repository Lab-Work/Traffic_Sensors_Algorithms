__author__='Yanning Li'
"""
This script implement and test the Projection And Mixture model Linear regression (PAML).
The basic steps are:
- Project the points to a 1d along all possible directions.
    : have to try all degrees and hard to tell which is the right projection (i.e., clustering is best)
- Use Bayesian Gaussian Mixture model to fit the projected data and identify potential clusters.
    : Tried density kernel which however is sensitive to the bandwidth. Can come up with nice segmentation, but hard
      to compare with different projections in each direction (note, min(std) wound work due to no one-to-one label of
      the cluster and the noise)
- Refine each cluster by linear regression to get the finalized speed.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from matplotlib import gridspec
from datetime import datetime
from datetime import timedelta
from os.path import exists
import os
from collections import OrderedDict
import sys
import time
import glob
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import random
import scipy
import statsmodels.api as sm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from sklearn.decomposition import FastICA, PCA

def main():
    # ==========================================
    # select the dataset for testing
    # dataset = '20161013_213309_727757'  # nice clean single vehicle
    # dataset = '20161013_213921_970537'  # nice clean four vehicles
    # dataset = '20161013_213458_177875'  # two vehicles: different signal variance
    # dataset = '20161013_213301_405012'   # five vehicles with two overlapping
    #
    dataset = '20161013_213255_441458' # three: two overlapping, one weak signal
    # dataset = '20161013_213952_192220' # four: very noisy upper half and weak signal, opposite direction

    # dataset = '20161013_213325_428855' # two vehicle, opposite direction
    # dataset = '20161013_213526_373052' # three vehicles: one opposite direction,
    #                                                                 # different signal variance
    # dataset = '20161013_213313_429930' # no vehicle


    # ==========================================
    # configuration
    save_dir = '../figs/test_PAML/{0}/'.format(dataset)

    if not exists(save_dir):
        os.makedirs(save_dir)

    sample_veh = np.load('../workspace/{0}.npy'.format(dataset))
    sample_veh = sample_veh[()]

    # get the space grid
    # 120 degree field of view
    ratio = 2.0
    d = 3.5 # m
    speed_ratio = ratio*d*2.24
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


    # ==========================================
    # run the algorithm
    estimator = PAML(time=data[:,0], space=data[:,1], plot=True, save_dir=save_dir)
    estimator.estimate_speed(speed_range=(1,60,3), speed_ratio=speed_ratio,
                             max_components=3, concentration_prior=0.0000000001, alg='SegGMM')
    # estimator.ica_proj(data[:,0:2])


class PAML:
    """
    The class for Projection And Mixture model Linear regression algorithm.
    """
    def __init__(self, time=None, space=None, plot=False, save_dir=''):
        """
        Initialize with data
        :param time: 1d array, s
        :param space: 1d array, m
        :param plot: True or False, plot figures in progress
        :param save_dir: save the figures and result in this dir
        :return:
        """
        self.time = time
        self.space = space
        self.plot = plot
        self.save_dir = save_dir


    def estimate_speed(self, speed_range=(1, 70, 1), speed_ratio=3.5*6*2.24,
                       max_components=5, concentration_prior=0.001, alg='SegGMM'):

        # speed range to project
        speeds = np.arange(speed_range[0], speed_range[1], speed_range[2]).astype(float)
        slopes = -speeds/speed_ratio      # one direction

        best_mdls = []   # (mean, std, weight, k)
        num_clusters = []
        for counter, k in enumerate(slopes):
            print('projecting direction {0}'.format(counter))
            # project to each direction
            dists = self.compute_dist(self.time, self.space, (k, 0.0))

            if alg == 'VBGM':
                # run the Bayesian Gaussian mixture model: (mean, std, weight)
                mdls = self.fit_mixture_model_VBGM(dists,max_components=max_components,
                                              concentration_prior=concentration_prior)
            elif alg == 'SegGMM':
                x_ticks = np.linspace(0, np.sqrt(np.max(self.time)**2 + np.max(self.space)**2), 500)
                mdls, logs_dens, segs = self.fit_mixture_model_SegGMM(dists, x_ticks, k,
                                                                      min_headway=0.2, min_num_pts=100)
                num_clusters.append(mdls.shape[0])
            if alg == 'GMM':
                mdls, aic, bic = self.fit_mixture_model_GMM(dists, None, num_component=max_components)

            # determine if should update the best distribution or if should divide and add more
            # assuming one to one correspondence
            for i, mdl in enumerate(mdls):
                try:
                    if mdl[1] <= best_mdls[i][1]:
                        # smaller standard deviation hence better fitting
                        best_mdls[i] = np.concatenate([mdl, [k]])
                except IndexError:
                    # out of bound, divide and append to best model
                    best_mdls.append( np.concatenate([mdl, [k]]) )

            if self.plot:
                if alg == 'VBGM' or alg == 'GMM':
                    self.plot_progress(k,mdls,best_mdls,save_name='{0}'.format(counter),
                                   title='Project to speed {0} mph (d=3.5m)'.format(-k*speed_ratio),
                                   speed_ratio=speed_ratio)
                elif alg == 'SegGMM':
                    self.plot_progress(k,mdls,best_mdls,save_name='{0}'.format(counter),
                                   title='Project to speed {0} mph (d=3.5m)'.format(-k*speed_ratio),
                                   speed_ratio=speed_ratio, x_ticks=x_ticks, kernel_density=logs_dens,
                                       segmentation=segs)

        # check the number of clusters
        plt.figure()
        plt.plot(speeds, num_clusters, linewidth=2, marker='*')
        plt.title('Change of number of clusters while rotation', fontsize=18)
        plt.xlabel('speeds (mph)', fontsize=16)
        plt.grid(True)
        plt.show()


        # run linear regression in each distribution


    def plot_progress(self, k, mdls, best_mdls, save_name=None, title='', speed_ratio=3.5*6*2.24,
                      x_ticks=None, kernel_density=None, segmentation=None):

        # plot the initial figure
        fig = plt.figure(figsize=(13,18))
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])

        # data, mdl, and best mdl
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])

        # ----------------------------------------------------
        # ax0: the original data points
        ax0.scatter(self.time, self.space, color='0.6')

        # plot the line and std for the mdl fitting
        x_limit = np.asarray([np.min(self.time), np.max(self.time)])

        for mdl in mdls:
            # plot the mean line
            y = k*x_limit + mdl[0]*np.sqrt(1+k**2)
            ax0.plot(x_limit, y, linewidth=2, color='b')

            # plot the one sigma line
            y = k*x_limit + (mdl[0]-mdl[1])*np.sqrt(1+k**2)
            ax0.plot(x_limit, y, linewidth=2, color='r', linestyle='--')
            y = k*x_limit + (mdl[0]+mdl[1])*np.sqrt(1+k**2)
            ax0.plot(x_limit, y, linewidth=2, color='r', linestyle='--')

        # plot the best_mdls points in green
        for b_mdl in best_mdls:
            # get the mean line
            c = b_mdl[0]*np.sqrt(1+b_mdl[3]**2)
            dists = self.compute_dist(self.time, self.space, (b_mdl[3], c))
            inlier_idx = (np.abs(dists) <= b_mdl[1])
            ax0.scatter(self.time[inlier_idx], self.space[inlier_idx], color='g')

        ax0.set_title('{0}'.format(title), fontsize=20)
        ax0.set_xlabel('Time (s)', fontsize=18)
        ax0.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax0.set_xlim(x_limit)
        ax0.set_ylim([np.min(self.space), np.max(self.space)])

        # ----------------------------------------------------
        # ax1: the distribution of latest mdls
        # - the distribution of the distance of points to the line intercept at origin with slope k
        # - the measures: total num_points, slope

        # compute the signed distance of all points
        dists = self.compute_dist(self.time, self.space, (k, 0.0))

        if x_ticks is None:
            dist_lim = [0, np.sqrt(np.max(self.time)**2 + np.max(self.space)**2)]
        else:
            dist_lim = [0, x_ticks[-1]]
        # plot histogram
        bin_width = 0.01
        n, bins, patches = ax1.hist(dists, bins=np.arange(dist_lim[0], dist_lim[1],bin_width),
                                        normed=1, facecolor='green', alpha=0.75)

        # plot the density kernel function
        if kernel_density is not None:
            ax1.plot(x_ticks, np.exp(kernel_density), '--', linewidth=3, color='m')

        # plot the segmentation
        if segmentation is not None:
            for s in segmentation:
                ax1.axvline(s, linestyle='-', linewidth=2, color='k')

        # plot the mixture model
        for mdl in mdls:
            if kernel_density is not None:
                # normalize the density
                ax1.plot(bins, mlab.normpdf(bins, mdl[0], mdl[1])*mdl[2], 'b')
            else:
                ax1.plot(bins, mlab.normpdf(bins, mdl[0], mdl[1]), 'b')
            # mark the current tol threshold
            sig_1_l = mdl[0]-mdl[1]
            sig_1_u = mdl[0]+mdl[1]
            if kernel_density is not None:
                sig_1_y = mlab.normpdf(sig_1_l, mdl[0], mdl[1])*mdl[2]
            else:
                sig_1_y = mlab.normpdf(sig_1_l, mdl[0], mdl[1])

            ax1.plot([sig_1_l, sig_1_l], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')
            ax1.plot([sig_1_u, sig_1_u], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')

            text = ' std: {0:.04f}\n speed: {1} mph'.format(mdl[1], -k*speed_ratio)
            ax1.annotate(text, xy=(mdl[0]-mdl[1], np.max(n)*1.2), fontsize=14)

        ax1.set_ylim([0, np.max(n)*1.5])
        ax1.set_xlim(dist_lim)
        # ax1.set_xlabel('Normalized distance', fontsize=16)
        ax1.set_title('Distribution of distances to current lines',fontsize=18)

        # ----------------------------------------------------
        # ax2: the best distributions
        # - the distribution of the distance of points to the line intercept at origin with slope k
        # - the measures: total num_points, slope

        # get the distances that fits each best model
        all_dists = np.array([])
        for b_mdl in best_mdls:
            c = b_mdl[0]*np.sqrt(1+b_mdl[3]**2)
            dists = self.compute_dist(self.time, self.space, (b_mdl[3], c))
            inlier_idx = (np.abs(dists) <= b_mdl[1])

            # compute the distance of those points to the line across origin
            dists = self.compute_dist(self.time[inlier_idx], self.space[inlier_idx], (b_mdl[3], 0))
            all_dists = np.concatenate([all_dists, dists])


        # plot histogram only for those inliers
        n, bins, patches = ax2.hist(all_dists, bins=np.arange(dist_lim[0], dist_lim[1],bin_width),
                                            normed=1, facecolor='green', alpha=0.75)
        for b_mdl in best_mdls:
            ax2.plot(bins, mlab.normpdf(bins, b_mdl[0], b_mdl[1]), 'b')

            sig_1_l = b_mdl[0]-b_mdl[1]
            sig_1_u = b_mdl[0]+b_mdl[1]
            sig_1_y = mlab.normpdf(b_mdl[0]-b_mdl[1], b_mdl[0], b_mdl[1])
            ax2.plot([sig_1_l, sig_1_l], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')
            ax2.plot([sig_1_u, sig_1_u], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')

            text = ' std: {0:.04f}\n speed: {1} mph'.format(b_mdl[1], -b_mdl[3]*speed_ratio)
            ax2.annotate(text, xy=(b_mdl[0]-b_mdl[1], np.max(n)*1.2), fontsize=14)

        ax2.set_xlim(dist_lim)
        ax2.set_ylim([0, np.max(n)*1.5])
        # ax2.set_xlabel('Normalized distance', fontsize=16)
        ax2.set_title('Distribution of distances to best lines', fontsize=18)

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()


    @staticmethod
    def compute_dist(pts_t, pts_s, line):
        """
        This function computes the distance of the points to the line mdl
        :param pts_t: time, t
        :param pts_s: space, m
        :param mdl: (k, c); space = k*time + c
        :return: a list of dists; NOT absolute value
        """
        pts_t = np.asarray(pts_t)
        pts_s = np.asarray(pts_s)
        k, c = line
        dist = -(pts_t*k-pts_s+c)/np.sqrt(1+k**2)

        return dist

    @staticmethod
    def fit_mixture_model_VBGM(dists, max_components=5, concentration_prior=0.2):

        mix_mdl = BayesianGaussianMixture(covariance_type='diag', n_components=max_components,
                                          weight_concentration_prior_type='dirichlet_distribution',
                                          weight_concentration_prior=concentration_prior)
        mix_mdl.fit(dists[:,np.newaxis])

        # return a list of models, each has (mean, std, weight)
        print('fitted model:')
        print(np.array(zip(np.squeeze(mix_mdl.means_), np.squeeze(np.sqrt(mix_mdl.covariances_)), mix_mdl.weights_)))
        print('unique labels:{0}\n'.format(np.unique(mix_mdl.predict(dists[:,np.newaxis]))))
        return np.array(zip(np.squeeze(mix_mdl.means_), np.squeeze(np.sqrt(mix_mdl.covariances_)), mix_mdl.weights_))


    def fit_mixture_model_SegGMM(self, dists, x_ticks, k, min_headway=0.2, min_num_pts=100):

        # fit the gaussian kernel
        kde = KernelDensity(bandwidth=0.02, kernel='gaussian').fit(dists[:,np.newaxis])

        # find the minimum point
        log_dens = kde.score_samples(x_ticks[:,np.newaxis])

        # find the local minimums
        x_minimas_idx = argrelextrema(log_dens, np.less)[0]

        # remove the local minimum if the two maximas are <min_headway away.
        # x_maximas_idx = argrelextrema(log_dens, np.greater)[0]
        # x_locs_idx = np.sort(np.concatenate([x_minimas_idx, x_maximas_idx]))
        # tmp_mins = []
        # for i, x_min_idx in enumerate(x_minimas_idx):
        #     if i==0 or i== len(x_locs_idx)-1:
        #         # if first or last are local minimas
        #         tmp_mins.append(x_min_idx)
        #         continue
        #
        #     last_max_idx = x_locs_idx[i-1]
        #     next_max_idx = x_locs_idx[i+1]
        #     d_dist = x_ticks[next_max_idx] - x_ticks[last_max_idx]
        #     dt = -d_dist*np.sqrt(1+k**2)/k
        #
        #     if dt > min_headway:
        #         # append this minimum only if two maximums around are separate enough.
        #         tmp_mins.append(x_min_idx)
        #     else:
        #         print('-- skipping minimum {0} with headway {1}'.format(x_ticks[x_min_idx], dt))
        #
        # x_minimas_idx = tmp_mins

        x_segs = zip(np.concatenate([[0], x_minimas_idx]), np.concatenate([x_minimas_idx, [len(x_ticks)-1]]))

        # fit one Gaussian to each segment.
        means = []
        stds = []
        weights = []    # this weight is actually the portion percent
        aic = []
        bic = []
        for seg_s, seg_e in x_segs:
            seg_data_idx = (dists >= x_ticks[seg_s]) & (dists < x_ticks[seg_e])

            if sum(seg_data_idx) <= min_num_pts:
                # mini cluster, skip
                continue

            seg_data = dists[seg_data_idx]

            gmm = GaussianMixture()
            r = gmm.fit(seg_data[:,np.newaxis])
            means.append(r.means_[0,0])
            stds.append(np.sqrt(r.covariances_[0,0]))
            weights.append(float(len(seg_data))/len(dists))
            aic.append(gmm.aic(seg_data[:,np.newaxis]))
            bic.append(gmm.bic(seg_data[:,np.newaxis]))

        # return fitted Gaussian and segments
        np.set_printoptions(linewidth=125)
        print('fitted model')
        print(np.array(zip(means, stds, weights, aic, bic)))
        return np.array(zip(means, stds, weights)), log_dens, x_ticks[x_minimas_idx]


    def fit_mixture_model_GMM(self, dists, x_ticks, num_component=1):

        # fit the GMM
        gmm = GaussianMixture(n_components=num_component)
        r = gmm.fit(dists[:,np.newaxis])

        # return a list of models, each has (mean, std, weight)
        print('fitted model: aic {0}, bic {1}'.format(gmm.aic(dists[:,np.newaxis]), gmm.bic(dists[:,np.newaxis])))
        print(np.array(zip(np.squeeze(r.means_), np.squeeze(np.sqrt(r.covariances_)), r.weights_)))
        print('unique labels:{0}\n'.format(np.unique(gmm.predict(dists[:,np.newaxis]))))
        return np.array(zip(np.squeeze(r.means_), np.squeeze(np.sqrt(r.covariances_)), r.weights_)), \
               gmm.aic(dists[:,np.newaxis]), gmm.bic(dists[:,np.newaxis])


    def ica_proj(self, X):

        # whiten signal
        means = np.mean(X,0)
        X = X-means

        pca = PCA()
        S_pca_ = pca.fit(X).transform(X)

        ica = FastICA()
        S_ica_ = ica.fit(X).transform(X)  # Estimate the sources
        S_ica_ /= S_ica_.std(axis=0)

        plt.figure()
        axis_list = [pca.components_.T, ica.mixing_]
        self.plot_samples(X / np.std(X), axis_list=axis_list)
        legend = plt.legend(['PCA', 'ICA'], loc='upper right')
        legend.set_zorder(100)
        plt.title('Observations')

        # plt.subplot(2, 2, 3)
        # plot_samples(S_pca_ / np.std(S_pca_, axis=0))
        # plt.title('PCA recovered signals')
        #
        # plt.subplot(2, 2, 4)
        # plot_samples(S_ica_ / np.std(S_ica_))
        # plt.title('ICA recovered signals')

        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
        plt.show()

    @staticmethod
    def plot_samples(S, axis_list=None):

        plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                    color='steelblue', alpha=0.5)
        if axis_list is not None:
            colors = ['orange', 'red']
            for color, axis in zip(colors, axis_list):
                axis /= axis.std()
                x_axis, y_axis = axis
                # Trick to get legend to work
                plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
                plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                           color=color)

        plt.hlines(0, -3, 3)
        plt.vlines(0, -3, 3)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('x')
        plt.ylabel('y')



if __name__=='__main__':
    main()