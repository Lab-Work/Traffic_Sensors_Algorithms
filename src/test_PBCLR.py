__author__ = 'Yanning Li'
"""
This script implement and test the Projection Based Clustering Linear Regression (PBCLR):
The basic steps are:
- Use DBSCAN to get initial clusters
- For each cluster, iteratively refine the estimate by repeating until convergence
    + inliers => (LR) => line
    + line, all_pts => (Projector + GMM) => sigma, or [mean, sigmas] if split cluster
    + line, sigma => (update inlier) => inliers
"""

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec
from os.path import exists
import os
from sklearn.cluster import DBSCAN
import scipy
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from collections import Counter
import sys
from datetime import datetime


def main():
    # ==========================================
    # select the dataset for testing
    # dataset = '20161013_213309_727757'  # nice clean single vehicle
    # dataset = '20161013_213921_970537'  # nice clean four vehicles
    # dataset = '20161013_213458_177875'  # two vehicles: different signal variance
    dataset = '20161013_213301_405012'  # five vehicles with two overlapping
    #
    # dataset = '20161013_213255_441458'  # three: two overlapping, one weak signal
    # dataset = '20161013_213952_192220' # four: very noisy upper half and weak signal, all opposite direction

    # dataset = '20161013_213325_428855' # two vehicle, opposite direction
    # dataset = '20161013_213526_373052' # three vehicles: one opposite direction,
    #                                                                 # different signal variance
    # dataset = '20161013_213313_429930' # no vehicle


    # ==========================================
    # configuration
    save_dir = '../figs/test_PBCLR/{0}/'.format(dataset)

    if not exists(save_dir):
        os.makedirs(save_dir)

    sample_veh = np.load('../workspace/{0}.npy'.format(dataset))
    sample_veh = sample_veh[()]

    # get the space grid
    # 120 degree field of view
    ratio = 6.0
    d = 3.5  # m
    speed_ratio = ratio * d * 2.24
    _dup = sample_veh['data'].shape[0] / 32
    d_theta = (60.0 / 16) * np.pi / 180.0
    spaces = []
    for i in range(-16, 16):
        for d in range(0, _dup):
            # duplicate the nonlinear operator for vec
            spaces.append(np.tan(d_theta / 2 + i * d_theta))
    spaces = -np.asarray(spaces) / ratio
    # print('spaces: {0}'.format(spaces))

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

    # print('Removed nan and got {0}'.format(data.shape))

    # ==========================================
    print('\n\n\n=====================================================================================================')
    print('================================= Time: '),
    print(datetime.now()),
    print('=================================')
    print('=====================================================================================================')
    # run the algorithm
    estimator = PBCLR(time=data[:, 0], space=data[:, 1], plot=False, save_dir=save_dir)
    # 0.02 slope corresponds to 1 mph error assuming 3.5 m distance
    # 0.05 intercept corresponds to 1 m error assuming 3.5 distance
    estimator.estimate_speed(stop_tol=(0.002, 0.01), speed_ratio=speed_ratio, r2_thres=0.8)

    # plt.show()

class PBCLR:
    """
    The class for Projection Based Clustering Linear Regression (PBCLR)
    """

    def __init__(self, time=None, space=None, plot=False, save_dir=''):
        """
        Initialize with data
        :param time: 1d array, s
        :param space: 1d array, m   # scaled
        :param plot: True or False, plot figures in progress
        :param save_dir: save the figures and result in this dir
        :return:
        """
        self.time = time
        self.space = space
        self.plot = plot
        self.save_dir = save_dir

        # each mdl: (line, sigma, inlier_idx, r2)
        self.all_mdls = []
        self.labeled_pts = np.array([]).astype(int)

    def estimate_speed(self, stop_tol=(0.002, 0.01), speed_ratio=3.5 * 6 * 2.24, r2_thres=0.85, min_num_pts=100):

        # first use RANSAC to get the clusters
        clusters = self.get_clusters(db_radius=0.05, db_min_size=30, min_num_pts=min_num_pts)

        if len(clusters) == 0:
            # Meaning all pts in frame is < min_num_pts, then skip
            print('Warning: failed to find any cluster supporting a vehicle in this frame')
            return -1
        else:
            # for each cluster
            for counter, inlier_idx in enumerate(clusters):
                print('\n===========================================================================')
                print('-- Cluster {0}: {1} pts'.format(counter, len(inlier_idx)))

                self.fit_mixed_mdls(initial_idx=inlier_idx, stop_tol=stop_tol, speed_ratio=speed_ratio,
                                    counter=str(counter), r2_thres=r2_thres, min_num_pts=min_num_pts)

            # check remaining points
            if len(self.time) - len(self.labeled_pts) >= min_num_pts:
                # remaining points may still support one car
                rem_idx = np.arange(0, len(self.time), 1).astype(int)
                # removed labeled points
                rem_idx = np.delete(rem_idx, self.labeled_pts)
                print('\n===========================================================================')
                print('-- Checking remaining points: {0}'.format(len(rem_idx)))
                self.fit_mixed_mdls(initial_idx=rem_idx, stop_tol=stop_tol, speed_ratio=speed_ratio,
                                    counter='rem', r2_thres=r2_thres, min_num_pts=min_num_pts)

            # plot final estimates
            self.plot_progress(cur_mdl=None, save_name='final_est', title='Final speed estimation',
                               sig_ratio=2.0, speed_ratio=speed_ratio)

    def fit_mixed_mdls(self, initial_idx, stop_tol=(0.0001, 0.01),
                       speed_ratio=2.5 * 6 * 2.24, counter='0', r2_thres=0.8,
                       min_num_pts=100, top_n=8, min_init_pts=10):

        mdl = self.fit_mdl(initial_idx, stop_tol=stop_tol,
                           speed_ratio=speed_ratio, counter=str(counter))
        r2 = mdl[3]
        if r2 < r2_thres and len(mdl[2]) >= min_num_pts:
            # a bad fitting but has a lot of points which could potentially be multiple vehicles
            # then split the cluster by projection
            print('\n$$$$ Bad fitting (r2 = {0}) with {1} pts.\n$$$$ Splitting to subclusters...'.format(r2, len(mdl[2])))

            # specify candidate slopes to speed up convergence.
            # They could be the current slope, and the slopes of finalized models
            candidate_slopes = [mdl[0][0]]
            for m in self.all_mdls:
                candidate_slopes.append(m[0][0])

            # (lines, sigmas, weight, aic, bic)
            sub_clusters = self.split_cluster_exhaustive(pt_idx=mdl[2], min_num_pts=min_num_pts,
                                                         speed_range=(-50,50,5),speed_ratio=speed_ratio,
                                                         counter=counter, top_n=top_n,
                                                         candidate_slopes= [mdl[0][0]])
            # sub_clusters = self.split_cluster(mdl[0], mdl[2], min_num_pts=min_num_pts, counter=counter)

            for i, (line, sigma, w, aic, bic) in enumerate(sub_clusters):

                inlier_idx = self.update_inliers(line, tol=sigma)
                print('---- Sub-cluster {0} with {1} pts'.format(i, len(inlier_idx)))

                if len(inlier_idx) < min_init_pts:
                    # requires at least two points to fit a line
                    continue

                sub_mdl = self.fit_mdl(inlier_idx=inlier_idx,
                                       stop_tol=stop_tol, speed_ratio=speed_ratio,
                                       counter='sub{0}_{1}'.format(counter, i))

                # append the converged model only if r2>r2_thres and # pts > min_num_pts
                r2 = sub_mdl[3]
                num_pts = len(sub_mdl[2])
                if r2 >= r2_thres and num_pts >= min_num_pts:
                    self.all_mdls.append(sub_mdl)
                    # register labeled points
                    self.labeled_pts = np.concatenate([self.labeled_pts, sub_mdl[2]])
                    print('$$$$ Good fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))
                else:
                    print('$$$$ Discarding bad fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))

        elif r2 >= r2_thres and len(mdl[2]) >= min_num_pts:
            print('$$$$ Good fitting of {0} pts with r2: {1}'.format(len(mdl[2]), r2))
            # a good fitting with strong data support
            # append either the converged line or the last line
            self.all_mdls.append(mdl)
            # register labeled points
            self.labeled_pts = np.concatenate([self.labeled_pts, mdl[2]])

        else:
            num_pts = len(mdl[2])
            print('$$$$ Discarding bad fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))

    def fit_mdl(self, inlier_idx, stop_tol=(0.0001, 0.01),
                speed_ratio=2.5 * 6 * 2.24, counter='0'):

        _pre_line = None
        converged = False
        print('---------- Fitting a line, iterations: ')
        for i in range(0, 100):
            # sys.stdout.write('\r')
            # sys.stdout.write('..... {0},'.format(i))
            # sys.stdout.flush()
            print('               # {0}: {1} pts'.format(i, len(inlier_idx)))

            # ---------------------------------------------------------------------
            # fit a line
            # Reason: at each loc, data is more gaussian; at each time, spaces are nonliearly stretched.
            _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(self.space[inlier_idx],
                                                                                      self.time[inlier_idx])
            # express in space = coe*time + intercept
            line = np.array([1 / _slope, -_intercept / _slope])
            r2 = _r_value ** 2

            if _pre_line is not None and \
                    (np.asarray(line) - np.asarray(_pre_line) <= np.asarray(stop_tol)).all():
                # converged
                print('---------- Converged: y = {0:.03f}x + {1:.03f}, Speed {2:.01f} mph'.format(line[0], line[1],
                                                                                               -speed_ratio * line[0]))
                converged = True

            # ---------------------------------------------------------------------
            # projection, density kernel segmentation, and GMM
            sig, dists = self.get_sigma(line, pt_idx=inlier_idx)

            # determine whether to expand or contract
            if r2 <= 0.8:
                # bad fit, then expand
                sigma_ratio = 3.0
            else:
                # contract to a good fit
                sigma_ratio = 2.0

            if self.plot is True:
                self.plot_progress((line, sig, inlier_idx, r2), save_name='clus{0}_{1}'.format(counter, i),
                                   title='Cluster {0} round {1}, converged:{2}'.format(counter, i, converged),
                                   sig_ratio=sigma_ratio)

            if not converged:
                # update cluster
                inlier_idx = self.update_inliers(line, tol=sigma_ratio * sig)
                _pre_line = deepcopy(line)
            else:
                return line, sig, inlier_idx, r2

    def get_clusters(self, db_radius=0.05, db_min_size=30, min_num_pts=100):
        """
        This function returns a list of candidate clusters using DBSCAN.
        :param db_radius: the radius of a point, 1/64 = 0.015625
        :param db_min_size: the number of point within the radius for a point being considered as core
        :return: [cluster_1, cluster_2], each cluster contains at least two indices (points)
        """
        clusters = []

        samples = np.vstack([self.time, self.space]).T
        y_pre = DBSCAN(eps=db_radius, min_samples=db_min_size).fit_predict(samples)
        num_clusters = len(set(y_pre)) - (1 if -1 in y_pre else 0)
        y_pre = np.asarray(y_pre)

        # determine if should return outliers as a cluster
        # num_outliers = sum(y_pre == -1)
        # if num_outliers >= min_num_pts:
        #     # consider outliers as a cluster with a positive label
        #     y_pre[y_pre == -1] = num_clusters
        #     num_clusters += 1

        print('{0} clusters:'.format(num_clusters))
        for i in range(0, num_clusters):
            print('-- Cluster {0}: {1} pts'.format(i, sum(y_pre == i)))

        # plot the clustering
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(samples[:, 0], samples[:, 1], color='0.6')

        colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w']
        for cluster_label in range(0, num_clusters):
            clus = (y_pre == cluster_label)
            ax.scatter(samples[clus, 0], samples[clus, 1], color=colors[cluster_label])
            clusters.append([i for i, x in enumerate(clus) if x])

        ax.set_title('DBSCAN clustering', fontsize=20)
        plt.savefig(self.save_dir + 'DBSCAN_cluster.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        # return
        return clusters

    def get_sigma(self, line, pt_idx=None):

        if pt_idx is None:
            # if not specified, project all points
            pt_idx = np.arange(0, len(self.time)).astype(int)

        # if normalize is True:
        #     # normalize the data points
        #     t_max = np.max(self.time[pt_idx])
        #     s_max = np.max(self.space[pt_idx])
        #
        #     pts_t = self.time[pt_idx]/t_max
        #     pts_s = self.space[pt_idx]/s_max
        #     k = line[0]*t_max/s_max
        #     c = line[1]/s_max
        # else:
        pts_t = self.time[pt_idx]
        pts_s = self.space[pt_idx]
        k, c = line

        # compute the distance of those points to the line
        dists = self.compute_dist(pts_t, pts_s, (k, c))

        # fit GMM and update sigma
        gmm = GaussianMixture()
        r = gmm.fit(dists[:, np.newaxis])
        sigma = np.sqrt(r.covariances_[0, 0])

        # if normalize is True:
        #     # recover
        #     sigma = sigma*np.sqrt(s_max**2+line[0]**2*t_max**2)/np.sqrt(1+line[0]**2)

        return sigma, dists

    def split_cluster(self, line, pt_idx=None, min_num_pts=100, counter='0'):

        # print('-- Splitting cluster {0} ...'.format(counter))

        if pt_idx is None:
            # if not specified, project all points
            pt_idx = np.arange(0, len(self.time)).astype(int)

        pts_t = self.time[pt_idx]
        pts_s = self.space[pt_idx]
        k, c = line

        # compute the distance of those points to the line
        dists = self.compute_dist(pts_t, pts_s, (k, c))

        # use gaussian kernel for density estimation
        # print('-------- Performing kernel density analysis ...')
        kde = KernelDensity(bandwidth=0.01, kernel='gaussian').fit(dists[:, np.newaxis])

        # find the minimum point
        x_ticks = np.linspace(np.min(dists), np.max(dists), 100)
        log_dens = kde.score_samples(x_ticks[:, np.newaxis])

        # find the local minimums
        x_minimas_idx = argrelextrema(log_dens, np.less)[0]
        x_segs = zip(np.concatenate([[0], x_minimas_idx]), np.concatenate([x_minimas_idx, [len(x_ticks) - 1]]))

        # fit one Gaussian to each segment
        means = []
        stds = []
        weights = []  # this weight is actually the portion percent
        aic = []
        bic = []
        # print('-------- Splitting to {0} segments ...'.format(len(x_segs)))
        for seg_s, seg_e in x_segs:
            seg_data_idx = (dists >= x_ticks[seg_s]) & (dists < x_ticks[seg_e])

            if sum(seg_data_idx) <= min_num_pts:
                # mini cluster, skip
                # print('Skip small segments with {0} pts'.format(sum(seg_data_idx)))
                continue

            seg_data = dists[seg_data_idx]

            gmm = GaussianMixture()
            r = gmm.fit(seg_data[:, np.newaxis])
            means.append(r.means_[0, 0])
            stds.append(np.sqrt(r.covariances_[0, 0]))
            weights.append(float(len(seg_data)) / len(dists))
            aic.append(gmm.aic(seg_data[:, np.newaxis]))
            bic.append(gmm.bic(seg_data[:, np.newaxis]))

        # return fitted Gaussian and segments
        np.set_printoptions(linewidth=125)
        subclusters = np.array(zip(means, stds, weights, aic, bic))
        # print('-------- Spilted to sub clusters:\n{0}'.format(subclusters))

        if len(subclusters) != 0:
            means = np.asarray(means)
            # converge means to intercepts
            intercepts = c + means * np.sqrt(k ** 2.0 + 1.0)
            lines = [(k, i) for i in intercepts]

            # plot sub cluster
            if self.plot:
                self.plot_sub_cluster(ref_line=(k, c), pt_idx=pt_idx, dists=dists, gaussians=zip(means, stds, weights),
                                  x_ticks=x_ticks, log_dens=log_dens, minimas=x_ticks[x_minimas_idx], counter=counter,
                                  save_name='split_cluster{0}'.format(counter))

            return np.array(zip(lines, stds, weights, aic, bic))
        else:
            return []

    def split_cluster_exhaustive(self, pt_idx=None, min_num_pts=100, speed_range=(-70, 70, 10),
                                 speed_ratio=3.5*6*2.24, counter='0', top_n=8, candidate_slopes=None):

        # -------------------------------------------------------------------------------------
        # explore all directions of lines with reference to the left bottom corner
        speeds = np.arange(speed_range[0], speed_range[1], speed_range[2]).astype(float)
        slopes = -speeds/speed_ratio

        # also explore the candidate directions
        if candidate_slopes is not None:
            slopes = np.concatenate([slopes, np.asarray(candidate_slopes)])
            speeds = np.concatenate([speeds, -np.asarray(candidate_slopes)*speed_ratio])

        all_groups = []
        group_sizes = []

        print('------ Exploring directions:')
        for i, k in enumerate(slopes):

            group = self.split_cluster((k, 0), pt_idx=pt_idx, min_num_pts=min_num_pts,
                                          counter=counter+'_{0:.01f}mph'.format(speeds[i]))
            print('             At {0} mph: {1} subclusters'.format(speeds[i], len(group)))

            all_groups.append(group)
            group_sizes.append(len(group))

        # -------------------------------------------------------------------------------------
        # Determine the most suitable number of clusters by majority vote
        # # majority vote to find the right cluster size except size 0
        # voters = Counter(group_sizes)
        # print('---- Voting in Cluster {0}: {1}'.format(counter, voters.most_common()))
        # num_cluster, counts = voters.most_common()[0]
        # if num_cluster == 0:
        #     try:
        #         num_cluster, counts = voters.most_common()[1]
        #     except IndexError:
        #         return []
        #
        # # find the best projection direction by minimizing sum sig for all clusters that have size num_cluster
        # _idx = [i for i,v in enumerate(np.array(group_sizes) == num_cluster) if v]
        # print _idx
        # best_i = 0
        # min_sigsum = np.inf
        # for i in _idx:
        #     sigsum = np.sum(all_groups[i][:,1])
        #     if sigsum < min_sigsum:
        #         best_i = i
        #         min_sigsum = sigsum

        # -------------------------------------------------------------------------------------
        # Determine the most suitable number of cluster by how much data points are included
        # best_i = 0
        # max_wsum = 0.0
        # for i, g in enumerate(all_groups):
        #     if len(g) == 0:
        #         continue
        #
        #     wsum = np.sum(g[:,2])
        #     if wsum > max_wsum:
        #         best_i = i
        #         max_wsum = wsum
        # print('##### Selected best split for cluster {0} at speed {1} mph'.format(counter, speeds[best_i]))
        # return all_groups[best_i]

        # -------------------------------------------------------------------------------------
        # Simply return the top 5 directions, and check the sub clusters in each direction
        # check all weights and avg_stds and num_clusters
        _weights = []
        _avg_std = []
        _num_clus = []
        _avg_aic = []
        _avg_bic = []
        for i, g in enumerate(all_groups):
            if len(g) == 0:
                _num_clus.append(0)
                _avg_std.append(0)
                _weights.append(0)
                _avg_aic.append(0)
                _avg_bic.append(0)
            else:
                _num_clus.append(len(g))
                _avg_std.append(np.mean(g[:,1]))
                _weights.append(np.sum(g[:,2]))
                _avg_aic.append(np.mean(g[:,3]))
                _avg_bic.append(np.mean(g[:,4]))

        # fig = plt.figure(figsize=(10,8))
        # plt.plot(speeds, _weights, linewidth=2, color='r', label='sum weight')
        # plt.plot(speeds, np.array(_avg_std)*10.0, linewidth=2, color='g', label='avg sigma*10')
        # plt.plot(speeds, -np.array(_avg_aic)/5000.0, linewidth=2, color='b', label='-avg aic/5000')
        # plt.plot(speeds, -np.array(_avg_bic)/5000.0, linewidth=2, color='c', label='-avg bic*5000')
        # plt.plot(speeds, np.array(_num_clus)/10.0, linewidth=2, color='b', label='num cluster/10')
        # plt.title('Spliting cluster {0}'.format(counter))
        # plt.grid(True)
        # plt.legend(loc=2)
        # plt.xlim([-75,60])
        # plt.draw()

        # pick top n in weights and

        top5_w = np.array([i[0] for i in sorted(enumerate(-np.array(_weights)), key=lambda x:x[1])])[0:top_n]
        top5_aic = np.array([i[0] for i in sorted(enumerate(_avg_aic), key=lambda x:x[1])])[0:top_n]

        # last column keeps a record of the number of clusters
        possible_groups = np.zeros((0,5))
        possible_speeds = []
        tot_clus = 0
        for i, g in enumerate(all_groups):
            if len(g) !=0 :
                if i in top5_w and i in top5_aic:
                    possible_groups = np.vstack([possible_groups, g])
                    possible_speeds.append((speeds[i], len(g)))
                    tot_clus += len(g)
        print('------ Found {0} subclusters at speeds : {1}\n'.format(tot_clus, possible_speeds))

        return possible_groups


    def update_inliers(self, line=None, tol=0.1):
        """
        This function updates the inliers, i.e., get data points that lies within tol perpendicular distance to model
        :param mdl: tuple (k, c): y = kx + c
        :param tol: the tolerance for being considered as an inlier
        :return: index list
        """
        k = line[0]
        c = line[1]

        dist = np.abs(self.time * k - self.space + c) / np.sqrt(1 + k ** 2)

        idx = (dist <= tol)

        return np.array([i for i, x in enumerate(idx) if x and i not in self.labeled_pts]).astype(int)

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
        dist = -(pts_t * k - pts_s + c) / np.sqrt(1 + k ** 2)

        return dist

    def plot_progress(self, cur_mdl, save_name=None, title=None, sig_ratio=2.0, speed_ratio=6 * 3.5 * 2.24):

        # plot the initial figure
        fig = plt.figure(figsize=(10, 13))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        # Axis 1 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ===========================================================================
        # plot ax0: the fitting
        x_line = np.asarray([np.min(self.time), np.max(self.time)])
        ax0.scatter(self.time, self.space, color='0.6')

        # --------------------------------------------------------------------
        # scatter the estimated cnverged models
        if len(self.all_mdls) != 0:
            colors = ['b', 'g', 'm', 'c', 'purple', 'teal', 'dogerblue', 'b', 'g', 'm', 'c', 'purple', 'teal',
                      'dogerblue']
            for i, (line, sig, inlier_idx, r2) in enumerate(self.all_mdls):
                ax0.scatter(self.time[inlier_idx], self.space[inlier_idx], color=colors[i], alpha=0.75)
                y_line = line[0] * x_line + line[1]
                ax0.plot(x_line, y_line, linewidth=3, color='k')

                # # plot the tolerance
                # if line[0] != 0:
                #     c1 = line[1] + np.sqrt((sig_ratio * sig) ** 2 + (sig_ratio * sig * line[0]) ** 2)
                #     c2 = line[1] - np.sqrt((sig_ratio * sig) ** 2 + (sig_ratio * sig * line[0]) ** 2)
                # else:
                #     c1 = line[1] + sig_ratio * sig
                #     c2 = line[1] - sig_ratio * sig
                #
                # y_line_1 = line[0] * x_line + c1
                # ax0.plot(x_line, y_line_1, linewidth=2, color='r', linestyle='--')
                # y_line_2 = line[0] * x_line + c2
                # ax0.plot(x_line, y_line_2, linewidth=2, color='r', linestyle='--')

        # --------------------------------------------------------------------
        # plot the current model
        if cur_mdl is not None:
            (line, sig, inlier_idx, r2) = cur_mdl
            ax0.scatter(self.time[inlier_idx], self.space[inlier_idx], color='r')
            y_line = line[0] * x_line + line[1]
            ax0.plot(x_line, y_line, linewidth=3, color='b')

            # plot the tolerance
            if line[0] != 0:
                c1 = line[1] + np.sqrt((sig_ratio * sig) ** 2 + (sig_ratio * sig * line[0]) ** 2)
                c2 = line[1] - np.sqrt((sig_ratio * sig) ** 2 + (sig_ratio * sig * line[0]) ** 2)
            else:
                c1 = line[1] + sig_ratio * sig
                c2 = line[1] - sig_ratio * sig

            y_line_1 = line[0] * x_line + c1
            ax0.plot(x_line, y_line_1, linewidth=2, color='r', linestyle='--')
            y_line_2 = line[0] * x_line + c2
            ax0.plot(x_line, y_line_2, linewidth=2, color='r', linestyle='--')

        ax0.set_title('{0}'.format(title), fontsize=20)
        ax0.set_xlabel('Time (s)', fontsize=18)
        ax0.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax0.set_xlim([np.min(self.time), np.max(self.time)])
        ax0.set_ylim([np.min(self.space), np.max(self.space)])

        # ===========================================================================
        # plot ax1: the analysis of the current model
        # - the distribution of the distance of points to the line
        # - the measures: r^2, num_old_inliers, num_new_inliers, total num_points, slope
        # plot histogram
        if cur_mdl is not None:
            # projection, density kernel segmentation, and GMM
            sig, dists = self.get_sigma(cur_mdl[0], pt_idx=cur_mdl[2])

            sig = cur_mdl[1]
            bin_width = 0.005
            n, bins, patches = ax1.hist(dists, bins=np.arange(-3 * sig, 3 * sig, bin_width),
                                        normed=1, facecolor='green', alpha=0.75)
            # fill the one-sig space.
            x_fill = np.linspace(-sig, sig, 100)
            ax1.fill_between(x_fill, 0, mlab.normpdf(x_fill, 0, sig), facecolor='r', alpha=0.65)
            ax1.plot(bins, mlab.normpdf(bins, 0, sig), linewidth=2, c='r')
            text = ' R2: {0:.4f}; Speed: {1:.2f} mph'.format(cur_mdl[3], -cur_mdl[0][0] * speed_ratio)
            ax1.annotate(text, xy=(0.05, 0.65), xycoords='axes fraction', fontsize=14)
            ax1.set_ylim([0, np.max(n) * 1.5])
            ax1.set_title('Analyzing current model', fontsize=16)

        elif len(self.all_mdls) != 0:
            # plot the final distribution of all clusters
            offset = 0
            y_lim = 0
            for i, (line, sigma, inlier_idx, r2) in enumerate(self.all_mdls):
                # projection, density kernel segmentation, and GMM
                sigma, dists = self.get_sigma(line, pt_idx=inlier_idx)

                # shift means
                dists += 3 * sigma + offset

                bin_width = 0.005
                n, bins, patches = ax1.hist(dists, bins=np.arange(offset, 6 * sigma + offset, bin_width),
                                            normed=1, facecolor=colors[i], alpha=0.75)
                # fill the one-sig space.
                x_fill = np.linspace(2 * sigma + offset, 4 * sigma + offset, 100)
                # fill the onesigma
                ax1.fill_between(x_fill, 0, mlab.normpdf(x_fill, offset + 3 * sigma, sigma), facecolor='r', alpha=0.65)
                # the gaussian line
                ax1.plot(bins, mlab.normpdf(bins, offset + 3 * sigma, sigma), linewidth=2, c='r')

                text = ' R2: {0:.4f}\n {1:.2f} mph\n #:{2}'.format(r2, -line[0] * speed_ratio, len(inlier_idx))
                ax1.annotate(text, xy=(offset + sigma, np.max(n) * 1.3), fontsize=10)
                ax1.set_title('All converged models', fontsize=16)
                y_lim = np.max([np.max(n), y_lim])
                # update offset to the right 3sigma of this distribution
                offset += 6 * sigma
            ax1.set_ylim([0, y_lim * 1.8])
            ax1.set_xlim([0, offset])

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()

    def plot_sub_cluster(self, ref_line, pt_idx, dists, gaussians, x_ticks=None,
                         log_dens=None, minimas=None, counter='0', save_name=None):

        # plot the initial figure
        fig = plt.figure(figsize=(10, 13))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        # Axis 1 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ----------------------------------------------------
        # plot ax0: the fitting
        x_line = np.asarray([np.min(self.time), np.max(self.time)])
        ax0.scatter(self.time, self.space, color='0.6')  # all points

        # the points being projected
        ax0.scatter(self.time[pt_idx], self.space[pt_idx], color='g')

        # plot the models
        for mean, std, w in gaussians:
            line = (ref_line[0], ref_line[1] + mean * np.sqrt(ref_line[0] ** 2 + 1))
            # plot the mean line
            y_line = line[0] * x_line + line[1]
            ax0.plot(x_line, y_line, linewidth=2, color='b')

            # plot the std line
            y_line = line[0] * x_line + line[1] - std * np.sqrt(1 + line[0] ** 2)
            ax0.plot(x_line, y_line, linewidth=2, color='r', linestyle='--')
            y_line = line[0] * x_line + line[1] + std * np.sqrt(1 + line[0] ** 2)
            ax0.plot(x_line, y_line, linewidth=2, color='r', linestyle='--')

        ax0.set_title('Splitting cluster {0}'.format(counter), fontsize=20)
        ax0.set_xlabel('Time (s)', fontsize=18)
        ax0.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax0.set_xlim([np.min(self.time), np.max(self.time)])
        ax0.set_ylim([np.min(self.space), np.max(self.space)])

        # ----------------------------------------------------
        # plot the analysis of the histogram
        # plot histogram
        bin_width = 0.01
        n, bins, patches = ax1.hist(dists, bins=np.arange(x_ticks[0], x_ticks[-1], bin_width),
                                    normed=1, facecolor='green', alpha=0.75)

        # plot the density kernel function
        if log_dens is not None:
            ax1.plot(x_ticks, np.exp(log_dens), '--', linewidth=3, color='m')

        # plot the segmentation
        if minimas is not None:
            for s in minimas:
                ax1.axvline(s, linestyle='-', linewidth=2, color='k')

        for mdl in gaussians:
            # normalize the density
            ax1.plot(bins, mlab.normpdf(bins, mdl[0], mdl[1]) * mdl[2], 'b')

            # mark the current tol threshold
            sig_1_l = mdl[0] - mdl[1]
            sig_1_u = mdl[0] + mdl[1]
            sig_1_y = mlab.normpdf(sig_1_l, mdl[0], mdl[1]) * mdl[2]

            ax1.plot([sig_1_l, sig_1_l], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')
            ax1.plot([sig_1_u, sig_1_u], [0, sig_1_y],
                     linestyle='--', linewidth=2, color='r')

            text = ' std: {0:.04f}'.format(mdl[1][0])
            ax1.annotate(text, xy=(mdl[0] - mdl[1], np.max(n) * 1.2), fontsize=14)

        ax1.set_ylim([0, np.max(n) * 1.5])
        ax1.set_xlim([x_ticks[0], x_ticks[-1]])
        # ax1.set_xlabel('Normalized distance', fontsize=16)
        ax1.set_title('Distribution of distances to current lines', fontsize=18)

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()


if __name__ == '__main__':
    main()
