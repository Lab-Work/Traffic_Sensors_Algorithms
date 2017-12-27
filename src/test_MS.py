import sys, os
from os.path import exists

import itertools
import time, glob
from datetime import datetime
from datetime import timedelta
from contextlib import contextmanager
from copy import deepcopy
import imutils
from collections import OrderedDict
from ast import literal_eval as make_tuple

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.patches as patches

import numpy as np
import scipy
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

__author__ = 'Yanning Li'
__version__ = "1.0"
__email__ = 'yli171@illinois.edu'

"""
This script is used to test the midpoint smoothing algorithm for speed estimation.

The key heuristics are follows
    - The trajectories always extends linearly along some trajectory. So starting from certain row of the trajectory, we
      can extend it along its direction.
    - To deal with outliers, simply use midpoint smoothing of those data in the trajectory.

However, the above heuristics are quite difficulty to implement, since there are too many corner cases for determining
how the data should be extended along the trajectory, especially with a lot of noise data. But the main idea is follows:
Our earlier algorithm uses iterative linear regression to converge to all the data point in the dataset. However, it is
possible to simple starting from one portion of the trajectory and extend it based on heuristics. It may not converge to
the best fit of the trajectory, but is cheaper than iterative regression.

For implementation:
    - First run DBSCAN to get initial clusters which we know for sure belongs to certain trajectory.
    - Seconds, extend the data points in the DBSCAN cluster along its tentative direction to the full trajectory.
        The extension works as follows:
        - For each bar (a sequence of points in a row) in DBSCAN cluster, extend it to its left most and right most
          (i.e., NaN value), this gives index [head, tail] for this row
        - Then in the head and tail, look row above or below, include all data in the row above or below [head, tail],
        - Then extend the new row data to its leftmost and rightmost.
        - Continue to get the full trajectory
    - Finally, compute slope by
        - discard the trajectory, if the number of data point is too small.
        - clean the data in the trajectory (remove those with excessively long width)
        - compute the midpoint position
        - either perform linear regression of the midpoint, or, use smoothing of the midpoint


# For comparison with the existing algorithm, we only compute the slope and compare the slopes.
each veh is a dict
    - 't_in': datetime, vehicle enter time
    - 't_out': datetime, vehicle exit time
    - 'line': (k,c), s = kt+c, where t is in seconds, and s is relative space after nonlinear transformation
    - 'inliers': [(t,s)], a list of tuples, each tuple is (t,x), t is datetime
    - 'detection_window': tuple of datetime, the start and end time of the detection window
    - 'medians': a list of tuples, each tuple is (t,x), t is datetime
    # - 'lb': a list of tuples, each tuple is (t,x), which gives the percentile lb
    # - 'ub': a list of tuples, each tuple is (t,x), which gives the percentile ub
"""

def main():

    # run_alg()

    # compute_speed_err_ms_vs_lr()

    compute_speed_err_vs_true()


def compute_speed_err_ms_vs_lr():
    """
    Here we only compare the midpoint smoothing algorithm vs the original linear regression algorithm.
    :return:
    """

    # linear regression vs midpoint smoothing
    LR_file = '../workspace/0530_2017/figs_MS/speed_LR.txt'
    MS_file = '../workspace/0530_2017/figs_MS/speed_MS.txt'

    # ===========================================================
    # Load the speed LR: t_in, t_out, dist (m), speed (mph), valid
    # only save [t_mean, dist (m), speed(mph), valid]
    lr = []
    with open(LR_file, 'r') as f:
        next(f)
        for line in f:
            if line[0] == '#':
                continue

            items = line.strip().split(',')
            t_start = str2time(items[0])
            t_end = str2time(items[1])
            lr.append( [t_start + (t_end-t_start)/2, float(items[2]), float(items[3]), items[4]] )

    # sort by t_mean
    lr.sort(key=lambda x: x[0])
    lr = np.asarray(lr)

    valid_idx = (lr[:,3] == 'True')

    # ===========================================================
    # Load the speed_MS: t_in, t_out, slope
    # save t_mean, slope
    ms = []
    with open(MS_file, 'r') as f:
        next(f)
        for line in f:
            if line[0] == '#':
                continue
            items = line.split(',')
            t_start = str2time(items[0])
            t_end = str2time(items[1])
            ms.append( [t_start + (t_end-t_start)/2, float(items[2]) ] )

    ms.sort(key=lambda x: x[0])
    ms = np.asarray(ms)

    # ===========================================================
    # Now they should corresponds one to one
    # compute the speed using the distance from speed_lr: speed (mph) = slope * distance * 6.0 * 2.23694
    speed_ms = ms[:,1]*lr[:,1]*6.0*2.23694
    speed_err = speed_ms - lr[:, 2]

    # save ms estimation result
    _ms = np.hstack(([ms, np.expand_dims(speed_ms, axis=1)]))
    np.save('../workspace/0530_2017/figs_MS/MS_detected_vehs.npy', _ms)

    plot_hist([speed_err], labels=None, title='speed error: MS vs LR', xlabel='speed (mph)')
    print('length of speeds: {0}'.format(len(speed_err)))
    print('Outliers: lr_t_m, ms_t_m, lr_speed, ms_speed')
    for i, v in enumerate(speed_err):
        if abs(v) >= 5:
            print('    {0},  {1},  {2:.2f},  {3:.2f}, {4:.2f}'.format(time2str(lr[i,0]), time2str(ms[i,0]),
                                                             lr[i, 2], speed_ms[i] , speed_err[i] ))

    # speed_ms = ms[valid_idx,1]*lr[valid_idx,1]*6.0*2.23694
    # speed_err = (speed_ms - lr[valid_idx, 2])
    # print('length of speeds: {0}'.format(len(speed_err)))
    # plot_hist([speed_err], labels=None, title='speed error: MS vs LR', xlabel='speed (mph)')
    # print('Outliers: lr_t_m, ms_t_m, lr_speed, ms_speed')
    # for i, v in enumerate(speed_err):
    #     if abs(v) >= 5:
    #         print('    {0},  {1},  {2:.2f},  {3:.2f}, {4:2f}'.format(time2str(lr[valid_idx,0][i]), time2str(ms[valid_idx,0][i]),
    #                                                          lr[valid_idx, 2][i], speed_ms[i], speed_err[i] ))

    plt.show()


def compute_speed_err_vs_true():
    """
    Here we compute the speed error using
    :return:
    """
    # ===================================================================================
    # [start_time (dt), end_time (dt), speed (mph), distance (m), image speed (px/frame), image distance (px)]
    s1_true_file = '../workspace/0530_2017/labels_v11_post.npy'

    # each item is a dict:  ['t_in',  't_out', 'distance', 'speed', 'closer_lane']
    s1_lr_file = '../workspace/0530_2017/figs/speed/s1/v2_3/detected_vehs_post_comb_v2.npy'

    # [mean_time (dt), slope, speed (mph)]
    s1_ms_file = '../workspace/0530_2017/figs_MS/MS_detected_vehs.npy'

    # -----------------------------------------------------------
    # Load true and save in array: [mean_time (dt), speed (mph)]
    _tmp = np.load(s1_true_file)
    s1_true = []
    for row in _tmp:
        s1_true.append( [row[0] + (row[1]-row[0])/2, row[2] ])
    s1_true.sort(key=lambda x:x[0])
    s1_true = np.asarray(s1_true)

    # -----------------------------------------------------------
    # Load detection result using adaptiveLR: [mean_time (dt), speed (mph)]
    _tmp = np.load(s1_lr_file)
    s1_lr = []
    for v in _tmp:
        if v['closer_lane'] is True:
            s1_lr.append( [ v['t_in'] + (v['t_out']-v['t_in'])/2, abs(v['speed']) ] )
    s1_lr.sort(key=lambda x:x[0])
    s1_lr = np.asarray(s1_lr)

    # -----------------------------------------------------------
    # Load detection result using Midpoint smoothing: [mean_time (dt), speed (mph)]
    _tmp = np.load(s1_ms_file)
    all_ms = []
    for row in _tmp:
        all_ms.append( [ row[0], abs(row[2]) ] )
    all_ms.sort(key=lambda x:x[0])
    all_ms = np.asarray(all_ms)

    # ===================================================================================
    print('Loaded results:   true,   LR,   MS')
    print('                   {0},   {1},   {2}'.format(len(s1_true), len(s1_lr), len(all_ms)))


    # -----------------------------------------------------------
    # match LR vs true
    dt = 0.5    # seconds
    # matches: a list of tuple [(item1, item2),... ], where first item from list1 and second from list2
    #          Each item is (mean_time, idx), where idx is the index in the corresponding list
    lr_true = np.asarray(match_lists(s1_lr[:,0], s1_true[:,0], dt))
    lr_tp = sum( ~pd.isnull(lr_true[:,0]) & ~pd.isnull(lr_true[:,1]) )
    lr_fp = sum( ~pd.isnull(lr_true[:,0]) & pd.isnull(lr_true[:,1]) )
    lr_fn = sum( pd.isnull(lr_true[:,0]) & ~pd.isnull(lr_true[:,1]) )
    print('Matched LR vs True:  TP,    FP,    FN')
    print('                    {0},   {1},   {2}'.format(lr_tp, lr_fp, lr_fn))

    # Compute the RMSE and plot histogram
    # error list of TP: [s1_dt, true_dt, s1_speed, true_speed, error]
    lr_err = []
    for it1, it2 in lr_true:
        if it1 is not None and it2 is not None:
            lr_err.append([ s1_lr[it1[1],0], s1_true[it2[1],0], s1_lr[it1[1],1], s1_true[it2[1],1],
                            s1_lr[it1[1],1]-s1_true[it2[1],1] ])
    lr_err = np.array(lr_err)
    lr_rmse = np.sqrt( np.sum(lr_err[:,4]**2)/len(lr_err) )

    # -----------------------------------------------------------
    # match MS vs LR to extract those on the correct lane
    dt = 0.5    # seconds
    # matches: a list of tuple [(item1, item2),... ], where first item from list1 and second from list2
    #          Each item is (mean_time, idx), where idx is the index in the corresponding list
    ms_lr = np.asarray(match_lists(all_ms[:,0], s1_lr[:,0], dt))
    ms_lr_tp = sum( ~pd.isnull(ms_lr[:,0]) & ~pd.isnull(ms_lr[:,1]) )
    ms_lr_fp = sum( ~pd.isnull(ms_lr[:,0]) & pd.isnull(ms_lr[:,1]) )
    ms_lr_fn = sum( pd.isnull(ms_lr[:,0]) & ~pd.isnull(ms_lr[:,1]) )
    print('Matched MS vs LR:  TP,    FP,    FN')
    print('                    {0},   {1},   {2}'.format(ms_lr_tp, ms_lr_fp, ms_lr_fn))

    # extract the TP as s1_ms
    s1_ms = []
    for it1, it2 in ms_lr:
        # true positives
        if it1 is not None and it2 is not None:
            s1_ms.append( all_ms[it1[1],:] )
    s1_ms = np.array(s1_ms)
    print('Extracted s1_ms with length: {0}'.format(len(s1_ms)))

    # -----------------------------------------------------------
    # # match MS vs true
    dt = 0.5    # seconds
    # matches: a list of tuple [(item1, item2),... ], where first item from list1 and second from list2
    #          Each item is (mean_time, idx), where idx is the index in the corresponding list
    ms_true = np.asarray(match_lists(s1_ms[:,0], s1_true[:,0], dt))
    ms_tp = sum( ~pd.isnull(ms_true[:,0]) & ~pd.isnull(ms_true[:,1]) )
    ms_fp = sum( ~pd.isnull(ms_true[:,0]) & pd.isnull(ms_true[:,1]) )
    ms_fn = sum( pd.isnull(ms_true[:,0]) & ~pd.isnull(ms_true[:,1]) )
    print('Matched MS vs True:  TP,    FP,    FN')
    print('                    {0},   {1},   {2}'.format(ms_tp, ms_fp, ms_fn))

    # Compute the RMSE and plot histogram
    # error list of TP: [s1_dt, true_dt, s1_speed, true_speed, error]
    ms_err = []
    for it1, it2 in ms_true:
        if it1 is not None and it2 is not None:
            ms_err.append([ s1_ms[it1[1],0], s1_true[it2[1],0], s1_ms[it1[1],1], s1_true[it2[1],1],
                            s1_ms[it1[1],1]-s1_true[it2[1],1] ])
    ms_err = np.array(ms_err)
    ms_rmse = np.sqrt( np.sum(ms_err[:,4]**2)/len(ms_err) )

    # -----------------------------------------------------------
    # print out the results and plot
    print('\nRMSE:  LR,           MS')
    print('    :  {0:.3f},     {1:.3f}'.format(lr_rmse, ms_rmse))
    plot_hist([lr_err[:,4], ms_err[:,4]], labels=['LR', 'MS'], title='Speed estimation error', xlabel='Speed (mph)')





    plt.show()



def run_alg():

    # ====================================================================
    # Configuration of parameters
    paras= OrderedDict()
    paras['pir_res'] = (4,32)
    paras['pir_fov'] = 53.0
    paras['sampling_freq'] = 64.0
    paras['pir_fov_offset'] = 3.0
    paras['tx_ratio'] = 6.0

    # initial clustering using DBSCAN
    # max number of points in r=3.5 is 37; in r=2.5 is 21; in r=1.5 is 9
    paras['DBSCAN_r'] = 2.5
    paras['DBSCAN_min_pts'] = 15

    # index for determining if a traj should be discarded, num_data_rows >= threshold (max 32)
    paras['min_num_data_rows'] = 15
    paras['min_avg_width'] = 5  #   ~ 0.07 s
    paras['max_slope_difference'] = 0.12    # the maximum slope difference of slopes of percentiles for splitting.
                                            # 0.06 ~ about 5 mph; 0.12 about 10 mph; 0.2 about 15 mph
    paras['default_slope'] = -0.25
    paras['default_width'] = 20     # ~ 0.3 s
    paras['max_width_ratio_ub'] = 2.0
    paras['max_width_ratio_lb'] = 0.1
    paras['median_tol'] = 5
    paras['percentile_lb'] = 20
    paras['percentile_ub'] = 80

    paras['min_inliers'] = 164
    # same vehicle threshold, same vehicle if overlapping percentage of inliers of two vehicle exceeds the threshold
    paras['TH_same_veh_overlapping'] = 0.5

    # ====================================================================
    # Load the data
    folder = '0530_2017'
    speed_range = (-71,-1)
    save_dir = '../workspace/{0}/'.format(folder)

    # read in the data with background subtracted.
    # only run the algorithm for this period of time

    # two overlapping trajectories
    t_start = str2time('2017-05-30 21:21:04.5')
    t_end = str2time('2017-05-30 21:21:09.5')

    # excessive amount of noise
    # t_start = str2time('2017-05-30 20:57:47.5')
    # t_end = str2time('2017-05-30 20:57:52.5')

    # long noise strip
    # t_start = str2time('2017-05-30 20:58:52.5')
    # t_end = str2time('2017-05-30 20:58:57.5')

    # other direction vehicle
    # t_start = str2time('2017-05-30 21:02:25.0')
    # t_end = str2time('2017-05-30 21:02:30.5')

    # thin trajectory
    # t_start = str2time('2017-05-30 21:13:07.5')
    # t_end = str2time('2017-05-30 21:13:12.5')

    if False:
        norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
        norm_df.index = norm_df.index.to_datetime()

        frames = (norm_df.index >= t_start) & (norm_df.index <= t_end)
        _norm_df = norm_df.ix[frames,:]
        _norm_df.to_csv(save_dir + 's1_small_{0}__{1}.csv'.format(time2str_file(t_start), time2str_file(t_end)))
    else:
        _norm_df = pd.read_csv(save_dir + 's1_small_{0}__{1}.csv'.format(time2str_file(t_start),
                                                                         time2str_file(t_end)), index_col=0)
        _norm_df.index = _norm_df.index.to_datetime()

    # ====================================================================
    # Run the speed estimation algorithm
    norm_df = pd.read_csv(save_dir + 's1_2d_KF__20170530_205400_0__20170530_214500_0_prob95.csv', index_col=0)
    norm_df.index = norm_df.index.to_datetime()
    alg = MedianFit(paras)
    alg.run(norm_df, window_s=5.0, step_s=2.5, speed_range=speed_range,
            save_dir='../workspace/{0}/figs_MS/s1/'.format(folder),
            t_start=str2time('2017-05-30 20:55:0.0'),
            t_end=str2time('2017-05-30 21:45:0.0'), plot_final=True, plot_progress=False)


    # est = DirectCluster(_norm_df, paras=paras, window_s=(t_end-t_start).total_seconds(), speed_range=speed_range,
    #                 plot_final=False, plot_progress=False, save_dir='../workspace/{0}/figs_MS/s1/'.format(folder))
    #
    # est.estimate_slope()

    # plt.show()


class MedianFit:
    """
    This class is the top layer algorithm which detects the vehicle and estimate the trajectory slope using directional
    clustering
    """
    def __init__(self, paras):

        self.paras = paras

        # save all vehicles in this list
        self.vehs = []

    def run(self, norm_df, window_s=5.0, step_s=2.5, speed_range=(1,60), save_dir='./', t_start=None, t_end=None,
            plot_final=True, plot_progress=False):

        # -------------------------------------------------------------------------------------------------------
        # Only work on those data within in the time interval
        if t_start is None: t_start = norm_df.index[0]
        if t_end is None: t_end = norm_df.index[-1]

        frames = (norm_df.index >= t_start) & (norm_df.index <= t_end)
        _norm_df = norm_df.ix[frames]

        # -------------------------------------------------------------------------------------------------------
        # Detect the vehicle using VehDet class
        windows = self.detect_vehs(_norm_df, window_s=window_s, step_s=step_s)

        # -------------------------------------------------------------------------------------------------------
        # Speed estimation
        if not exists(save_dir): os.makedirs(save_dir)

        for win in windows:
            est = DirectCluster(norm_df=_norm_df.ix[win[0]:win[1]], paras=self.paras, window_s=window_s,
                           speed_range=speed_range,
                           plot_final=plot_final, plot_progress=plot_progress, save_dir=save_dir)

            est.estimate_slope()

            # convert to the correct format
            vehs_in_win = est.convert_to_veh()

            for veh in vehs_in_win:
                # register the vehicle to self.vehs list
                self._register_veh(veh, speed_range=speed_range)

        # ----------------------------------------------------------------------------------------------------
        # save the final detection result, remove those None
        _vehs = [v for v in self.vehs if v is not None]

        np.save(save_dir + 'detected_vehs.npy', _vehs)
        self._save_vehs_txt(_vehs, save_dir, 'detected_vehs.txt')
        self._save_paras(save_dir, 'paras.txt')


    def _register_veh(self, veh, speed_range=None):
        """
        This function registers the vehicle detected in each time window. In addition:
            - It checks if this vehicle has been detected in previous windows,
            - It caps the speed to speed_range assuming ditance is 10 meters (max)
        :param veh: the vehicle to be registered
        :param speed_range: (-50,-1) mph in one direction or (1,50) mph in the other direction
        :return:
        """

        min_dist = 3.0

        if speed_range is not None:
            # max speed in m/s
            max_speed = np.max([ abs(speed_range[0]), abs(speed_range[1]) ])/2.24
        else:
            max_speed = np.inf

        if len(self.vehs) == 0:
            # If speed is higher than the max speed, then discard
            if abs(veh['line'][0]*self.paras['tx_ratio']*min_dist) <= max_speed:
                self.vehs.append(veh)

        else:
            # [old_v, num_inliers, idx_in_self_vehs]
            old_veh_list = []
            for i, old_v in enumerate(self.vehs):
                # there could be multiple old vehicles
                if self._same_veh(old_v, veh):
                    old_veh_list.append([old_v, len(old_v['inliers']), i])

            if len(old_veh_list) != 0:
                # Found duplicated vehicles

                old_veh_list = np.array(old_veh_list)
                # Determine which vehicle estimate should be left: the vehicle containing the maximum number of point should
                # be the one left
                if len(veh['inliers']) > np.max(old_veh_list[:,1]):
                    # This vehicle contains the largest number of inliers
                    if abs(veh['line'][0]*self.paras['tx_ratio']*min_dist) <= max_speed:
                        # If speed is good, then remove all old vehicles and register this one

                        print('######################## Updated vehicle entering at {0}\n'.format(veh['t_in']))
                        for idx in old_veh_list[:,2]:
                            # remove the entry by setting it as none
                            self.vehs[idx] = None

                        self.vehs.append(veh)
                    else:
                        # current speed is not good, then do nothing. DO NOT remove old vehciles
                        pass

                else:
                    # only keep the old vehicle with the maximum num_inliers
                    max_num_inliers = np.max(old_veh_list[:,1])
                    for old_v, n, idx in old_veh_list:
                        if n != max_num_inliers:
                            self.vehs[idx] = None
            else:
                # no duplicated vehicles, check speed and register
                if abs(veh['line'][0]*self.paras['tx_ratio']*min_dist) <= max_speed:
                    self.vehs.append(veh)


    def detect_vehs(self, norm_df, window_s=5.0, step_s=2.5):
        """
        This function outputs the windows that may contain a vehicle
        :param norm_df: the pandas DataFrame structure
        :param TH_det: the threshold for determining if there are vehicles present in the window
        :param window_s: seconds, the window size
        :param step_s: seconds, moving windows by steps
        :return: a list of tuples (t_s, t_e) wher t_s and t_e are datetime
        """

        pir_data = deepcopy(norm_df.ix[:, [i for i in norm_df.columns if 'pir' in i]])
        # replace the nan value to 0.0
        # pir_data.values[np.isnan(pir_data.values)] = 0.0

        # compute the energy in each 0.5 window
        dw = timedelta(seconds=window_s)
        ds = timedelta(seconds=step_s)
        t_s = pir_data.index[0]
        t_e = pir_data.index[0] + dw

        veh_t = []
        veh_e = []
        wins = []

        while t_e <= pir_data.index[-1]:
            # e = np.sum(pir_data.ix[(pir_data.index >= t_s) & (pir_data.index <= t_e)].values)
            e = np.sum(~np.isnan(pir_data.ix[(pir_data.index >= t_s) & (pir_data.index <= t_e)].values))
            veh_t.append(t_s)
            veh_e.append(e)
            t_s = t_s + ds
            t_e = t_s + dw

            # if e >= TH_det:
            if e >= self.paras['min_inliers']:
                wins.append((t_s, t_e))
        return wins


    def _save_paras(self, save_dir, save_name):
        """
        This function saves the parameters used to generate the results
        :param save_dir: the directory for saving
        :param save_name: the file name
        :return:
        """
        with open(save_dir + save_name, 'w+') as f:
            for key in self.paras.keys():
                f.write('{0}: {1}\n'.format(key, self.paras[key]))


    def _save_vehs_txt(self, _vehs, save_dir, save_name):
        """
        This function extracts and only saves the most important detection results from self.vehs
            [t_in, t_out, slope]
        :param save_dir: the directory for saving
        :param save_name: the file name
        :return:
        """
        with open(save_dir+save_name, 'w+') as f:
            f.write('t_in, t_out, slope\n')
            for veh in _vehs:
                f.write('{0},{1},{2}\n'.format(veh['t_in'], veh['t_out'], veh['line'][0]))


    def _same_veh(self, v1, v2):
        """
        This function evaluates if v1 and v2 are essentially fitting to the trace of the same vehicle
        :param v1: vehicle dict
        :param v2: vehicle dict
        :return: True if same
        """
        if v1 is None or v2 is None:
            return False

        if v1['t_out'] <= v2['t_in'] or v1['t_in'] >= v2['t_out']:
            return False

        # Use the amount of overlapping of supporting data point to determine if they are the same vehicle.
        overlapping_pts = [p for p in set(v1['inliers']) & set(v2['inliers'])]

        overlapping_perc = float(len(overlapping_pts))/np.min([len(set(v1['inliers'])), len(set(v2['inliers']))])

        if overlapping_perc >= self.paras['TH_same_veh_overlapping']:
            print('########## Found duplicated vehicles with overlapping {0}'.format(overlapping_perc))
            print('                 duplicated v1: ({0}, {1})'.format(v1['t_in'], v1['t_out']))
            print('                 duplicated v2: ({0}, {1})'.format(v2['t_in'], v2['t_out']))
            return True
        else:
            return False



class DirectCluster:
    """
    This class progressively expand the points in a trajecotry and then find the midpoint at each spacial location and
    the use smoothing to estimate its speed.
    """

    def __init__(self, norm_df, paras, window_s=5.0, speed_range=(1,60),
                 plot_final=False, plot_progress=False, save_dir='./'):
        """
        Initialize the algorithm
        :param norm_df: the data used for detecting a vehicle. Columns are:
            'pir_0x0', 'pir_1x0', ..., 'ultra', 'Tamb_1', 'Tamb_2'
        :param paras: the parameters used for this algorithm
        :param window_s: the time window of norm_df
        :param speed_range: the speed range
        :param plot_final: True or False, plot the final fitted result
        :param save_dir: directory for saving the result
        :return:
        """
        # ------------------------------------------------------------
        # Important properties
        self.paras = paras
        self.window_s = window_s
        self.speed_range = speed_range
        self.mps2mph = 2.23694  # 1 m/s = 2.23694 mph
        self.save_dir = save_dir
        self.plot_final = plot_final
        self.plot_progress = plot_progress

        # ------------------------------------------------------------
        # determine the direction of the slopes; only consider the correct direction
        if self.speed_range[0] >= 0 and self.speed_range[1] >= 0:
            # positive direction
            self.direction = 'positive'
        elif self.speed_range[0] <= 0 and self.speed_range[1] <= 0:
            # negative direction
            self.direction = 'negative'
        else:
            # both direction
            self.direction = 'both'

        # ------------------------------------------------------------
        # Nonlinear transform
        # x_grid is 128 x 1, with each 4 row duplicated to the same position
        self.x_grid = self._new_nonlinear_transform()
        self.init_dt, self.end_dt = norm_df.index[0], norm_df.index[-1]

        self.t_grid = np.asarray([(t - norm_df.index[0]).total_seconds() for t in norm_df.index])

        # ------------------------------------------------------------
        # Convert the PIR matrix (n_samples x 128) to a list of data points tuples
        _time = []
        _space = []
        i = 0
        for cur_t, row in norm_df.iterrows():
            for col in range(0, self.paras['pir_res'][0] * self.paras['pir_res'][1]):
                if ~np.isnan(row.values[col]):
                    _time.append(self.t_grid[i])
                    _space.append(self.x_grid[col])
            i += 1

        # remove duplicated data points
        pts = set(zip(_time, _space))
        l_pts = np.asarray([list(i) for i in pts])
        self.pts_time = l_pts[:, 0]
        self.pts_space = l_pts[:, 1]

        # ------------------------------------------------------------
        # construct a conversion between the point list and point matrix
        #   - Given clus as the index of points in the list.
        #     Then self.idx_time[clus] gives the column index, and self.idx_space[clus] gives the row index
        #   - Given t_idx, and x_idx tuples of the entries.
        #     Then self.mat2list[x_idx, t_idx] gives the index for pts in the list

        self._t_grid = sorted(list(set(self.t_grid)))
        self._x_grid = sorted(list(set(self.x_grid)))

        self.mat2list = -np.ones((self.paras['pir_res'][1], len(self._t_grid))).astype(int)

        # construct a time space domain with len(x_grid) x len(t_grid) and save the unique data points
        # For self.tx_cor:
        #       np.nan - no data point;
        #       0.0 - not labeled yet
        #       < 0 ~ initial DBSCAN cluster, can be merged
        #       > 0 ~ final expanded cluster, can not be merged
        self.tx_cor = np.ones((len(self._x_grid), len(self._t_grid)))*np.nan

        self.idx_time = []
        self.idx_space = []
        for i in xrange(0, len(self.pts_time)):

            t_idx, s_idx = self._t_grid.index( self.pts_time[i] ), self._x_grid.index( self.pts_space[i] )

            # save the index of points for clustering
            self.idx_time.append( t_idx )
            self.idx_space.append( s_idx )
            # mark the points in the matrix for expanding the trajectory
            self.tx_cor[s_idx, t_idx] = 0.0

            self.mat2list[s_idx, t_idx] = i

        self.idx_time = np.asarray(self.idx_time)
        self.idx_space = np.asarray(self.idx_space)

        # ------------------------------------------------------------
        # save the final trajectories and information
        # a list of pts index for each trajectory
        self.all_traj = []
        # a list of list of int, one for each trajectory, if no median, set as np.nan
        self.all_medians = []
        # a list of list, one for each trajectory, if no data row, set as 0 or np.nan
        self.all_widths = []
        # a list of int, one for each trajectory
        self.all_traj_num_rows = []
        # a list of list of int, one for each trajectory, gives the percentile
        self.all_percentile_lb = []
        # a list of list of int, one for each trajectory, gives the percentile
        self.all_percentile_ub = []

        # the structure saving all vehicles in correct format.


    def estimate_slope(self):
        """
        This function estimates the speed of the vehicle
        :return:
        """

        # ================================================================================
        # First use DBSCAN on the index space of points to cluster the initial inliers
        clusters = self._init_clusters_idx()
        # self._plot_clusters(clusters, title='DBSCAN', save_name= None, option='index')

        # ================================================================================
        # Second, expand each cluster to include all data point in its trajectory
        for i, clus in enumerate(clusters):
            print('\n$$$ Expanding cluster {0}...'.format(i+1))
            self.expand_traj(clus, i+1)

        # ================================================================================
        # Third, convert each trajectory to the predefined format

        # ================================================================================
        # Finally, print and visualize the result
        print('\nFinalized {0} trajectories'.format(len(self.all_traj)))
        for i, clus in enumerate(self.all_traj):
            print('------ trajectory {0}: {1} pts'.format(i, len(clus)))

        if self.plot_final:
            self._plot_fitted_clusters(clusters=None, title='{0}'.format(time2str(self.init_dt)),
                                   save_name='{0}'.format(time2str_file(self.init_dt)))

    def convert_to_veh(self):
        """
        This function converts the detected trajecotries into vehs dict with predefined keys
            - 'line': (k,c), s = kt+c, where t is in seconds, and s is relative space after nonlinear transformation
            - 't_in': datetime, vehicle enter time
            - 't_out': datetime, vehicle exit time
            - 'detection_window': tuple of datetime, the start and end time of the detection window
            - 'medians': a list of tuples, each tuple is (t,x), t is datetime
            - 'inliers': [(t,s)], a list of tuples, each tuple is (t,x), t is datetime
        :return:
        """
        all_vehs = []

        for i, clus in enumerate(self.all_traj):

            veh = OrderedDict()

            # ---------------------------------------------------------------
            # compute the line through the median: line = (k,c), where x = kt+c
            line = self.run_lr(self.all_medians[i])
            veh['line'] = line

            # ---------------------------------------------------------------
            # compute the t_in and t_out
            t_l, t_r = (self._x_grid[0]-line[1])/line[0], (self._x_grid[-1]-line[1])/line[0]
            if t_l > t_r:
                t_l, t_r = t_r, t_l

            veh['t_in'] = self.init_dt + timedelta(seconds=t_l)
            veh['t_out'] = self.init_dt + timedelta(seconds=t_r)

            # ---------------------------------------------------------------
            # save the time window for this detection
            veh['detection_window'] = (self.init_dt, self.end_dt)

            # ---------------------------------------------------------------
            # save the medians in tuples (t datetime, x relative)
            medians_tx = []
            for row, col in enumerate(self.all_medians[i]):
                if ~np.isnan(col):
                    # convert units to datetime from seconds
                    t_sec, x_loc = self.rowcol_to_loc(row, col)
                    medians_tx.append( [ self.init_dt + timedelta(seconds=t_sec) , x_loc ] )
            veh['medians'] = np.asarray(medians_tx)

            # ---------------------------------------------------------------
            # save the list of inliers in (t datetime, x relative)
            pts_t = [ self.init_dt + timedelta(seconds=self.pts_time[pt_idx]) for pt_idx in self.all_traj[i] ]

            # for pt_idx in self.all_traj[i]:
            #     inliers.append( [ self.init_dt + timedelta(seconds=self.pts_time[pt_idx]),
            #                       self.pts_space[pt_idx]] )

            veh['inliers'] = zip( pts_t, self.pts_space[self.all_traj[i]] )

            # ---------------------------------------------------------------
            # append to all vehicles
            all_vehs.append(veh)

        return all_vehs


    def run_lr(self, medians):
        """
        This function runs linear regression through the medians, and returns a line
            line = (k,c) where x = kt+c
        :param medians: 32 x1 array, each row contains the time index of the median in that row
        :return:
            line = (k,c) where x = kt+c
        """
        # convert the medians to the actual time space locations in units
        #   (seconds, relative space after nonlinear transform)
        tx = []

        for x, t in enumerate(medians):
            if ~np.isnan(t):
                tx.append( self.rowcol_to_loc(x, t) )
        tx = np.asarray(tx)

        t, x = tx[:,0], tx[:,1]

        # t = _slope*s + _intercept
        _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x, t)

        # convert to the line format: s = kt + c
        line = np.array([1 / _slope, -_intercept / _slope])

        return line


    def _init_clusters_idx(self):
        """
        This function performs DBSCAN in the index space of data points to identify initial inliers
        :return: [cluster_1, cluster_2], each cluster is a list of int (indices) of idx_time, idx_space
        """
        clusters = []

        samples = np.vstack([self.idx_time, self.idx_space]).T

        if len(samples) == 0:
            return []

        y_pre = DBSCAN(eps=self.paras['DBSCAN_r'], min_samples=self.paras['DBSCAN_min_pts']).fit_predict(samples)
        num_clusters = len(set(y_pre)) - (1 if -1 in y_pre else 0)
        y_pre = np.asarray(y_pre)

        # print out the clustering information
        print('{0} clusters:'.format(num_clusters))
        for i in range(0, num_clusters):
            print('-- Cluster {0}: {1} pts'.format(i, sum(y_pre == i)))

        # convert clusters to list of indices
        for cluster_label in range(0, num_clusters):
            clus = (y_pre == cluster_label)
            clusters.append([i for i, x in enumerate(clus) if x])

        # ----------------------------------------------------------------------------------
        # set the clusters in tx_cor
        for i, clus in enumerate(clusters):
            self.tx_cor[self.idx_space[clus], self.idx_time[clus]] = -(i+1)

        return clusters



    def expand_traj(self, clus, cluster_idx):
        """
        This function expands the data points in each cluster to include the full trajectory
        :param clus: index of pts
        :param cluster_idx: the index for this cluster
        :return: updated_clus, num_data_rows
        """
        clus = list(clus)
        updated_clus = []

        _, num_cols = self.tx_cor.shape

        # ----------------------------------------------------------------------------------
        # save the width and medians of each data row in 0~32 rows in the array
        # traj_data is a 32 x n list. Each row contains the time index at that row
        traj_data = [[] for i in xrange(self.paras['pir_res'][1])]
        widths = np.ones(self.paras['pir_res'][1])*np.nan
        medians = np.ones(self.paras['pir_res'][1])*np.nan
        percentile_lb = np.ones(self.paras['pir_res'][1])*np.nan
        percentile_ub = np.ones(self.paras['pir_res'][1])*np.nan

        # ----------------------------------------------------------------------------------
        # First determine if this cluster is merged by other clusters
        vals = self.tx_cor[self.idx_space[clus], self.idx_time[clus]]

        if sum(vals == -cluster_idx) == 0:
            # already merged by other clusters, skip
            print('Cluster {0} was merged by other clusters'.format(cluster_idx))
            return None
        elif sum(vals == -cluster_idx) != len(clus):
            print('Cluster {0} was partially merged by other clusters from {1} to {2}'.format(cluster_idx, len(clus),
                                                                                              sum(vals == -cluster_idx)))

        # ----------------------------------------------------------------------------------
        # expand each row first to include all consecutive data points in this row belonging to this trajectory
        expanded_row = []
        for x in xrange(0, self.paras['pir_res'][1]):

            # points in this row
            _idx = (self.idx_space[clus]==x)

            if sum( _idx ) != 0:
                # non empty row, expand
                t_idxs = self.idx_time[clus][_idx]
                pts_idx, _t_idxs = self.expand_row(x, t_idxs, cluster_idx)

                # ------------------------------------------------------------------
                # determine if this row of data is reasonable by checking its width
                cur_width = len(_t_idxs)
                avg_width = self.compute_expected_width(widths)

                # only pose the constraint if gets a valid average width
                if avg_width is None or \
                   self.paras['max_width_ratio_lb']*avg_width <= cur_width <= self.paras['max_width_ratio_ub']*avg_width:

                    # save the trajectory data, the width, and the median
                    traj_data[x] = _t_idxs
                    widths[x] = cur_width

                    if len(_t_idxs) != 0:
                        # print('debug: _t_idx:{0}, and median {1}'.format(_t_idxs, np.median(_t_idxs)))
                        medians[x] = int(np.round(np.median(_t_idxs)))
                        percentile_lb[x] = np.percentile(_t_idxs, self.paras['percentile_lb'])
                        percentile_ub[x] = np.percentile(_t_idxs, self.paras['percentile_ub'])

                    # add into cluster
                    updated_clus += pts_idx

                else:
                    # reset the row to be unlabeled
                    self.tx_cor[x, _t_idxs] = 0.0

                expanded_row.append(x)

        # print('cluster {0}, expanded rows {1}, widths: {2}, medians {3}'.format(cluster_idx, expanded_row, widths, medians))

        # if self.plot_progress:
        #     self._plot_clusters([updated_clus], title='cluster {0}'.format(cluster_idx), save_name=None,
        #                                           option='index', medians=medians)

        # ----------------------------------------------------------------------------------
        # discard this cluster if
        #   - the slope direction is not expected
        #   - 0 good row data

        if sum(~np.isnan(medians)) == 0 or self.compute_expected_slope(medians) >= 0:
            return None

        # ----------------------------------------------------------------------------------
        # expand to adjacent rows where previously no cluster was included
        # start from the smallest row index and expand downwards. If min row index is not 0, also expend upwards
        min_row_idx = np.min(expanded_row)

        # ------------------------------------------------
        # expand upward using these min
        if min_row_idx != 0:
            # expand the upper row
            for x in xrange(min_row_idx-1, -1, -1):
                if x not in expanded_row:

                    # ------------------------------------------------------------------
                    # find the min and max for searching data in the next row
                    exp_med = self.compute_expected_median_LR(medians, x)
                    avg_width = self.compute_expected_width(widths)
                    # min_t, max_t = (exp_med - int(np.ceil(avg_width/2.0))), (exp_med + int(np.ceil(avg_width/2.0)))
                    if avg_width is None:
                        avg_width = self.paras['default_width']
                    min_t, max_t = exp_med - avg_width, exp_med + avg_width

                    if self.plot_progress:
                        self._plot_progress([updated_clus], title='cluster {0}'.format(cluster_idx), save_name=None,
                                              option='index', cur_medians=medians, cur_x=x, cur_ts=[min_t, exp_med, max_t],
                                              cur_percentile_lb=percentile_lb, cur_percentile_ub=percentile_ub)

                    pts_idx, t_idxs = self.expand_adj_row(x, min_t, max_t, cluster_idx)

                    # ------------------------------------------------------------------
                    # determine if this row of data is reasonable, by
                    #   - median in [min_t, max_t]
                    #   - width is no longer than paras['max_width_ratio']*avg_width
                    if len(t_idxs) != 0:
                        # a non empty row
                        cur_median = int(np.round(np.median(t_idxs)))
                        cur_width = len(t_idxs)

                        # for negative slope, the median should be greater then current median as x decreases
                        min_med_t, max_med_t = exp_med - self.paras['median_tol'], \
                                               exp_med + avg_width*self.paras['max_width_ratio_ub']
                        if min_med_t <= cur_median <= max_med_t and \
                                        avg_width*self.paras['max_width_ratio_lb']<= cur_width \
                                        <= avg_width*self.paras['max_width_ratio_ub']:

                            # save the traj data, the width, and the median
                            traj_data[x] = t_idxs
                            widths[x] = len(t_idxs)
                            medians[x] = int(np.round(np.median(t_idxs)))
                            percentile_lb[x] = np.percentile(t_idxs, self.paras['percentile_lb'])
                            percentile_ub[x] = np.percentile(t_idxs, self.paras['percentile_ub'])

                            # add the points into new cluster
                            updated_clus += pts_idx
                            expanded_row.append(x)

                        else:
                            # print('debug. resetted row {0}'.format(x))
                            # reset this row as unlabeled
                            self.tx_cor[x, t_idxs] = 0.0

        # ------------------------------------------------
        # expand downward
        for x in xrange(min_row_idx+1, self.paras['pir_res'][1]):
            if x not in expanded_row:

                # find the min and max for searching data in the next row
                exp_med = self.compute_expected_median_LR(medians, x)
                avg_width = self.compute_expected_width(widths)

                if avg_width is None:
                        avg_width = self.paras['default_width']

                # min_t, max_t = (exp_med - int(np.ceil(avg_width/2.0))), (exp_med + int(np.ceil(avg_width/2.0)))
                min_t, max_t = exp_med - avg_width, exp_med + avg_width

                if self.plot_progress:
                    self._plot_progress([updated_clus], title='cluster {0}'.format(cluster_idx), save_name=None,
                                              option='index', cur_medians=medians, cur_x=x, cur_ts=[min_t, exp_med, max_t],
                                        cur_percentile_lb=percentile_lb, cur_percentile_ub=percentile_ub)

                pts_idx, t_idxs = self.expand_adj_row(x, min_t, max_t, cluster_idx)

                # ------------------------------------------------------------------
                # determine if this row of data is reasonable, by
                #   - median in [min_t, max_t]
                #   - width is no longer than paras['max_width_ratio']*avg_width
                if len(t_idxs) != 0:
                    # a non empty row
                    cur_median = int(np.round(np.median(t_idxs)))
                    cur_width = len(t_idxs)

                    # for negative slope, the median should be smaller then current median as x increases
                    min_med_t, max_med_t = exp_med - avg_width*self.paras['max_width_ratio_ub'], \
                                           exp_med + self.paras['median_tol']
                    if min_med_t <= cur_median <= max_med_t and \
                                    avg_width*self.paras['max_width_ratio_lb'] <= cur_width <= \
                                            avg_width*self.paras['max_width_ratio_ub']:

                        # save the traj data, the width, and the median
                        traj_data[x] = t_idxs
                        widths[x] = len(t_idxs)
                        if len(t_idxs) !=0 :
                            medians[x] = int(np.round(np.median(t_idxs)))
                            percentile_lb[x] = np.percentile(t_idxs, self.paras['percentile_lb'])
                            percentile_ub[x] = np.percentile(t_idxs, self.paras['percentile_ub'])

                        updated_clus += pts_idx
                        expanded_row.append(x)
                    else:
                        # reset this row as unlabeled
                        self.tx_cor[x, t_idxs] = 0.0

        # ----------------------------------------------------------------------------------
        print('-- Updated cluster {0} from {1} pts to {2}'.format(cluster_idx, len(clus), len(updated_clus)))
        # print('           traj data: {0}'.format(traj_data))
        # print('           widths: {0}'.format(widths))
        # print('           medians: {0}'.format(medians))

        # ----------------------------------------------------------------------------------
        # determine if this trajecotry is acceptable
        num_data_rows = sum(~np.isnan(medians))
        row_width = self.compute_expected_width(widths)
        slope = self.compute_expected_slope(medians)

        if num_data_rows >= self.paras['min_num_data_rows'] and \
            len(updated_clus) >= self.paras['min_inliers'] and \
                        row_width is not None and \
                        row_width >= self.paras['min_avg_width'] and slope < 0:

            # -------------------------------------------------------------
            # determine if the trajectory should be split: NOTE: here slope is t = slope*x + c. The speed is 1/slope
            slope_lb = self.compute_expected_slope(percentile_lb)
            slope_ub = self.compute_expected_slope(percentile_ub)

            if np.abs(1.0/slope_lb - 1.0/slope_ub) >= self.paras['max_slope_difference']:
                # split the trajectory at the medians
                clus_lb, clus_ub, widths_lb, widths_ub = self.split_clusters(medians, cluster_idx)

                print('$$$ Split trajectory {0} with {1} data points'.format(cluster_idx, len(updated_clus)))
                # ---------------------------------------------------------
                # append the left half to property
                self.all_traj.append(clus_lb)
                self.all_medians.append(percentile_lb)
                self.all_widths.append(widths_lb)
                # append None for percentiles
                self.all_percentile_lb.append(None)
                self.all_percentile_ub.append(None)
                self.all_traj_num_rows.append( sum( widths_lb!=0 ) )
                print('$$$      Saved trajectory {0} with {1} data rows {2} data points'.format(cluster_idx,
                                                                                                sum( widths_lb!=0 ),
                                                                                                len(clus_lb)))

                # ---------------------------------------------------------
                # append the left half to property
                self.all_traj.append(clus_ub)
                self.all_medians.append(percentile_ub)
                self.all_widths.append(widths_ub)
                # append None for percentiles
                self.all_percentile_lb.append(None)
                self.all_percentile_ub.append(None)
                self.all_traj_num_rows.append( sum( widths_ub!=0 ) )
                print('$$$     Saved trajectory {0} with {1} data rows {2} data points'.format(cluster_idx,
                                                                                               sum( widths_ub!=0 ),
                                                                                               len(clus_ub)))

            else:
                # no split, and save the result in property
                self.all_traj.append(updated_clus)
                self.all_medians.append(medians)
                self.all_widths.append(widths)
                self.all_percentile_lb.append(percentile_lb)
                self.all_percentile_ub.append(percentile_ub)
                self.all_traj_num_rows.append(num_data_rows)

                print('$$$ Saved trajectory {0} with {1} data rows'.format(cluster_idx, num_data_rows))

        else:
            # remove clusters by reset their values to 0.0
            self.tx_cor[ self.idx_space[updated_clus], self.idx_time[updated_clus] ] = 0.0
            print('$$$ Discard traj {0} with {1} data rows'.format(cluster_idx, num_data_rows))


    def split_clusters(self, medians, cluster_idx):
        """
        This function splits the cluster into two clusters at the medians.
        :param medians: a 32 x1 list of int, each int gives the position of the median (which may be .5 if number is even)
        :param cluster_idx: the index of the current cluster.
            After split, the left half will be marked as cluster_idx-0.1, while the right half cluster_idx+0.1
        :return:
            - the tx_cor will be marked with updated labels
            - clus_lb, clus_ub, the indexes of points in the list of self.pts_time, self.pts_space
        """
        clus_lb = []
        clus_ub = []

        num_rows, num_cols = self.tx_cor.shape

        widths_lb = np.zeros(num_rows)
        widths_ub = np.zeros(num_rows)

        for row in xrange(0, num_rows):
            if ~np.isnan(medians[row]):
                # split this row
                for col in xrange(0, num_cols):
                    if col <= medians[row] and self.tx_cor[row, col] == cluster_idx:
                        # assign to left cluster and update label, update width
                        clus_lb.append( self.mat2list[row, col] )
                        self.tx_cor[row, col] = cluster_idx - 0.1
                        widths_lb[row] += 1
                    elif col > medians[row] and self.tx_cor[row, col] == cluster_idx:
                        # assign to right cluster and update label, update width
                        clus_ub.append( self.mat2list[row, col] )
                        self.tx_cor[row, col] = cluster_idx + 0.1
                        widths_ub[row] += 1


        return clus_lb, clus_ub, widths_lb, widths_ub

    def compute_expected_median_LR(self, medians, x_idx):
        """
        This function computes the expected median at row x_idx by
            - first run linear regression from medians
            - then compute the expected median
        :param medians: a (32,) ndarray
        :param x_idx: int, [0,32)
        :return: int, shift if x index increment by 1
        """

        # ------------------------------------------------------------
        # get the not nan medians x: is space, y: is time
        # use linear regression only if there are at least two points
        if sum(~np.isnan(medians)) > 1:
            x = []
            y = []
            for r, v in enumerate(medians):
                if ~np.isnan((v)):
                    x.append(r)
                    y.append(v)

            _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x,y)
            exp_med = _slope*x_idx + _intercept
            return int(np.round(exp_med))

        else:
            # print('debug: medians: {0}'.format(medians))
            notnan_idx = np.where(~np.isnan(medians))[0][0]
            exp_med = self.paras['default_slope']*(x_idx - notnan_idx) + medians[notnan_idx]
            return int(np.round(exp_med))

    def compute_expected_slope(self, medians):
        """
        This function computes the expected median at row x_idx by
            - first run linear regression from medians
            - then compute the expected median
        :param medians: a (32,) ndarray
        :param x_idx: int, [0,32)
        :return: int, shift if x index increment by 1
        """

        # ------------------------------------------------------------
        # get the not nan medians x: is space, y: is time
        # use linear regression only if there are at least two points
        if sum(~np.isnan(medians)) > 1:
            x = []
            y = []
            for r, v in enumerate(medians):
                if ~np.isnan((v)):
                    x.append(r)
                    y.append(v)

            _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x,y)
            return _slope
        else:
            return self.paras['default_slope']

    def compute_expected_median(self, medians, x_idx):
        """
        This function computes the expected median at row x_idx by
            - first compute the average shift per row
            - compute the medians shifted from known medians in other rows
            - compute the median of medians as the expected median
        :param medians: a (32,) ndarray
        :param x_idx: int, [0,32)
        :return: int, shift if x index increment by 1
        """

        # ------------------------------------------------------------
        # compute the average shift
        shifts = []

        counter = 0.0
        pre, post = 0, 0

        while True:
            post += 1
            if post == self.paras['pir_res'][1]:
                break

            counter += 1.0
            if np.isnan(medians[post]):
                continue
            else:
                s = (post-pre)/counter

                if s>0:
                    shifts.append( s )

                # reset
                pre = post
                counter = 0.0

        avg_shift = int(np.round(np.median(shifts)))

        # ------------------------------------------------------------
        # compute the exp_medians from each row
        exp_medians = []
        for i, v in enumerate(medians):
            if ~np.isnan(v):
                exp_medians.append( v + (i-x_idx)*avg_shift )

        # ------------------------------------------------------------
        # compute the expected median as the median of medians, then round
        exp_med = int(np.round(np.median(exp_medians)))

        return exp_med

    def compute_expected_width(self, widths):
        """
        This function returns the expected widths
        :param widths: a list of 32 int, may contain np.nan
        :return:
        """
        valid_idx = (~np.isnan(widths)) & (widths != 0.0)

        if sum(valid_idx) >= 5:
            # If there was less than three rows, then simply use default width to avoid unreliable estimates
            valid_widths = widths[valid_idx]
            return int(np.round(np.median(valid_widths)))
        else:
            # otherwise, return None to not impose the constraint on width
            return None
            #return self.paras['default_width']

    def expand_row(self, x_idx, time_idxs, cluster_idx):
        """
        This function expands the current spatial row to its first notnan value and last notnan value
        :param x_idx: index of spatial position [0,32)
        :param time_idxs: list of index in time domain
        :param cluster_idx: the idx of the cluster to be set in the tx_cor matrix
        :return: list of additional pts indexes to be included as part of the row
        """
        add_t = []

        _, num_cols = self.tx_cor.shape
        min_t, max_t = np.min(time_idxs), np.max(time_idxs)

        # -----------------------------------------------------------------------
        # First abs(value) to make it as labeled
        for t in time_idxs:
            if self.tx_cor[x_idx, t] == -cluster_idx:
                self.tx_cor[x_idx, t] = cluster_idx
                add_t.append(t)
            else:
                pass
                # print('Warning: found ({0}, {1}) not belonging to cluster {2} with value {3}'.format(x_idx, t,
                #                                                                                      cluster_idx,
                #                                                                                      self.tx_cor[x_idx, t]))

        # -----------------------------------------------------------------------
        # get the points ahead, allow breaking by one point
        counter = 0
        for i in xrange(min_t-1, -1, -1):
            if ~np.isnan(self.tx_cor[x_idx, i]) and self.tx_cor[x_idx, i] <= 0.0:
                # a valid data point
                add_t.append(i)
                self.tx_cor[x_idx, i] = cluster_idx

                # reset counter
                counter = 0
            else:
                if counter == 1:
                    # stop once found nan or labeled point
                    break
                else:
                    # increment counter to mark encountered one break
                    counter += 1

        # -----------------------------------------------------------------------
        # get the points after
        counter = 0
        for i in xrange(max_t+1, num_cols):
            if ~np.isnan(self.tx_cor[x_idx, i]) and self.tx_cor[x_idx, i] <= 0.0:
                # a valid data point
                add_t.append(i)
                self.tx_cor[x_idx, i] = cluster_idx

                # reset counter of missing point
                counter = 0
            else:
                if counter == 1:
                    # stop once found nan or labeled point
                    break
                else:
                    # increment counter to mark encountered one break
                    counter += 1

        # -----------------------------------------------------------------------
        # convert the points into pts index
        if len(add_t) != 0:

            pts_idx = self.rowidx_to_ptsidx(add_t, x_idx)

            return pts_idx, add_t

        else:
            return [], []

    def expand_adj_row(self, x_idx, min_t_idx, max_t_idx, cluster_idx):
        """
        This function expand the trajectory to this adjacent row by
            - first include all data point within [min_t_idx, max_t_idx] in this row
            - then extend the row data to its consecutive neighbors ahead and after
        Similar to expand_row, it only expands to the data point that has label <= 0: unlabeled or initialized.
        :param x_idx: the row index to be expanded
        :param min_t_idx: the min t index to be included
        :param max_t_idx: the max t index to be included
        :param cluster_idx: the current cluster index
        :return: list of additional pts indexes to be included as part of the row
        """

        # ----------------------------------------------------------------------------------
        # make sure the min and max index are in range
        _, num_cols = self.tx_cor.shape

        if min_t_idx >= num_cols or max_t_idx < 0:
            return [], []
        else:
            min_t_idx = np.max([0, min_t_idx])
            max_t_idx = np.min([num_cols-1, max_t_idx])

        add_t = []
        # ----------------------------------------------------------------------------------
        # first include all data point within [min_t_idx, max_t_index]
        for i in xrange(min_t_idx, max_t_idx+1):
            if ~np.isnan(self.tx_cor[x_idx, i]) and self.tx_cor[x_idx, i] <= 0.0:
                # a valid data point
                add_t.append(i)
                self.tx_cor[x_idx, i] = cluster_idx

        # ----------------------------------------------------------------------------------
        if len(add_t) != 0:
            # expand this row to its first data point and last data point in the sequence
            _, num_cols = self.tx_cor.shape
            min_t, max_t = np.min(add_t), np.max(add_t)

            # get the points ahead
            for i in xrange(min_t-1, -1, -1):
                if ~np.isnan(self.tx_cor[x_idx, i]) and self.tx_cor[x_idx, i] <= 0.0:
                    # a valid data point
                    add_t.append(i)
                    self.tx_cor[x_idx, i] = cluster_idx
                else:
                    # stop once found nan or labeled point
                    break

            # get the points after
            for i in xrange(max_t+1, num_cols):
                if ~np.isnan(self.tx_cor[x_idx, i]) and self.tx_cor[x_idx, i] <= 0.0:
                    # a valid data point
                    add_t.append(i)
                    self.tx_cor[x_idx, i] = cluster_idx
                else:
                    # stop once found nan or labeled point
                    break

        # ----------------------------------------------------------------------------------
        # convert the t, x point index to list index
        if len(add_t) == 0:
            return [], []
        else:

            pts_idx = self.rowidx_to_ptsidx(add_t, x_idx)

            return pts_idx,  add_t

    def rowidx_to_ptsidx(self, t_idx, x):
        """
        This function converts the column index (t_idx) at row x to the point index in self.pts_time
        :param t_idx: [] list of int
        :param x: int, [0.32)
        :return: [] list of int
        """
        # pts_idx = []
        # for t in t_idx:
        #     _idx = (self.idx_time == t) & (self.idx_space == x)
        #
        #     if sum(_idx) != 1:
        #         raise Exception('Non-unique conversion of ({0}, {1}) to pts index'.format(x, t))
        #     else:
        #         pts_idx.append(np.where(_idx)[0][0])
        #
        # return pts_idx

        return list( self.mat2list[int(x), [int(i) for i in t_idx]] )

    def txidx_to_ptsidx(self, t, x):
        """
        This function converts the column index (t_idx) at row x to the point index in self.pts_time
        :param t_idx: [] list of int
        :param x: int, [0.32)
        :return: [] list of int
        """
        # _idx = (self.idx_time == int(t)) & (self.idx_space == int(x))
        #
        # if sum(_idx) != 1:
        #     raise Exception('Non-unique conversion of ({0}, {1}) to pts index with {1} duplicates'.format(t, x,
        #                                                                                                   sum(_idx)))
        # else:
        #     pts_idx = np.where(_idx)[0][0]

        return self.mat2list[int(x), int(t)]

    def rowcol_to_loc(self, row, col):
        """
        This function returns the nonlinear location of the row and col
        :param row:
        :param col:
        :return: time, space
        """
        return [self._t_grid[int(col)], self._x_grid[int(row)]]

    def find_minmax_in_row(self, x_idx, cluster_idx):
        """
        This function returns the min and max column index (time) of the data row belong to this cluster
        :param x_idx: the row index [0,32)
        :param cluster_idx: the cluster index
        :return:
        """
        _b_idx = (self.tx_cor[x_idx, :] == cluster_idx)

        if sum( _b_idx ) == 0:
            # empty row, expand
            return None
        else:
            idx = np.where(_b_idx)[0]
            return idx[0], idx[-1]

    def find_times_in_row(self, x_idx, cluster_idx):
        """
        This function returns the column index (time) of the data row belonging to this cluster
        :param x_idx: the row index [0,32)
        :param cluster_idx: the cluster index
        :return:
        """
        _b_idx = (self.tx_cor[x_idx, :] == cluster_idx)

        if sum( _b_idx ) == 0:
            # empty row, expand
            return None
        else:
            idx = np.where(_b_idx)[0]
            return list(idx)

    def _new_nonlinear_transform(self):
        """
        This function performs the nonlinear transform to the norm_df data
        :return: space grid
        """
        _dup = self.paras['pir_res'][0]
        d_theta = (self.paras['pir_fov'] / 15) * np.pi / 180.0

        alpha = np.tan( self.paras['pir_fov_offset']*np.pi/180.0)

        x_grid_pos = []
        for i in range(0, 16):
            for d in range(0, _dup):
                # duplicate the nonlinear operator for vec
                x_grid_pos.append(np.tan(alpha + i * d_theta ) / self.paras['tx_ratio'])
        x_grid_pos = np.asarray(x_grid_pos)

        x_grid_neg = np.sort(-deepcopy(x_grid_pos))

        x_grid = np.concatenate([x_grid_neg, x_grid_pos])

        return -x_grid

    def _plot_clus(self, clusters, title='', save_name=None, option='index',
                       medians=None, cur_x=None, cur_ts=None, original_clusters=None):
        """
        This function is only used for debugging. It visualizes how the subclusters are obtained using kernel density
        estimation
        :param clusters
        :return:
        """

        # plot the initial figure
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Axis 1 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ----------------------------------------------------
        # plot ax0: plot the all data points
        if option == 'index':
            ax0.scatter(self.idx_time, self.idx_space, color='0.6')  # all points
        else:
            ax0.scatter(self.pts_time, self.pts_space, color='0.6')  # all points

        # plot data points for each cluster
        colors = itertools.cycle(['b', 'g',  'c',
                                  'darkorange', 'olive', 'deepskyblue', 'fuchsia', 'deeppink'])
        for pt_idx in clusters:
            c = next(colors)
            if option == 'index':
                ax0.scatter(self.idx_time[pt_idx], self.idx_space[pt_idx], color=c)

                # scatter the medians to check if it is progressing as expected
                if medians is not None:
                    for i, v in enumerate(medians):
                        if ~np.isnan(v):
                            ax0.scatter(v, i, color='r')
                if cur_x is not None:
                    min_t, exp_med, max_t = cur_ts

                    ax0.scatter(exp_med, cur_x, color='r')
                    ax0.scatter(min_t, cur_x, marker='<', color='r', s=100)
                    ax0.scatter(max_t, cur_x, marker='>', color='r', s=100)

            else:
                ax0.scatter(self.pts_time[pt_idx], self.pts_space[pt_idx], color=c)

        ax0.set_title('{0}'.format(title), fontsize=28)

        if option == 'index':
            # ax0.set_xlabel('Time index', fontsize=24)
            ax0.set_ylabel('Space index', fontsize=24)
            ax0.set_xlim([0, len(self.t_grid)])
            ax0.set_ylim([0, self.paras['pir_res'][1]])

        else:
            # ax0.set_xlabel('Time (s)', fontsize=24)
            ax0.set_ylabel('Relative space', fontsize=24)
            ax0.set_xlim([0, self.window_s])
            ax0.set_ylim([np.min(self.pts_space), np.max(self.pts_space)])
        ax0.tick_params(axis='both', which='major', labelsize=18)

        # ----------------------------------------------------
        # ax1: plot the original clusters
        if original_clusters is not None:
            if option == 'index':
                ax1.scatter(self.idx_time, self.idx_space, color='0.6')  # all points
            else:
                ax1.scatter(self.pts_time, self.pts_space, color='0.6')  # all points

            colors = itertools.cycle(['b', 'g',  'c',
                                      'darkorange', 'olive', 'deepskyblue', 'fuchsia', 'deeppink'])
            for pt_idx in original_clusters:
                c = next(colors)
                if option == 'index':
                    ax1.scatter(self.idx_time[pt_idx], self.idx_space[pt_idx], color=c)
                else:
                    ax1.scatter(self.pts_time[pt_idx], self.pts_space[pt_idx], color=c)

            ax1.set_title('DBSCAN clusters'.format(title), fontsize=28)

            if option == 'index':
                # ax1.set_xlabel('Time index, original clusters', fontsize=24)
                ax1.set_ylabel('Space index', fontsize=24)
                ax1.set_xlim([0, len(self.t_grid)])
                ax1.set_ylim([0, self.paras['pir_res'][1]])

            else:
                # ax1.set_xlabel('Time (s)', fontsize=24)
                ax1.set_ylabel('Relative space', fontsize=24)
                ax1.set_xlim([0, self.window_s])
                ax1.set_ylim([np.min(self.pts_space), np.max(self.pts_space)])
            ax1.tick_params(axis='both', which='major', labelsize=18)

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.show()
            # plt.draw()


    def _plot_progress(self, cur_clus, title='', save_name=None, option='index',
                       cur_medians=None, cur_x=None, cur_ts=None, original_clusters=None,
                       cur_percentile_lb=None, cur_percentile_ub=None):
        """
        This function is only used for debugging. It visualizes how the subclusters are obtained using kernel density
        estimation
        :param clusters
        :return:
        """

        # plot the initial figure
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Axis 1 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ----------------------------------------------------
        # plot ax0: plot the all data points
        if option == 'index':
            ax0.scatter(self.idx_time, self.idx_space, color='0.6')  # all points
        else:
            ax0.scatter(self.pts_time, self.pts_space, color='0.6')  # all points

        # ----------------------------------------------------
        # plot the labeled clusters
        if len(self.all_traj) != 0:
            colors = itertools.cycle(['g',  'c',
                                      'darkorange', 'olive', 'deepskyblue', 'fuchsia', 'deeppink'])
            for i, pt_idx in enumerate(self.all_traj):
                c = next(colors)
                if option == 'index':
                    ax0.scatter(self.idx_time[pt_idx], self.idx_space[pt_idx], color=c)

                    # scatter the medians of labeled trajectories
                    for x, t in enumerate(self.all_medians[i]):
                        if ~np.isnan(t):
                            ax0.scatter(t, x, color='r')

                else:
                    ax0.scatter(self.pts_time[pt_idx], self.pts_space[pt_idx], color=c)

        # ----------------------------------------------------
        # plot the current trajectory
        c = 'b'
        if option == 'index':
            ax0.scatter(self.idx_time[cur_clus], self.idx_space[cur_clus], color=c)

            # scatter the medians to check if it is progressing as expected
            if cur_medians is not None:
                for i, v in enumerate(cur_medians):
                    if ~np.isnan(v):
                        ax0.scatter(v, i, color='r')

            # scatter the percentiles to check if it is going as expected
            if cur_percentile_lb is not None:
                for i, v in enumerate(cur_percentile_lb):
                    if ~np.isnan(v):
                        ax0.scatter(v, i, color='r', marker='<')

            if cur_percentile_ub is not None:
                for i, v in enumerate(cur_percentile_ub):
                    if ~np.isnan(v):
                        ax0.scatter(v, i, color='r', marker='>')

            if cur_x is not None:
                min_t, exp_med, max_t = cur_ts

                ax0.scatter(exp_med, cur_x, color='r')
                ax0.scatter(min_t, cur_x, marker='<', color='r', s=100)
                ax0.scatter(max_t, cur_x, marker='>', color='r', s=100)

        else:
            ax0.scatter(self.pts_time[cur_clus], self.pts_space[cur_clus], color=c)


        # ----------------------------------------------------
        ax0.set_title('{0}'.format(title), fontsize=28)

        if option == 'index':
            # ax0.set_xlabel('Time index', fontsize=24)
            ax0.set_ylabel('Space index', fontsize=24)
            ax0.set_xlim([0, len(self.t_grid)])
            ax0.set_ylim([0, self.paras['pir_res'][1]])

        else:
            # ax0.set_xlabel('Time (s)', fontsize=24)
            ax0.set_ylabel('Relative space', fontsize=24)
            ax0.set_xlim([0, self.window_s])
            ax0.set_ylim([np.min(self.pts_space), np.max(self.pts_space)])
        ax0.tick_params(axis='both', which='major', labelsize=18)

        # ----------------------------------------------------
        # ax1: plot the original clusters
        if original_clusters is not None:
            if option == 'index':
                ax1.scatter(self.idx_time, self.idx_space, color='0.6')  # all points
            else:
                ax1.scatter(self.pts_time, self.pts_space, color='0.6')  # all points

            colors = itertools.cycle(['b', 'g',  'c',
                                      'darkorange', 'olive', 'deepskyblue', 'fuchsia', 'deeppink'])
            for pt_idx in original_clusters:
                c = next(colors)
                if option == 'index':
                    ax1.scatter(self.idx_time[pt_idx], self.idx_space[pt_idx], color=c)
                else:
                    ax1.scatter(self.pts_time[pt_idx], self.pts_space[pt_idx], color=c)

            ax1.set_title('DBSCAN clusters'.format(title), fontsize=28)

            if option == 'index':
                # ax1.set_xlabel('Time index, original clusters', fontsize=24)
                ax1.set_ylabel('Space index', fontsize=24)
                ax1.set_xlim([0, len(self.t_grid)])
                ax1.set_ylim([0, self.paras['pir_res'][1]])

            else:
                # ax1.set_xlabel('Time (s)', fontsize=24)
                ax1.set_ylabel('Relative space', fontsize=24)
                ax1.set_xlim([0, self.window_s])
                ax1.set_ylim([np.min(self.pts_space), np.max(self.pts_space)])
            ax1.tick_params(axis='both', which='major', labelsize=18)

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.show()
            # plt.draw()


    def _plot_fitted_clusters(self, clusters=None, title='', save_name=None):
        """
        This function is only used for debugging. It visualizes how the subclusters are obtained using kernel density
        estimation
        :param clusters
        :return:
        """

        # plot the initial figure
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        # Axis 1 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ----------------------------------------------------
        # plot ax0: plot the all data points
        ax0.scatter(self.pts_time, self.pts_space, color='0.6')  # all points

        # ----------------------------------------------------
        # plot clusters
        colors = itertools.cycle(['g', 'b', 'c',
                                  'darkorange', 'olive', 'deepskyblue', 'fuchsia', 'deeppink'])
        if clusters is None:
            clusters = self.all_traj

        # plot the saved labeled clusters
        for i, pt_idx in enumerate(clusters):
            c = next(colors)

            # ----------------------------------------------------
            # scatter all trajectory data point
            ax0.scatter(self.pts_time[pt_idx], self.pts_space[pt_idx], color=c)

            # ----------------------------------------------------
            # run linear regression through the medians
            line = self.run_lr(self.all_medians[i])
            _slope, _intercept = 1.0/line[0], -line[1]/line[0]

            # medians_tx = []
            # for x, t in enumerate(self.all_medians[i]):
            #     if ~np.isnan(t):
            #         medians_tx.append( self.rowcol_to_loc(x, t) )
            # medians_tx = np.asarray(medians_tx)
            # ax0.scatter(medians_tx[:,0], medians_tx[:,1], color=c, edgecolor='k' )
            #
            # # linear regression through the medians and plot the line
            # x, t = medians_tx[:,1], medians_tx[:,0]
            # # t = _slope*s + _intercept
            # _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x, t)

            # plot the fitted linear line through the medians
            x_lim = [self._x_grid[0], self._x_grid[-1]]
            t_val = np.array(x_lim)*_slope + _intercept

            ax0.plot(t_val, x_lim, linewidth=2, color=c)

            # ----------------------------------------------------
            # compute the percentile lb of labeled trajectories
            if self.all_percentile_lb[i] is not None:
                # linear regression through the lower percentiles and plot the line
                line = self.run_lr(self.all_percentile_lb[i])
                _slope, _intercept = 1.0/line[0], -line[1]/line[0]

                # percentile_lb_tx = []
                # for x, t in enumerate(self.all_percentile_lb[i]):
                #     if ~np.isnan(t):
                #         percentile_lb_tx.append( self.rowcol_to_loc(x, t) )
                # percentile_lb_tx = np.asarray(percentile_lb_tx)
                # ax0.scatter(percentile_lb_tx[:,0], percentile_lb_tx[:,1], color=c,  marker='<' , edgecolor='k')
                #
                # x, t = percentile_lb_tx[:,1], percentile_lb_tx[:,0]
                # # t = _slope*s + _intercept
                # _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x, t)

                # plot the fitted linear line through the medians
                x_lim = [self._x_grid[0], self._x_grid[-1]]
                t_val = np.array(x_lim)*_slope + _intercept

                ax0.plot(t_val, x_lim, linewidth=2, color=c, linestyle='--')

            # ----------------------------------------------------
            # compute the percentile ub of labeled trajectories
            if self.all_percentile_ub[i] is not None:
                # linear regression through the lower percentiles and plot the line
                line = self.run_lr(self.all_percentile_ub[i])
                _slope, _intercept = 1.0/line[0], -line[1]/line[0]

                # percentile_ub_tx = []
                # for x, t in enumerate(self.all_percentile_ub[i]):
                #     if ~np.isnan(t):
                #         percentile_ub_tx.append( self.rowcol_to_loc(x, t) )
                # percentile_ub_tx = np.asarray(percentile_ub_tx)
                # ax0.scatter(percentile_ub_tx[:,0], percentile_ub_tx[:,1], color=c,  marker='>' , edgecolor='k')
                #
                # # linear regression through the medians and plot the line
                # x, t = percentile_ub_tx[:,1], percentile_ub_tx[:,0]
                # # t = _slope*s + _intercept
                # _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x, t)

                # plot the fitted linear line through the medians
                x_lim = [self._x_grid[0], self._x_grid[-1]]
                t_val = np.array(x_lim)*_slope + _intercept

                ax0.plot(t_val, x_lim, linewidth=2, color=c, linestyle='--')


        # ----------------------------------------------------
        ax0.set_title('{0}'.format(title), fontsize=28)
        # ax0.set_xlabel('Time (s)', fontsize=24)
        ax0.set_ylabel('Relative space', fontsize=24)
        ax0.set_xlim([0, self.window_s])
        ax0.set_ylim([np.min(self.pts_space), np.max(self.pts_space)])
        ax0.tick_params(axis='both', which='major', labelsize=18)

        # ----------------------------------------------------
        # ax1: plot the original clusters

        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            # plt.show()
            plt.draw()





def time2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def str2time(dt_str="%Y-%m-%d %H:%M:%S.%f"):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")


def time2str_file(dt):
    return dt.strftime("%Y%m%d_%H%M%S_%f")


def str2time_file(dt_str="%Y%m%d_%H%M%S_%f"):
    return datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")


def plot_hist( arrs, labels, title='', xlabel='', fontsizes = (22, 18, 16), xlim=None, ylim=None, text_loc=None):
        """
        This function plots the histogram of the list of arrays
        :param arrs: list of arrs
        :param labels: label for each array
        :param title: title
        :param xlabel: xlabel
        :param fontsizes: (title, label, tick)
        :return:
        """
        plt.figure(figsize=(11.5,10))

        colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
        text = []
        y_max = 0.0
        std_max = 0.0
        for i, d in enumerate(arrs):
            c = next(colors)
            mu, std = np.mean(d), np.std(d)
            if labels is not None:
                n, bins, patches = plt.hist(d, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
            else:
                n, bins, patches = plt.hist(d, 50, normed=1, facecolor=c, alpha=0.75)
            plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                     linewidth=2)
            if labels is not None:
                text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))
            else:
                text.append('Bias: {0:.2f}\nStd : {1:.2f}'.format(mu, std))
            # text.append('Mean: {0:.2f}\nStandard deviation: {1:.2f}'.format(mu, std))

            y_max = np.max([y_max, np.max(n)])
            std_max = np.max([std_max, std])

        text_str = '\n'.join(text)
        if text_loc is None:
            plt.text(mu-4*std_max, y_max*1.0, text_str, fontsize=fontsizes[1])
        else:
            plt.text(text_loc[0], text_loc[1], text_str, fontsize=fontsizes[1])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim(0, y_max*1.2)
        else:
            plt.ylim(ylim)

        if labels is not None: plt.legend()
        plt.xlabel(xlabel, fontsize=fontsizes[1])
        plt.ylabel('Distribution', fontsize=fontsizes[1])
        plt.title(title, fontsize=fontsizes[0])
        plt.tick_params(axis='both', which='major', labelsize=fontsizes[2])
        plt.draw()


def match_lists(list1, list2, dt):
        """
        This function matches items in two lists: two items are matched if they are separated by less than dt.
        :param list1: a list of datetime
        :param list2: a list of datetime
        :param dt: float, seconds
        :return: matches: a list of tuples; first from list1, second from list2
            (item1, item2), where item1 is [dt, idx], where idx is the idx in list1
            (item1, None), no match in list2
            (None, item2), no match in list1
        """
        l = []
        for i, t in enumerate(list1):
            l.append([t, i, 0])
        for i, t in enumerate(list2):
            l.append([t, i, 1])

        l.sort(key=lambda x: x[0])

        matches = []
        group = [l[0]]
        for i in xrange( len(l)-1 ):
            if abs((l[i][0]-l[i+1][0]).total_seconds()) <= dt and l[i][2] != l[i+1][2]:
                group.append(l[i+1])
            else:
                if len(group) == 1:
                    # One item, did not find match in the other list, find which list it is in
                    if group[0][2] == 0: matches.append( (group[0][0:2], None) )
                    else: matches.append( (None, group[0][0:2]) )
                elif len(group) == 2:
                    # two items, one from each list
                    if group[0][2] == 0: matches.append( (group[0][0:2], group[1][0:2]) )
                    else: matches.append( (group[1][0:2], group[0][0:2]) )
                elif len(group) == 3:
                    # three items, 010 or 101, find which is closer to the center one
                    if abs((group[1][0]-group[0][0]).total_seconds()) <= abs((group[1][0]-group[2][0]).total_seconds()):
                        # group to 01,0 or 10,1
                        if group[1][2] == 0:
                            matches.append( (group[1][0:2], group[0][0:2]) )
                            matches.append( (None, group[2][0:2]) )
                        else:
                            matches.append( (group[0][0:2], group[1][0:2]) )
                            matches.append( (group[2][0:2], None) )
                    else:
                        # group to 0,10 or 1,01
                        if group[1][2] == 0:
                            matches.append( (group[1][0:2], group[2][0:2]) )
                            matches.append( (None, group[0][0:2]) )
                        else:
                            matches.append( (group[2][0:2], group[1][0:2]) )
                            matches.append( (group[0][0:2], None) )
                elif len(group) == 4:
                    # 1010 or 0101, break into 10,10 or 01,01
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                elif len(group) == 5:
                    # 01010, or 10101, group to 01,01,0 or 10,10,1
                    # This may not be the optimal solution
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                        matches.append( (group[4][0:2], None) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                        matches.append( (None, group[4][0:2]) )
                elif len(group) == 6:
                    # 101010 or 010101, group into 10,10,10 or 01,01,01
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                        matches.append( (group[4][0:2], group[5][0:2]) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                        matches.append( (group[5][0:2], group[4][0:2]) )
                else:
                    print('Warning: {0} items are grouped together: {1}'.format(len(group), np.array(group)[:,2]))

                group = [l[i+1]]

        # The last group
        if len(group) == 1:
            # One item, did not find match in the other list, find which list it is in
            if group[0][2] == 0: matches.append( (group[0][0:2], None) )
            else: matches.append( (None, group[0][0:2]) )
        elif len(group) == 2:
            # two items, one from each list
            if group[0][2] == 0: matches.append( (group[0][0:2], group[1][0:2]) )
            else: matches.append( (group[1][0:2], group[0][0:2]) )
        elif len(group) == 3:
            # three items, 010 or 101, find which is closer to the center one
            if abs((group[1][0]-group[0][0]).total_seconds()) <= abs((group[1][0]-group[2][0]).total_seconds()):
                # group to 01,0 or 10,1
                if group[1][2] == 0:
                    matches.append( (group[1][0:2], group[0][0:2]) )
                    matches.append( (None, group[2][0:2]) )
                else:
                    matches.append( (group[0][0:2], group[1][0:2]) )
                    matches.append( (group[2][0:2], None) )
            else:
                # group to 0,10 or 1,01
                if group[1][2] == 0:
                    matches.append( (group[1][0:2], group[2][0:2]) )
                    matches.append( (None, group[0][0:2]) )
                else:
                    matches.append( (group[2][0:2], group[1][0:2]) )
                    matches.append( (group[0][0:2], None) )
        elif len(group) == 4:
            # 1010 or 0101
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
        elif len(group) == 5:
            # This may not be the optimal solution
            # 01010, or 10101, group to 01,01,0 or 10,10,1
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
                matches.append( (group[4][0:2], None) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
                matches.append( (None, group[4][0:2]) )
        elif len(group) == 6:
            # 101010 or 010101, group into 10,10,10 or 01,01,01
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
                matches.append( (group[4][0:2], group[5][0:2]) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
                matches.append( (group[5][0:2], group[4][0:2]) )
        else:
            print('Warning: {0} items are grouped together'.format(len(group)))


        return matches





if __name__ == '__main__':
    main()

