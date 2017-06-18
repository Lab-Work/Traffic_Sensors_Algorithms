import cv2

print('Imported OpenCV version {0}'.format(cv2.__version__))
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
__version__ = "2.0"
__email__ = 'yli171@illinois.edu'

"""
This file contains all classes requried for vehicle detection and speed estimation. See details in paper:


Each vehicle structure: veh = {}
    'line': (k,c), s = kt + c
    'tol': float, [-tol, tol] is used to define the inliers
    'inlier_idx': ndarray of int the inlier indices
    'r2': the r2 of the fitting
    'dens': the density of the trace
    'residuals': the residual of the fitting
    'sigma': the std of the residual
    'num_inliers': the total number of inliers including those overlapping points

    'distance': m, the distance of the vehicle to the sensor
    'speed': mph, the speed of the vehicle
    'valid': True or False; valid if using the actual ultrasonic sensor reading

    't_in': datetime, the enter time
    't_out': datetime, the exit time
    't_left': datetime, time that the vehicle hit the left FOV boundary of PIR (top positive space value)
    't_right': datetime, time that the vehicle hit the right FOV boundary of PIR (bottome negative space value)

    'detection_window': (datetime, datetime), the initial and end time of this detection window that detects this veh
    'captured_percent': [0,1], the percentage of the trace being captured
    'captured_part': 'head', 'body', 'tail', 'full', which part of the trace is captured in this window

    'inliers': [[datetime, space], []], n x 2 ndarray, where n=len('inlier_idx')
"""


# ==================================================================================================================
# Some shared utility functions
# ==================================================================================================================
def time2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def str2time(dt_str="%Y-%m-%d %H:%M:%S.%f"):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")


def time2str_file(dt):
    return dt.strftime("%Y%m%d_%H%M%S_%f")


def str2time_file(dt_str="%Y%m%d_%H%M%S_%f"):
    return datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")


def print_loop_status(msg, i, total_iter):
    """
    This function prints the loop status
    :param i: the current loop counter
    :param total_iter: the total number of iterations
    :return:
    """
    sys.stdout.write('\r')
    sys.stdout.write('{0} {1}/{2}'.format(msg, i, total_iter))
    sys.stdout.flush()


def in_conf_intrvl(data, prob, mu, sigma):
    v_thres_u = stats.norm.ppf(1 - (1 - prob) / 2.0, mu, sigma)
    v_thres_l = mu - (v_thres_u - mu)

    return (data >= v_thres_l) & (data <= v_thres_u)


def in_2sigma(data, mu, sigma):
    return (data >= mu - 2.0 * sigma) & (data <= mu + 2.0 * sigma)


def f(B, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0] * x + B[1]


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ==================================================================================================================
# Top level traffic sensor algorithm
# ==================================================================================================================
class TrafficSensorAlg:
    """
    This class performs vehicle detection and speed estimation on normalized data set in pandas DataFrame structure.
    """

    def __init__(self, r2_thres=0.2, dens_thres=0.5, min_dt=0.2, pir_res=(4, 32), sampling_freq=64):
        """
        Initialize the algorithm with key parameters
        :param pir_res: the resolution of the PIR camera
        :param min_dt: the minimum width of trace (in seconds), i.e., the average time the vehicle is visible to a pixel
        :param dens_thres: the threshold for the density of supporting points in a converged trace model. Accept if
            greater than the threshold
        :param sampling_freq: the sampling freqeuncy of the PIR data.
        :return: initialized parameters
        """
        self.paras = OrderedDict()

        self.paras['pir_res'] = pir_res
        self.paras['sampling_freq'] = sampling_freq

        # -------------------------------------------------------------------------------------------------------
        # Initialize the default parameters which influence the performance but should be robust
        # -------------------------------------------------------------------------------------------------------

        ## For iterative regression
        # r2 describes the goodness of fit of a model to a trace. It is used to determine the tolerance in the next
        # iteration: if > TH_r2, expand; else contract
        self.paras['TH_r2'] = r2_thres
        # ratio multiplied by sigma is the new tolerance, 2.5 leads to expansion from bad fitting
        self.paras['expansion_ratio'] = 2.5
        # ratio multiplied by sigma is the new tolerance, 2.0 leads to contraction from bad fitting
        self.paras['contraction_ratio'] = 2.0
        # the stop criteria for determining the convergence of a model, the change of slope and intercept
        # 0.002 corresponds to 0.2 mph in further lane and 0.1 mph in closer lane
        self.paras['TH_stop'] = (0.002, 0.01)
        # maximum number of iterations for fitting a model
        self.paras['max_iter'] = 100
        # minimum number of inliers to perform linear fitting
        self.paras['min_inliers'] = 5
        # maximum added new point to stop the iteration, i.e., do NOT stop if there is more new pts added even if the
        # new model has negligible change from the old model
        self.paras['min_added_pts'] = 16

        ## Accept the model if >= dens_thres and >= min_pts
        # the density threshold: if > thres, accept the model
        self.paras['TH_dens'] = dens_thres
        # minimum number of supporting points for a model; reject the model if < min_pts
        self.paras['min_inliers'] = int(round(min_dt * pir_res[1] * sampling_freq * dens_thres))
        # Clean vehicle parameters
        # if the entering or exiting time of two vehicles are separated by dt_s seconds, then considered as two vehicles
        # merge vehicles that are too close ts1-ts2 <= dt_s and te1-te2<= dt_s
        self.paras['min_headway'] = 0.3
        # same vehicle threshold, same vehicle if overlapping percentage of inliers of two vehicle exceeds the threshold
        self.paras['TH_same_veh_overlapping'] = 0.5

        ## Ultrasonic sensor data filtering
        # the in and out space boundary of ultrasonic sensor fov in the time-space domain with ratio_tx = 6.0
        self.paras['ultra_fov_left'] = 0.16
        self.paras['ultra_fov_right'] = -0.16

        # the default distance if no ultrasonic reading is observed, only used if no historical median available
        self.paras['d_default'] = 4.0
        # if distance data is greater then no_ultra_thres, then there is no ultrasonic sensor reading
        self.paras['TH_no_ultra'] = 10.0
        # false positive threshold, if a reading is below self.fp_thres, then it is a false positive reading, and the
        # data with in i-self.fp_del_start: j+ self.fp_del_end will be replaced by 11
        self.paras['TH_ultra_fp'] = 2.5
        self.paras['ultra_fp_pre'] = 3
        self.paras['ultra_fp_post'] = 10

        # initialize clusters using DBSCAN, parameters for DBSCAN
        # DBSCAN_r: the radius of a point, 1/64 = 0.015625
        self.paras['DBSCAN_r'] = 0.05
        # DBSCAN_min_pts: the number of point within the radius for a point being considered as core
        self.paras['DBSCAN_min_pts'] = 30

        # For splitting clusters using kernel density estimation
        self.paras['KD_top_n'] = 8
        self.paras['KD_min_split_pts'] = 100
        self.paras['KD_min_init_pts'] = 10
        self.paras['speed_res'] = 5

        # for nonlinear transform
        self.paras['pir_fov'] = 53.0
        self.paras['pir_fov_offset'] = 3.0
        self.paras['tx_ratio'] = 6.0

        # -------------------------------------------------------------------------------------------------------
        # Save teh result in vehs
        # -------------------------------------------------------------------------------------------------------
        self.dists = []
        self.vehs = []

    def run(self, norm_df, TH_det=600, window_s=2.0, step_s=1.0, speed_range=(1, 50),
            plot_final=True, plot_debug=False, save_dir='./'):
        """
        This function runs the vehicle detection and speed estimation algorithm on the normalized data norm_df in a
        sliding window fashion with overlapping.
        :param norm_df: the pandas DataFrame data structure
        :param TH_det: the threshold for determining if there are vehicles present in the window
        :param window_s: seconds, the window size
        :param step_s: seconds, moving windows by steps
        :param speed_range: (1,50) mph, the range of speeds to be estimated
        :param plot_final: True or False, if want to plot the final estimates for each time window
        :param plot_debug: True or False, if want to plot the process, including splitting the clusters
        :param save_dir: directory for saving the results
        :return:
        """

        # -------------------------------------------------------------------------------------------------------
        # Detect the vehicle using VehDet class
        windows = self.detect_vehs(norm_df, TH_det=TH_det, window_s=window_s, step_s=step_s)

        # -------------------------------------------------------------------------------------------------------
        # Speed estimation
        if not exists(save_dir): os.makedirs(save_dir)

        for win in windows:
            est = SpeedEst(norm_df=norm_df.ix[win[0]:win[1]], paras=self.paras, window_s=window_s,
                           speed_range=speed_range,
                           plot_final=plot_final, plot_debug=plot_debug, save_dir=save_dir)
            vehs_in_win = est.estimate_speed()

            for veh in vehs_in_win:
                # register the vehicle to self.veh list
                self._register_veh(veh, speed_range=speed_range)

        # -------------------------------------------------------------------------------------------------------
        # Save the final detection result
        np.save(save_dir + 'detected_vehs.npy', self.vehs)
        self._save_det_vehs_txt(save_dir, 'detected_vehs.txt')
        self._save_paras(save_dir, 'paras.txt')

    def detect_vehs(self, norm_df, TH_det=600, window_s=2.0, step_s=1.0):
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
        pir_data.values[np.isnan(pir_data.values)] = 0.0

        # compute the energy in each 0.5 window
        dw = timedelta(seconds=window_s)
        ds = timedelta(seconds=step_s)
        t_s = pir_data.index[0]
        t_e = pir_data.index[0] + dw

        veh_t = []
        veh_e = []
        wins = []

        while t_e <= pir_data.index[-1]:
            e = np.sum(pir_data.ix[(pir_data.index >= t_s) & (pir_data.index <= t_e)].values)
            veh_t.append(t_s)
            veh_e.append(e)
            t_s = t_s + ds
            t_e = t_s + dw

            if e >= TH_det:
                wins.append((t_s, t_e))
        return wins

    def convert_vehs_to_txt(self, vehs, save_dir, save_name):
        """
        This function converts the vehs dict to the txt file containing [t_in, t_out, distance, speed, valid]
        :param vehs:
        :param save_dir:
        :param save_name:
        :return:
        """
        with open(save_dir+save_name, 'w+') as f:
            f.write('t_in, t_out, dist (m), speed (mph), estimated_dist\n')
            for veh in vehs:
                f.write('{0},{1},{2},{3},{4}\n'.format(veh['t_in'], veh['t_out'], veh['distance'],
                                                       veh['speed'], veh['valid']))

    def _register_veh(self, veh, speed_range=None):
        """
        This function registers the vehicle detected in each time window. In addition:
            - It checks if this vehicle has been detected in previous windows,
            - It updates its estimated speed (when ultrasonic reading is not available) using historical median
            - It caps the speed to speed_range
        :param veh: the vehicle to be registered
        :param speed_range: (-50,-1) mph in one direction or (1,50) mph in the other direction
        :return:
        """
        if len(self.vehs) == 0:

            # update the speeds using median distance
            if veh['valid'] is False and len(self.dists) != 0:
                m_d = np.median(self.dists)
                veh['speed'] = veh['speed'] * m_d / veh['distance']
                veh['distance'] = m_d
            else:
                self.dists.append(veh['distance'])

            # cap the speed to the limit
            if speed_range is not None:
                if veh['speed'] < speed_range[0]: veh['speed'] = speed_range[0]
                elif veh['speed'] > speed_range[1]: veh['speed'] = speed_range[1]

            self.vehs.append(veh)

        else:
            for i, old_v in enumerate(self.vehs):
                if self._same_veh(old_v, veh):
                    # replace the old estimate if the new estimate is better
                    if old_v['captured_percent'] < veh['captured_percent']:
                        # update the registered vehicles
                        print('######################## Updated vehicle entering at {0}\n'.format(veh['t_in']))
                        # also combine all data points from two vehicles
                        veh['inliers'] += old_v['inliers']

                        # update the speed using median distance
                        if veh['valid'] is True:
                            self.dists.append(veh['valid'])
                        else:
                            if old_v['valid'] is True:
                                # use the measured distance
                                m_d = old_v['distance']
                            else:
                                m_d = np.median(self.dists)

                            veh['speed'] = veh['speed'] * m_d / veh['distance']
                            veh['distance'] = m_d

                        # cap the speed to the limit
                        if speed_range is not None:
                            if veh['speed'] < speed_range[0]: veh['speed'] = speed_range[0]
                            elif veh['speed'] > speed_range[1]: veh['speed'] = speed_range[1]

                        # replace the old vehicle
                        self.vehs[i] = veh

                        return 0
                    else:
                        # duplicated vehicle, but old vehicle has better estimates, then ignore this new estimates
                        # combine the data points
                        old_v['inliers'] += veh['inliers']

                        # if new veh has ultrasonic reading, update
                        if veh['valid'] is True:
                            self.dists.append(veh['distance'])
                            old_v['speed'] = old_v['speed'] * veh['distance'] / old_v['distance']
                            old_v['distance'] = veh['distance']

                        print(
                        '######################## Discarded duplicated vehicle entering at {0}\n'.format(old_v['t_in']))
                        return 0

            # if not returned yet, meaning no duplicated vehicle, than register.
            if veh['valid'] is False and len(self.dists) != 0:
                m_d = np.median(self.dists)
                veh['speed'] = veh['speed'] * m_d / veh['distance']
                veh['distance'] = m_d
            else:
                self.dists.append(veh['distance'])

            # cap the speed to the limit
            if speed_range is not None:
                if veh['speed'] < speed_range[0]: veh['speed'] = speed_range[0]
                elif veh['speed'] > speed_range[1]: veh['speed'] = speed_range[1]

            self.vehs.append(veh)

    def _same_veh(self, v1, v2):
        """
        This function evaluates if v1 and v2 are essentially fitting to the trace of the same vehicle
        :param v1: vehicle dict
        :param v2: vehicle dict
        :return: True if same
        """

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

    def _save_det_vehs_txt(self, save_dir, save_name):
        """
        This function extracts and only saves the most important detection results from self.vehs
            [t_in, t_out, distance, speed, valid]
        :param save_dir: the directory for saving
        :param save_name: the file name
        :return:
        """
        with open(save_dir+save_name, 'w+') as f:
            f.write('t_in, t_out, dist (m), speed (mph), estimated_dist\n')
            for veh in self.vehs:
                f.write('{0},{1},{2},{3},{4}\n'.format(veh['t_in'], veh['t_out'], veh['distance'],
                                                       veh['speed'], veh['valid']))

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


class EvaluateResult:
    """
    This class evaluates and visualizes the estimation results.
    """
    def __init__(self):
        self.paras = {}

        self.mps2mph = 2.23694  # 1 m/s = 2.23694 mph

    def load_paras(self, paras_file):
        """
        This function loads the parameter files used for generating the results.
        :param paras_file: string, file dir + file name
        :return:
        """
        with open(paras_file, 'r') as fi:
            for line in fi:
                para, val_str = line.strip().split(':')
                val = make_tuple(val_str.strip())
                self.paras[para] = val

    def load_true_results(self, true_file, init_t, offset, drift_ratio):
        """
        This function loads the true result file (npy)
             start time, end time, speed (m/s), distance (m), image speed (px/frame), image distance (px)
        :param true_file: str
        :return:
        """
        true_vehs = np.load(true_file).tolist()

        # convert the first column into datetime and the speed to mph
        for true_v in true_vehs:
            # the video data is a bit drifting. 14 seconds over 2 hrs

            # # There is a time drift here
            # # For Jun 08
            # offset = 0.28
            # drift_cor = 1860.5/1864.0


            true_v[0] = init_t + timedelta(seconds=true_v[0]*drift_ratio + offset)
            true_v[1] = init_t + timedelta(seconds=true_v[1]*drift_ratio + offset)
            true_v[2] *= 2.24

        true_vehs = np.asarray(true_vehs)

        return true_vehs

    def visualize_results(self, norm_df, vehs, true_vehs=None, t_start=None, t_end=None):

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # axis 0 is the scattered PIR plot
        # axis 1 is the ultrasonic plot
        ax_pir = plt.subplot(gs[0])
        ax_ultra = plt.subplot(gs[1], sharex=ax_pir)
        plt.setp(ax_pir.get_xticklabels(), visible=False)

        # ==============================================================================
        # only plot the data in time interval
        if t_start is None: t_start = norm_df.index[0]
        if t_end is None: t_end = norm_df.index[-1]

        print('\n########################## Visualization: ')
        print('                 Data interval from: {0}'.format(norm_df.index[0]))
        print('                                 to: {0}'.format(norm_df.index[-1]))
        print('                 Plotting interval from: {0}'.format(t_start))
        print('                                     to: {0}'.format(t_end))

        # save the batch normalized data
        frames = norm_df.index[ (norm_df.index >= t_start) & (norm_df.index <= t_end) ]
        plot_df = norm_df.loc[frames,:]

        # align t_start and t_end
        t_start = plot_df.index[0]
        t_end = plot_df.index[-1]

        # ==============================================================================
        # Plot the detection result
        # --------------------------------------------------------
        # plot all the data point, perform nonlinear transform
        # ax_pir general
        pts, t_grid, x_grid = self._tx_representation(plot_df)
        ax_pir.scatter(pts[:,0], pts[:,1], color='0.6')

        # ax_ultra general
        ultra = plot_df['ultra']
        clean_ultra = self._clean_ultra(ultra)
        tmp_t = ultra.index - t_start
        rel_t = [i.total_seconds() for i in tmp_t]
        ax_ultra.plot(rel_t, ultra.values, linewidth=2, marker='*')
        ax_ultra.plot(rel_t, clean_ultra.values, linewidth=2, color='r')

        # plot the detected vehicles
        colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
        for veh in vehs:

            # vehicle enter and exit time
            t_left_s = (veh['t_left']-t_start).total_seconds()
            t_right_s = (veh['t_right']-t_start).total_seconds()

            # only plot those within the interval
            if np.max([t_left_s, t_right_s]) <=0 or np.min([t_left_s, t_right_s]) >= (t_end-t_start).total_seconds():
                continue

            # plot the supporting points and the line
            c = next(colors)
            sup_pts = np.array( [[(p[0]-t_start).total_seconds(), p[1]] for p in veh['inliers']] )
            ax_pir.scatter(sup_pts[:,0], sup_pts[:,1], color=c, alpha=0.75)
            ax_pir.plot([t_left_s, t_right_s],[x_grid[0], x_grid[-1]], linewidth=2, color='k')

            # plot the used distance value
            ax_ultra.plot([t_left_s, t_right_s], [veh['distance'], veh['distance']], linewidth=2, color=c)

        ax_pir.set_title('{0} ~ {1}'.format(time2str(t_start), time2str(t_end)), fontsize=20)
        # ax_pir.set_xlabel('Relative time (s)', fontsize=18)
        ax_pir.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax_pir.set_xlim([t_grid[0], t_grid[-1]])
        ax_pir.set_ylim([x_grid[-1], x_grid[0]])

        ax_ultra.set_title('Ultrasonic data', fontsize=20)
        ax_ultra.set_ylabel('Distance (m)', fontsize=18)
        ax_ultra.set_xlabel('Relative Time (s)', fontsize=18)
        ax_ultra.set_ylim([0,12])
        ax_ultra.set_xlim([rel_t[0], rel_t[-1]])


        # ==============================================================================
        # Plot the true vehicle detection
        if true_vehs is not None:
            for true_v in true_vehs:

                mean_t_s = ((true_v[0]-t_start).total_seconds() + (true_v[1]-t_start).total_seconds())/2.0

                # only plot those within the interval
                if (true_v[1]-t_start).total_seconds() <= 0 or \
                                (true_v[0]-t_start).total_seconds() >= (t_end-t_start).total_seconds(): continue

                # compute the slope
                slope = true_v[2]/(self.mps2mph*self.paras['tx_ratio']*true_v[3])
                true_t_in_s, true_t_out_s = mean_t_s + x_grid[0]/slope, mean_t_s + x_grid[-1]/slope

                # plot the true vehicle line in pir
                ax_pir.plot([true_t_in_s, true_t_out_s], [x_grid[-1], x_grid[0]], linewidth=2, linestyle='--', color='k')
                # plt the true vehicle distance in ultra
                ax_ultra.plot([true_t_in_s, true_t_out_s], [true_v[3], true_v[3]], linewidth=2, linestyle='--', color='k')


        plt.draw()

    def visualize_results_all(self, norm_df, vehs, true_vehs=None, t_start=None, t_end=None):
        """
        A wrapper function. split into sub intervals; otherwise too large to plot
        :param norm_df:
        :param ratio_tx:
        :param true_vehs:
        :param t_start:
        :param t_end:
        :return:
        """
        # only plot the data in time interval
        if t_start is None: t_start = norm_df.index[0]
        if t_end is None: t_end = norm_df.index[-1]

        _t_start = t_start
        _t_end = t_start + timedelta(seconds=15*60.0)
        while _t_end <= t_end:
            self.visualize_results(norm_df, vehs, true_vehs=true_vehs, t_start=_t_start, t_end=_t_end)
            _t_start = _t_end
            _t_end = _t_start + timedelta(seconds=15*60.0)

        # plot the rest
        self.visualize_results(norm_df, vehs, true_vehs=true_vehs, t_start=_t_start, t_end=t_end)

    def _tx_representation(self, norm_df):
        """
        This function performs the nonlinear transform of norm_df.
        :param norm_df: the dataframe
        :return: pts (2d np array, s, m), t_grid, x_grid
        """

        # ------------------------------------------------------------
        # initialize the space grid with nonlinear transformation
        # Slope * self.ratio * distance(m) = m/s
        x_grid = self._new_nonlinear_transform()

        # ------------------------------------------------------------
        # initialize the time grid in seconds
        ref_t = norm_df.index[0]
        t_grid = [(t-ref_t).total_seconds() for t in norm_df.index]
        t_grid = np.asarray(t_grid)

        # ------------------------------------------------------------
        # convert the matrix to a list of data point tuples
        pt_time = []
        pt_space = np.zeros(0)
        i = 0
        pir_len = self.paras['pir_res'][0]*self.paras['pir_res'][1]
        for cur_t, row in norm_df.iterrows():
            not_nan_idx = np.where(~np.isnan(row.values[0:pir_len]))[0]

            # append the not nan points using the grid
            pt_time += [t_grid[i]]*int(len(not_nan_idx))
            pt_space = np.concatenate([pt_space, x_grid[not_nan_idx]])

            # for col in range(0, self.pir_res[0]*self.pir_res[1]):
            #     if ~np.isnan(row.values[col]):
            #         pt_time.append(t_grid[i])
            #         pt_space.append(x_grid[col])
            i += 1

        pts = np.array(zip(pt_time, pt_space))

        return pts, t_grid, x_grid

    def _new_nonlinear_transform(self):
        """
        This function performs the nonlinear transform to the norm_df data
        :return: space grid
        """
        _dup = self.paras['pir_res'][0]
        d_theta = (self.paras['pir_fov'] / 16) * np.pi / 180.0

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

    def _clean_ultra(self, raw_ultra):
        """
        The ultrasonic sensor contains some erroneous readings. This function filters out those readings
        :param raw_ultra: the raw ultrasonic sensor data in DataFrame format
        :return: self.clean_ultra
        """

        clean_ultra = deepcopy(raw_ultra)

        len_d = len(raw_ultra)
        det = False

        i = 0
        start_idx, end_idx = 0, 0
        while i < len_d:
            v = clean_ultra.values[i]

            if det is False:
                if v <= self.paras['TH_ultra_fp']:
                    # set start index
                    start_idx = np.max([0, i - self.paras['ultra_fp_pre']])
                    end_idx = i
                    det = True
                i += 1
                continue
            else:
                # exiting a detection of false positive
                if v <= self.paras['TH_ultra_fp']:
                    # continue increasing idx
                    end_idx = i
                    i += 1
                    continue
                else:
                    # exit the detection of false positive
                    end_idx = np.min([len_d, end_idx + self.paras['ultra_fp_post']])

                    # replace the values
                    clean_ultra.values[start_idx:end_idx] = 11.0
                    det = False
                    # move on from the end_idx
                    i = end_idx

        return clean_ultra


class SpeedEst:
    """
    This class is used to estimate the speed of vehicles within a time window.
    """

    def __init__(self, norm_df, paras, window_s=5.0, speed_range=(1, 50),
                 plot_final=True, plot_debug=False, save_dir='./'):

        self.paras = paras
        self._debug = False
        # ------------------------------------------------------------
        # other properties
        self.plot_final = plot_final
        self.plot_debug = plot_debug
        self.save_dir = save_dir
        self.window_s = window_s
        self.speed_range = speed_range
        self.mps2mph = 2.23694  # 1 m/s = 2.23694 mph

        # ------------------------------------------------------------
        # determine the direction which can speed up the code by not consider the other direction models
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
        # initialize the space grid with nonlinear transformation
        # Slope * self.ratio * distance(m) = m/s
        # self.ratio_tx = 6.0  # to make the space and time look equal
        # _dup = self.paras['pir_res'][0]
        # d_theta = (60.0 / 16) * np.pi / 180.0
        # x_grid = []
        # for i in range(-16, 16):
        #     for d in range(0, _dup):
        #         # duplicate the nonlinear operator for vec
        #         x_grid.append(np.tan(d_theta / 2 + i * d_theta) / self.ratio_tx)
        # self.x_grid = -np.asarray(x_grid)

        # self.x_grid = self._old_nonlinear_transform()
        # print('old x_grid [0,15,16,31]: {0}, {1}, {2}, {3}'.format(self.x_grid[0], self.x_grid[15*4],
        #                                                            self.x_grid[16*4], self.x_grid[31*4]))

        self.x_grid = self._new_nonlinear_transform()
        # print('new x_grid [0,15,16,31]: {0}, {1}, {2}, {3}'.format(new_x_grid[0], new_x_grid[15*4],
        #                                                            new_x_grid[16*4], new_x_grid[31*4]))
        # print('new/old x_grid ratio: {0}'.format(new_x_grid[0]/self.x_grid[0]))

        # ------------------------------------------------------------
        # initialize the time grid in seconds
        self.init_dt = norm_df.index[0]
        self.end_dt = norm_df.index[-1]
        self.t_grid = [(t - norm_df.index[0]).total_seconds() for t in norm_df.index]

        # ------------------------------------------------------------
        # convert the PIR matrix to a list of data point tuples
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
        self.time = l_pts[:, 0]
        self.space = l_pts[:, 1]

        # ------------------------------------------------------------
        # extract the ultrasonic sensor data
        self.ultra = norm_df['ultra']
        self.clean_ultra = self._clean_ultra(norm_df['ultra'])

        # ------------------------------------------------------------
        # save detection results
        self.labeled_pts = np.array([]).astype(int)
        self.vehs = []

    def estimate_speed(self):
        """
        This is the top layer function that should be called in this class. It runs the iterative robust regression to
        fit linear models to the data points in this time window.
        :return: [veh1, veh2, ...], each veh: {}
            'line': (k,c), s = kt + c
            'tol': float, [-tol, tol] is used to define the inliers
            'inlier_idx': ndarray of int, the inlier indices
            'r2': the r2 of the fitting
            'dens': the density of the trace
            'residuals': the residual of the fitting
            'sigma': the std of the residual
            'num_inliers': the total number of inliers including those overlapping points

            'distance': m, the distance of the vehicle to the sensor
            'speed': mph, the speed of the vehicle
            'valid': True or False; valid if using the actual ultrasonic sensor reading

            't_in': datetime, the enter time
            't_out': datetime, the exit time
            't_left': datetime, time that the vehicle hit the left FOV boundary of PIR (top positive space value)
            't_right': datetime, time that the vehicle hit the right FOV boundary of PIR (bottome negative space value)

            'detection_window': (datetime, datetime), the initial and end time of this detection window that detects this veh
            'captured_percent': [0,1], the percentage of the trace being captured
            'captured_part': 'head', 'body', 'tail', 'full', which part of the trace is captured in this window

            'inliers': [[datetime, space], []], n x 2 ndarray, where n=len('inlier_idx')
        """

        # Use DBSCAN to initialize the algorithm. Optional, but effective.
        clusters = self._init_clusters()

        if len(clusters) == 0:
            # Meaning all pts in frame is < min_num_pts, then skip
            print('########################## Did not find vehicles in this frame starting at: {0}\n'.format(
                time2str_file(self.init_dt)))
            return []
        else:
            # For each cluster, fit multiple models
            for counter, inlier_idx in enumerate(clusters):
                print('\n===========================================================================')
                print('-- Cluster {0}: {1} pts'.format(counter, len(inlier_idx)))

                self._fit_mixed_mdls(inlier_idx)

            # check the remaining points
            if len(self.time) - len(self.labeled_pts) >= self.paras['min_inliers']:
                # remaining points may still support one car
                rem_idx = np.arange(0, len(self.time), 1).astype(int)
                # removed labeled points
                rem_idx = np.delete(rem_idx, self.labeled_pts)
                print('\n===========================================================================')
                print('-- Checking remaining points: {0}'.format(len(rem_idx)))

                self._fit_mixed_mdls(rem_idx)

            # ================================================================================================
            # plot final estimates
            print('########################## Finished estimation' +
                  ' ({0} models) for frame starting at: {1}\n'.format(len(self.vehs), time2str_file(self.init_dt)))

            if self.plot_final:
                self._plot_progress(cur_mdl=None, save_name='{0}'.format(time2str_file(self.init_dt)),
                                    title='{0}'.format(self.init_dt))

            # ================================================================================================
            # Clean vehicles by removing wrong direction and merge close traces
            cleaned = self._clean_models()
            if cleaned and self.plot_final:
                self._plot_progress(cur_mdl=None, save_name='{0}_cleaned'.format(time2str_file(self.init_dt)),
                                    title='{0}'.format(self.init_dt))

            # ================================================================================================
            # compute the vehicle speed and format output
            self._compute_speed()

            return self.vehs

    def _old_nonlinear_transform(self):
        """
        This function performs the nonlinear transform to the norm_df data:
        Old: assuming FOV is 120 and angle evenly spli
        :return: space grid
        """
        _dup = self.paras['pir_res'][0]
        d_theta = (60.0 / 16) * np.pi / 180.0
        x_grid = []
        for i in range(-16, 16):
            for d in range(0, _dup):
                # duplicate the nonlinear operator for vec
                x_grid.append(np.tan(d_theta / 2 + i * d_theta) / self.paras['tx_ratio'])

        return -np.asarray(x_grid)

    def _new_nonlinear_transform(self):
        """
        This function performs the nonlinear transform to the norm_df data
        :return: space grid
        """
        _dup = self.paras['pir_res'][0]
        d_theta = (self.paras['pir_fov'] / 16) * np.pi / 180.0

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


    def _fit_mixed_mdls(self, initial_idx):
        """
        This function fits one or multiple linear models starting from initial_idx
        :param initial_idx: the initial sets of point to start the algorithm
        :return: fitted models are saved in self.vehs
        """

        # First try to fit a single model
        mdl = self._fit_mdl(initial_idx)

        if mdl is not None and mdl['num_inliers'] >= self.paras['KD_min_split_pts']:
            # if the model inliers is greater than the minimum number of points that may be further split

            if self._accept_mdl(mdl):
                print('$$$$ Good fitting of {0} pts with r2: {1:.3f} and density: {2:.3f}'.format(
                    mdl['num_inliers'], mdl['r2'], mdl['dens']))
                self.vehs.append(mdl)
                self.labeled_pts = np.concatenate([self.labeled_pts, mdl['inlier_idx']])

            else:
                # A bad fitting which may due to multiple traces. Split
                print('\n$$$$ Bad fitting of {0} pts with r2: {1:.3f} and density: {2:.3f}'.format(
                    mdl['num_inliers'], mdl['r2'], mdl['dens']))
                print('$$$$ Splitting to subclusters...')

                # Specify some candidate slopes to speed up convergence
                candidate_slopes = [mdl['line'][0]]
                for veh in self.vehs:
                    candidate_slopes.append(veh['line'][0])

                # Split and get the subclusters
                # (line, sigma, weight, aic)
                sub_clusters = self._split_cluster_exhaustive(mdl['inlier_idx'], candidate_slopes=candidate_slopes)

                for i, (line, sigma, w, aic) in enumerate(sub_clusters):

                    # NOTE: here set the tol = sigma to start from a narrow core of the trace.
                    inlier_idx = self._update_inliers(line, tol=sigma)

                    if len(inlier_idx) >= self.paras['KD_min_init_pts']:
                        # only check those with sufficient initial number of points
                        print('      -- fitting subcluster with {0} initial pts'.format(len(inlier_idx)))
                        sub_mdl = self._fit_mdl(inlier_idx)
                        if sub_mdl is None:
                            print('                  No model converged')
                            continue

                        num_pts = sub_mdl['num_inliers']
                        if self._accept_mdl(sub_mdl):
                            self.vehs.append(sub_mdl)
                            self.labeled_pts = np.concatenate([self.labeled_pts, sub_mdl['inlier_idx']])
                            print('$$$$ Good fitting of {0} pts with r2: {1}\n'.format(num_pts, sub_mdl['r2']))
                        else:
                            print(
                                '$$$$ Discarding bad fitting of {0} pts with r2: {1}\n'.format(num_pts, sub_mdl['r2']))

    def _fit_mdl(self, inlier_idx):
        """
        This function fit a linear line to the inlier points using iterative regression
        :param inlier_idx: a list of integers (inliers)
        :return: a model dict
            'line': (k,c)
            'tol': float, [-tol, tol] is used to define the inliers
            'inlier_idx': ndarray of int the inlier indices
            'r2': the r2 of the fitting
            'dens': the density of the trace
            'residuals': the residual of the fitting
            'sigma': the std of the residual
            'num_inliers': the total number of inliers including those overlapping points
        """

        if len(inlier_idx) < self.paras['min_inliers']:
            return None

        try:
            # try to fit a line
            pre_line = None
            converged = False
            pre_num_pts = len(inlier_idx)
            added_pts = len(inlier_idx)
            print('---------- Fitting a line, iterations: ')
            for i in xrange(self.paras['max_iter']):
                print('               # {0}: {1} pts'.format(i, len(inlier_idx)))

                # ---------------------------------------------------------------------------------------
                # Fit a line using linear regression with s as the independent variable, i.e., t = ax+b.
                _slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(self.space[inlier_idx],
                                                                                          self.time[inlier_idx])

                # convert to the line format: s = kt + c
                line = np.array([1 / _slope, -_intercept / _slope])
                # compute the goodness of fit score r2
                r2 = _r_value ** 2

                if pre_line is not None and \
                        (np.asarray(line) - np.asarray(pre_line) <= np.asarray(self.paras['TH_stop'])).all() and \
                                added_pts < self.paras['min_added_pts']:
                    # # converged by conditions:
                    #   - no significant change of lines
                    #   - no significant change of points
                    print('---------- Converged: x = {0:.03f}t + {1:.03f}; {2:.01f} mph assuming {3:.01f} m'.format(
                        line[0], line[1], -self.paras['d_default'] * self.mps2mph * self.paras['tx_ratio'] * line[0],
                        self.paras['d_default']))

                    converged = True

                # ---------------------------------------------------------------------------------------
                # Compute the tolerance for the new iteration
                tol, sig, dists = self._compute_tol(line, inlier_idx, r2)

                # Plot the progress
                if self.plot_debug is True:
                    dens = self._compute_density(line, tol)

                    # num_inliers = len(self._get_all_inliers(line, tol))
                    num_inliers = len(inlier_idx)

                    cur_mdl = {'line': line, 'tol': tol, 'inlier_idx': inlier_idx, 'r2': r2, 'dens': dens,
                               'residuals': dists, 'sigma': sig, 'num_inliers': num_inliers}
                    self._plot_progress(cur_mdl, save_name=None, title=None)

                if not converged:
                    # update inliers for the next iteration
                    inlier_idx = self._update_inliers(line, tol)

                    # update previous mdl as the current model
                    added_pts = np.abs(len(inlier_idx) - pre_num_pts)
                    pre_num_pts = len(inlier_idx)
                    pre_line = deepcopy(line)

                else:
                    # Converged, then compute the density
                    dens = self._compute_density(line, tol)

                    # all number of inliers
                    # num_inliers = len(self._get_all_inliers(line, tol))
                    num_inliers = len(inlier_idx)

                    mdl = {'line': line, 'tol': tol, 'inlier_idx': inlier_idx, 'r2': r2, 'dens': dens,
                           'residuals': dists, 'sigma': sig, 'num_inliers': num_inliers}

                    return mdl

        except ValueError:
            return None

    def _update_inliers(self, line, tol):
        """
        This function updates the inliers, i.e., get data points that lies within tol perpendicular distance to model
         :param line: k,c. s = kt + c
        :param tol, [-tol, tol]
        :return: new_inliers,
        """

        k, c = line
        # the residual of t_data - f(s), where t = f(s) = (s-c)/k
        dist = np.abs(self.time - (self.space - c) / k)
        idx = (dist <= tol)

        new_inliers = np.array([i for i, x in enumerate(idx) if x and i not in self.labeled_pts]).astype(int)

        return new_inliers

    def _compute_tol(self, line, pre_inlier_idx, r2):
        """
        This function updates the tolerance of the model line, last_pt_idx are points used to fit the line
            use last_pt_idx to compute the sigma and then compute the tolerance
        :param line: k,c. s = kt + c
        :param pre_inlier_idx: list of integers
        :param r2: the r2 score of the fitting
        :return: tol, sigma, dists
        """

        # --------------------------------------------------------------
        # find the idx of points within in the old tolerance
        # compute the old sigma
        sig, dists = self._compute_sigma(line, pt_idx=pre_inlier_idx)

        # determine whether to expand or contract
        if r2 <= self.paras['TH_r2']:
            # bad fit, then expand
            tol = self.paras['expansion_ratio'] * sig
        else:
            # contract to a good fit
            tol = self.paras['contraction_ratio'] * sig

        return tol, sig, dists

    def _compute_sigma(self, line, pt_idx=None):
        """
        This function computes the sigma tolerance of the residual of pts to the line
        :param line: (k, b)
        :param pt_idx: index of points supporting this line
        :return: sigma, dists
        """

        if pt_idx is None:
            # if not specified, project all points
            pt_idx = np.arange(0, len(self.time)).astype(int)

        # Compute the residual of the points
        k, c = line
        dists = self.time[pt_idx] - (self.space[pt_idx] - c) / k

        sigma = np.std(dists)

        return sigma, dists

    def _compute_density(self, line, tol):
        """
        This function computes the points density in that model
        :param line: k,c, s=kt+c
        :param tol: the tolerance for computing the density
        :return: density [0,1]
        """
        inliers_idx = self._get_all_inliers(line, tol)

        # number of time levels
        n_t_levels = np.round(2 * tol * self.paras['sampling_freq'])

        # compute the percentage of the trace being captured
        perc = self._compute_captured_trace_percent(line)

        return len(inliers_idx) / (self.paras['pir_res'][1] * n_t_levels * perc)

    def _compute_captured_trace_percent(self, line):
        """
        This function computes the percentage of the trace that was captured in the time window
        :param line: k,c . s = kt + c, t = (s-c)/k
        :return: [0,1]
        """
        k, c = line

        # compute the enter and exit time.
        in_s = (self.x_grid[0] - c) / k
        out_s = (self.x_grid[-1] - c) / k

        # make sure the time window is in right order
        if in_s >= out_s: in_s, out_s = out_s, in_s

        # determine the frame location
        # compute the percent of the trace in the detection window, which will be used as an indicator on how much the
        # estimated speed should be trusted.
        if in_s >= 0 and out_s <= self.window_s:
            det_perc = 1.0
        elif in_s >= 0 and out_s > self.window_s:
            det_perc = (self.window_s - in_s) / (out_s - in_s)
        elif in_s < 0 and out_s <= self.window_s:
            det_perc = out_s / (out_s - in_s)
        else:
            det_perc = self.window_s / (out_s - in_s)

        return det_perc

    def _get_all_inliers(self, line, tol):
        """
        This function gets all the inliers for the line, including those pts labeled by other lines
        :param line: k, c, s = kt + c
        :param tol: the tolerance of the model [-tol, tol]
        :return: index list
        """
        k, c = line

        # the residual of t_data - f(s), where t = f(s) = (s-c)/k
        dist = np.abs(self.time - (self.space - c) / k)

        idx = (dist <= tol)

        return np.array([i for i, x in enumerate(idx) if x]).astype(int)

    def _accept_mdl(self, mdl):
        """
        This function determines if mdl is an acceptable mdl fitting a vehicle trace
        :param mdl: None or mdl dict
        :return: True or False
        """
        if mdl is None:
            return False
        else:
            r2, dens, num_pts = mdl['r2'], mdl['dens'], mdl['num_inliers']
            return r2 >= self.paras['TH_r2'] and dens >= self.paras['TH_dens'] and num_pts >= self.paras['min_inliers']

    def _init_clusters(self):
        """
        This function returns a list of candidate clusters using DBSCAN.
        :return: [cluster_1, cluster_2], each cluster is a list of int (indices)
        """
        clusters = []

        samples = np.vstack([self.time, self.space]).T

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

        return clusters

    def _split_cluster_exhaustive(self, pts_idx, candidate_slopes=None):
        """
        This function tries to splits pts_idx to clusters by projection to different directions
        :param pts_idx: the inliers idx to be splitted
        :param candidate_slopes: a list of candidate slopes that may help
        :return:
        """

        # -------------------------------------------------------------------------------------
        # explore all directions of lines with reference to the left bottom corner
        # explore all directions of lines with reference to the left bottom corner
        speeds = np.arange(self.speed_range[0], self.speed_range[1], self.paras['speed_res']).astype(float)
        slopes = -speeds / (self.paras['d_default'] * self.mps2mph * self.paras['tx_ratio'])

        # also explore the candidate directions
        if candidate_slopes is not None:
            slopes = np.concatenate([slopes, np.asarray(candidate_slopes)])
            speeds = np.concatenate(
                [speeds, -np.asarray(candidate_slopes) * (self.paras['d_default'] * self.mps2mph * self.paras['tx_ratio'])])

        all_dirs = []
        print('------ Exploring directions:')
        for i, k in enumerate(slopes):
            # The split in each direction will return the number of subclusters
            # each row of group [(k,c), sigma, weight, aic]
            group = self._split_cluster(k, pts_idx=pts_idx)
            all_dirs.append(group)

            print('             At {0} mph: {1} subclusters'.format(speeds[i], len(group)))

        # -------------------------------------------------------------------------------------
        # Return the Union of (top n of _weights) and (top n of _aic) directions to determine the best split
        _weights = []
        _avg_aic = []
        for i, g in enumerate(all_dirs):
            if len(g) == 0:
                _weights.append(0)
                _avg_aic.append(0)
            else:
                _weights.append(np.sum(g[:, 2]))
                _avg_aic.append(np.mean(g[:, 3]))

        top_w = np.array([i[0] for i in sorted(enumerate(-np.array(_weights)),
                                               key=lambda x: x[1])])[0:self.paras['KD_top_n']]
        top_aic = np.array([i[0] for i in sorted(enumerate(_avg_aic), key=lambda x: x[1])])[0:self.paras['KD_top_n']]

        # get the possible lines
        possible_lines = np.zeros((0, 4))
        for i, dire in enumerate(all_dirs):
            if len(dire) != 0 and i in top_w and i in top_aic:
                possible_lines = np.vstack([possible_lines, dire])

        print('------ Found {0} sub clusters\n'.format(len(possible_lines)))

        # sort the subclusters in all directions by the _weight
        possible_lines = sorted(possible_lines, key=lambda x: x[2])[::-1]

        return possible_lines

    def _split_cluster(self, slope, pts_idx=None):
        """
        This function splits the pts_idx into subclusters along slope
        :param slope: the slope for computing the residuals
        :param pts_idx: the index of points to be split
        :return: list of list, [subclus1, subclus2, ...]
            subclus: [(k,c), sigma, weight, aic]
        """

        # if not specified, project all points
        if pts_idx is None: pts_idx = np.arange(0, len(self.time)).astype(int)

        # comptue the residuals
        residuals = self.time[pts_idx] - (self.space[pts_idx]) / slope

        # use gaussian kernel for density estimation
        kde = KernelDensity(bandwidth=0.01, kernel='gaussian').fit(residuals[:, np.newaxis])

        x_ticks = np.linspace(np.min(residuals), np.max(residuals), 100)
        log_dens = kde.score_samples(x_ticks[:, np.newaxis])

        # find the local minimums
        x_minimas_idx = argrelextrema(log_dens, np.less)[0]
        x_segs = zip(np.concatenate([[0], x_minimas_idx]), np.concatenate([x_minimas_idx, [len(x_ticks) - 1]]))

        # For each segment in x_segs, fit a Gaussian to get the intercept, which is the candidate line
        means = []
        stds = []
        weights = []
        aic = []

        for seg_s, seg_e in x_segs:
            seg_data_idx = (residuals >= x_ticks[seg_s]) & (residuals < x_ticks[seg_e])

            if sum(seg_data_idx) >= self.paras['KD_min_init_pts']:
                # the cluster be sufficiently large to be considered as potentially belonging to a trace
                seg_data = residuals[seg_data_idx]
                gmm = GaussianMixture()
                r = gmm.fit(seg_data[:, np.newaxis])
                means.append(r.means_[0, 0])
                stds.append(np.sqrt(r.covariances_[0, 0]))
                weights.append(float(len(seg_data)) / len(residuals))
                aic.append(gmm.aic(seg_data[:, np.newaxis]))

        # Compute the lines for each subcluster
        if len(means) != 0:
            means = np.asarray(means)
            intercepts = means * np.sqrt(slope ** 2.0 + 1.0)
            lines = [(slope, i) for i in intercepts]

            if self.plot_debug:
                self._plot_sub_clusters(slope, pts_idx, residuals=residuals, gaussians=zip(means, stds, weights),
                                        x_ticks=x_ticks, log_dens=log_dens, minimas=x_ticks[x_minimas_idx],
                                        title='Splitting clusters', save_name='subclusters')

            return np.array(zip(lines, stds, weights, aic))

        else:
            return np.array([])

    def _clean_models(self):
        """
        This function cleans the converged models in the time window
            - remove wrong directions
            - merge multiple traces that are too close by fitting a line to the merged inliers (without check accept_mdl)
        :return: True False, if self.vehs has been cleaned
        """

        cleaned = False
        if len(self.vehs) == 0:
            # nothing to be cleaned
            return cleaned

        elif len(self.vehs) == 1:
            # if both positive or negative, then wrong direction, and remove the veh
            if (self.direction == 'positive' and self.vehs[0]['line'][0] > 0) or \
                    (self.direction == 'negative' and self.vehs[0]['line'][0] < 0):
                self.vehs = []
                cleaned = True

        else:
            # multiple models
            # get the t_in and t_out of all models and then use DBSCAN to cluster models that belong to the same vehicle
            # idx of model in vehs, t_in, t_out
            t_in_out = []
            for i, mdl in enumerate(self.vehs):

                # first check the direction
                if (self.direction == 'positive' and self.vehs[0]['line'][0] > 0) or \
                        (self.direction == 'negative' and self.vehs[0]['line'][0] < 0):
                    cleaned = True
                    continue

                # for the correct direction, compute the enter and exit time
                t_in = (self.x_grid[0] - mdl['line'][1]) / mdl['line'][0]
                t_out = (self.x_grid[-1] - mdl['line'][1]) / mdl['line'][0]

                # make sure the time window is in right order
                if t_in > t_out: t_in, t_out = t_out, t_in

                t_in_out.append([i, t_in, t_out])

            # ========================================================================
            # if no vehicle in the correct direction, then return True
            if len(t_in_out) == 0:
                self.vehs = []
                return cleaned

            # ========================================================================
            # Use DBSCAN to find the models that to be merged
            ts_te = [i[1:3] for i in t_in_out]

            y_pre = DBSCAN(eps=self.paras['min_headway'], min_samples=1).fit_predict(ts_te)
            num_clusters = len(set(y_pre)) - (1 if -1 in y_pre else 0)
            y_pre = np.asarray(y_pre)

            # ========================================================================
            # Save the final models in to cleaned vehs
            cleaned_vehs = []

            for clus in range(0, num_clusters):
                n_mdls = sum(y_pre == clus)

                if n_mdls == 1:
                    # only one model in this cluster, hence no need to merge
                    idx = [i for i, x in enumerate(y_pre) if x == clus]
                    cleaned_vehs.append(self.vehs[t_in_out[idx[0]][0]])

                else:
                    # merge multiple models into one by fitting a new line to the merged inlier idx
                    idx = [i for i, x in enumerate(y_pre) if x == clus]
                    _merge_idx = np.array([]).astype(int)
                    for i in idx:
                        _merge_idx = np.concatenate([_merge_idx, self.vehs[t_in_out[i][0]]['inlier_idx']])
                    _merged_mdl = self._fit_mdl(_merge_idx)
                    cleaned_vehs.append(_merged_mdl)
                    cleaned = True

            # ========================================================================
            # replace self.vehs
            if cleaned is True: self.vehs = cleaned_vehs

            return cleaned

    def _compute_speed(self):
        """
        This function computes the vehicle speeds, and save them into the structure
        :return:
        """

        for veh in self.vehs:
            # =====================================================================
            # Find the distance from ultrasonic sensor data
            # compute the in and out time to the FOV of ultrasonic sensor
            t_in = self.init_dt + timedelta(seconds=(self.paras['ultra_fov_left'] - veh['line'][1]) / veh['line'][0])
            t_out = self.init_dt + timedelta(seconds=(self.paras['ultra_fov_right'] - veh['line'][1]) / veh['line'][0])

            if t_in > t_out: t_in, t_out = t_out, t_in

            idx = (self.clean_ultra.index >= t_in) & (self.clean_ultra.index <= t_out)

            if len(self.clean_ultra[idx].values) == 0:
                veh['distance'] = self.paras['d_default']
                veh['valid'] = False
            else:
                # Check if there is false negative
                _d = np.min(self.clean_ultra[idx].values)
                if _d >= self.paras['TH_no_ultra']:
                    veh['distance'] = self.paras['d_default']
                    veh['valid'] = False
                else:
                    veh['distance'] = _d
                    veh['valid'] = True

            # =====================================================================
            # Compute the speed in mph
            # NOTE: speeds computed by d_default will be updated in Alg class using historical median
            veh['speed'] = np.abs(self.mps2mph * veh['line'][0] * self.paras['tx_ratio'] * veh['distance'])

            # =====================================================================
            # Compute the in and out time for the PIR FOV
            veh['t_left'] = self.init_dt + timedelta(seconds=(self.x_grid[0] - veh['line'][1]) / veh['line'][0])
            veh['t_right'] = self.init_dt + timedelta(seconds=(self.x_grid[-1] - veh['line'][1]) / veh['line'][0])

            if veh['t_left'] > veh['t_right']:
                veh['t_in'], veh['t_out'] = veh['t_right'], veh['t_left']
            else:
                veh['t_in'], veh['t_out'] = veh['t_left'], veh['t_right']

            # =====================================================================
            # save the inlier points in datetime and space for visualization
            _t = self.time[veh['inlier_idx']]
            pts_t = [self.init_dt + timedelta(seconds=i) for i in _t]
            veh['inliers'] = zip(pts_t, self.space[veh['inlier_idx']])

            # =====================================================================
            # save the detection window, captured part and percentage
            veh['detection_window'] = (self.init_dt, self.end_dt)

            in_s, out_s = (veh['t_in'] - self.init_dt).total_seconds(), (veh['t_out'] - self.init_dt).total_seconds()

            if in_s >= 0 and out_s <= self.window_s:
                veh['captured_part'] = 'full'
                veh['captured_percent'] = 1.0
            elif in_s >= 0 and out_s > self.window_s:
                veh['captured_part'] = 'head'
                veh['captured_percent'] = (self.t_grid[-1] - in_s) / (out_s - in_s)
            elif in_s < 0 and out_s <= self.window_s:
                veh['captured_part'] = 'tail'
                veh['captured_percent'] = (out_s - self.t_grid[0]) / (out_s - in_s)
            elif in_s < 0 and out_s > self.window_s:
                veh['captured_part'] = 'body'
                veh['captured_percent'] = (self.t_grid[-1] - self.t_grid[0]) / (out_s - in_s)

    def _clean_ultra(self, raw_ultra):
        """
        The ultrasonic sensor contains some erroneous readings. This function filters out those readings
        :param raw_ultra: the raw ultrasonic sensor data in DataFrame format
        :return: self.clean_ultra
        """

        clean_ultra = deepcopy(raw_ultra)

        len_d = len(raw_ultra)
        det = False

        i = 0
        start_idx, end_idx = 0, 0
        while i < len_d:
            v = clean_ultra.values[i]

            if det is False:
                if v <= self.paras['TH_ultra_fp']:
                    # set start index
                    start_idx = np.max([0, i - self.paras['ultra_fp_pre']])
                    end_idx = i
                    det = True
                i += 1
                continue
            else:
                # exiting a detection of false positive
                if v <= self.paras['TH_ultra_fp']:
                    # continue increasing idx
                    end_idx = i
                    i += 1
                    continue
                else:
                    # exit the detection of false positive
                    end_idx = np.min([len_d, end_idx + self.paras['ultra_fp_post']])

                    # replace the values
                    clean_ultra.values[start_idx:end_idx] = 11.0
                    det = False
                    # move on from the end_idx
                    i = end_idx

        return clean_ultra

    def _plot_progress(self, cur_mdl=None, save_name=None, title=None):
        """
        This function plots the current progress of the model
        :param cur_mdl: the current model being fitted; it will be highlighted
        :param save_name: the name to be saved
        :param title: the plot title
        :return:
        """
        # ===========================================================================
        # plot the initial figure
        plt.figure(figsize=(10, 15))
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])

        # Axis 0 will be used to plot the scatter plot of data and the fitted line
        # Axis 1 will be used to plot the ultrasonic sensor data
        # Axis 2 will be used to plot the analysis of the fitting
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])

        # ===========================================================================
        # plot ax0: the fitting
        x_line = np.asarray([0.0, self.window_s])
        ax0.scatter(self.time, self.space, color='0.6')

        # ===========================================================================
        # scatter the previously estimated converged models

        offset = 0
        y_lim = 0
        if len(self.vehs) != 0:
            colors = itertools.cycle(['b', 'g', 'm', 'c', 'purple'])
            for i, veh in enumerate(self.vehs):
                c = next(colors)
                line, tol, inlier_idx, r2, dens, sigma, residuals = \
                    veh['line'], veh['tol'], veh['inlier_idx'], veh['r2'], veh['dens'], veh['sigma'], veh['residuals']

                # --------------------------------------------------------
                # ax0: the scatter plot
                ax0.scatter(self.time[inlier_idx], self.space[inlier_idx], color=c, alpha=0.75)

                # plot the fitted line
                y_line = line[0] * x_line + line[1]
                ax0.plot(x_line, y_line, linewidth=3, color='k')

                # compute the enter and exit time.
                t_enter = (self.paras['ultra_fov_left'] - line[1]) / line[0]
                t_exit = (self.paras['ultra_fov_right'] - line[1]) / line[0]

                # make sure the times are correctly ordered
                if t_enter > t_exit: t_enter, t_exit = t_exit, t_enter

                # plot a vertical line indicating ultrasonic sensor FOV
                ax0.axvline(x=t_enter, linestyle='--')
                ax0.axvline(x=t_exit, linestyle='--')

                # --------------------------------------------------------
                # ax1: get the ultrasonic sensor data and find the corresponding distanct
                ax1.axvline(x=t_enter, linestyle='--')
                ax1.axvline(x=t_exit, linestyle='--')
                t_index = (self.clean_ultra.index >= self.clean_ultra.index[0] + timedelta(seconds=t_enter)) & \
                          (self.clean_ultra.index <= self.clean_ultra.index[0] + timedelta(seconds=t_exit))

                if len(self.clean_ultra[t_index].values) != 0:
                    d = np.min(self.clean_ultra[t_index].values)
                else:
                    d = self.paras['d_default']
                if d >= self.paras['TH_no_ultra']:
                    # False negative from ultrasonic sensor
                    d = self.paras['d_default']

                # --------------------------------------------------------
                # ax2: plot the distribution of residuals
                # shift the mean of the distribution
                residuals += 3 * sigma + offset
                bin_width = 0.01
                n, bins, _patches = ax2.hist(residuals, bins=np.arange(offset, 6 * sigma + offset, bin_width),
                                             normed=1, facecolor=c, alpha=0.75)
                # fill the one-sig space.
                x_fill = np.linspace(3 * sigma + offset - tol, 3 * sigma + offset + tol, 100)
                # fill the sig_ratio*sigma
                ax2.fill_between(x_fill, 0, mlab.normpdf(x_fill, 3 * sigma + offset, sigma), facecolor='r', alpha=0.65)
                # the gaussian line
                ax2.plot(bins, mlab.normpdf(bins, 3 * sigma + offset, sigma), linewidth=2, c='r')

                text = 'R2: {0:.3f}; #:{1}\n'.format(r2, len(inlier_idx)) + \
                       '{0:.1f} m; {1:.2f} mph\n'.format(d, -line[0] * d * self.mps2mph * self.paras['tx_ratio']) + \
                       'Density:{0:.2f}'.format(dens)

                ax2.annotate(text, xy=(offset + sigma, np.max(n) * 1.1), fontsize=10)
                ax2.set_title('All converged models', fontsize=16)
                y_lim = np.max([np.max(n), y_lim])
                # update offset to the right 3sigma of this distribution
                offset += 6 * sigma

        # ===========================================================================
        # plot the current model
        if cur_mdl is not None:

            line, tol, inlier_idx, r2, dens, sigma, residuals = \
                cur_mdl['line'], cur_mdl['tol'], cur_mdl['inlier_idx'], cur_mdl['r2'], cur_mdl['dens'], \
                cur_mdl['sigma'], cur_mdl['residuals']

            # --------------------------------------------------------
            # ax0
            ax0.scatter(self.time[inlier_idx], self.space[inlier_idx], color='r')
            y_line = line[0] * x_line + line[1]
            ax0.plot(x_line, y_line, linewidth=3, color='b')

            # plot the tolerance
            if line[0] != 0:
                # using time residual
                c1 = line[1] + (tol * line[0])
                c2 = line[1] - (tol * line[0])
            else:
                c1 = line[1] + tol
                c2 = line[1] - tol

            y_line_1 = line[0] * x_line + c1
            ax0.plot(x_line, y_line_1, linewidth=2, color='r', linestyle='--')
            y_line_2 = line[0] * x_line + c2
            ax0.plot(x_line, y_line_2, linewidth=2, color='r', linestyle='--')

            # --------------------------------------------------------
            # ax2: plot the histogram
            residuals += 3 * sigma + offset
            bin_width = 0.01
            n, bins, _patches = ax2.hist(residuals, bins=np.arange(offset, 6 * sigma + offset, bin_width),
                                         normed=1, facecolor='green', alpha=0.75)
            # fill the sig_ratio*sig space.
            x_fill = np.linspace(3 * sigma + offset - tol, 3 * sigma + offset + tol, 100)
            ax2.fill_between(x_fill, 0, mlab.normpdf(x_fill, 3 * sigma + offset, sigma), facecolor='r', alpha=0.65)
            ax2.plot(bins, mlab.normpdf(bins, 3 * sigma + offset, sigma), linewidth=2, c='r')

            text = 'R2: {0:.3f}; #:{1}\n'.format(r2, len(inlier_idx)) + \
                   '~{0:.1f} m; {1:.2f} mph\n'.format(self.paras['d_default'],
                                                      -line[0] * self.paras[
                                                          'd_default'] * self.mps2mph * self.paras['tx_ratio']) + \
                   'Density:{0:.2f}'.format(dens)

            ax2.annotate(text, xy=(offset + sigma, np.max(n) * 1.1), xycoords='axes fraction', fontsize=12)
            y_lim = np.max([np.max(n), y_lim])
            offset += 6 * sigma

        # ===========================================================================
        # set the axis properties
        ax0.set_title('{0}'.format(title), fontsize=20)
        ax0.set_xlabel('Time (s)', fontsize=18)
        ax0.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax0.set_xlim([0, self.window_s])
        ax0.set_ylim([self.x_grid[-1], self.x_grid[0]])

        s_tick= [i for i in ax0.get_yticks().tolist() if self.x_grid[-1]<=i<=self.x_grid[0] ]
        s_tick = [self.x_grid[-1]] + s_tick + [self.x_grid[0]]
        s_ticklabel = ['{0:.3f}'.format(i) for i in s_tick]
        ax0.set_yticks(s_tick, s_ticklabel)
        ax0.set_ylim([self.x_grid[-1]-0.005, self.x_grid[0]+0.005])

        if y_lim != 0: ax2.set_ylim([0, y_lim * 1.8])
        if offset != 0: ax2.set_xlim([0, offset])

        # ===========================================================================
        # plot ax1: the ultrasonic sensor data
        tmp_t = self.ultra.index - self.ultra.index[0]
        rel_t = [i.total_seconds() for i in tmp_t]
        ax1.plot(rel_t, self.ultra.values, linewidth=2, marker='*')
        ax1.plot(rel_t, self.clean_ultra.values, linewidth=2, color='r')
        ax1.set_title('Ultrasonic data', fontsize=16)
        ax1.set_ylabel('Distance (m)', fontsize=14)
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax1.set_ylim([0, 12])
        ax1.set_xlim([0, self.window_s])

        # ===========================================================================
        if save_name is not None:
            plt.savefig(self.save_dir + '{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.draw()

    def _plot_sub_clusters(self, slope, pt_idx, residuals, gaussians, x_ticks, log_dens, minimas, title,
                           save_name=None):
        """
        This function is only used for debugging. It visualizes how the subclusters are obtained using kernel density
        estimation
        :param slope: the slope of the projection
        :param pt_idx: the inlier points index
        :param residuals: the residuals to the line: (slope, 0)
        :param gaussians: zip(means, stds, weights), the gaussian fitting of the lines
        :param x_ticks: [],the x_ticks of the residuals
        :param log_dens: [], the log density from kernel density estimation
        :param minimas: [], the local minimums used for segmentation
        :param title: the title
        :param save_name: file name
        :return:
        """

        ref_line = slope, 0.0

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

        ax0.set_title('Splitting cluster {0}'.format(title), fontsize=20)
        ax0.set_xlabel('Time (s)', fontsize=18)
        ax0.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax0.set_xlim([np.min(self.time), np.max(self.time)])
        ax0.set_ylim([np.min(self.space), np.max(self.space)])

        # ----------------------------------------------------
        # plot the analysis of the histogram
        # plot histogram
        bin_width = 0.01
        n, bins, patches = ax1.hist(residuals, bins=np.arange(x_ticks[0], x_ticks[-1], bin_width),
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


# ==================================================================================================================
# sensor data class
# ==================================================================================================================
class SensorData:
    """
    This is the sensor data class, which mainly focuses on preprocessing, analysis and visualization of the data.
    """
    def __init__(self, pir_res=(4,32), save_dir='', plot=False):
        """
        Initialize the function with the data source and output options
        :return:
        """
        self.pir_res = pir_res
        self.tot_pix = pir_res[0]*pir_res[1]
        self.save_dir = save_dir
        self.plot = plot

        # ------------------------------------------------------------
        # Some parameters that may influence the performance, but should be robust
        # ------------------------------------------------------------
        # model noise
        self.Q = 0.05**2
        self.KF_update_freq = 4 # update every four measurements
        self.KF_veh_width = 128

    def load_txt_data(self, data_file, flip=False):
        """
        This function reads the data from the saved txt file.
        The data format in the file is assumed to be:
            - timestamps|pir_data(row by row)|ultra|Tamb1|Tamb2
        :param data_file: data file name
        :return: a pandas data frame. The index is timestamps,
                 the columns are [pir_0x0, pir_1x0, ..., ultra, Tamb_1, Tamb_2]
        """

        timestamps = []
        all_data = []

        # set the column labels
        # first 128 columns are PIR sensor data, then ultrasonic sensor data and then two ambient temperature
        columns = []
        for col in range(0, self.pir_res[1]):
            for row in range(0, self.pir_res[0]):
                columns.append('pir_{0}x{1}'.format(row, col))
        columns = columns + ['ultra', 'Tamb_1', 'Tamb_2']

        if data_file is not None and exists(data_file):

            with open(data_file, 'r') as f:
                for line in f:
                    item = line.strip().split('|')
                    timestamps.append( str2time(item[0]) )

                    # get the pir sensor data
                    # # the pir sensor data 4x32 was saved row by row
                    val = [float(i) for i in item[1].split(',')]

                    # NOTE: verified sensor s2 has two pir arrays misconnected, hence need to swap first 64 values with
                    # last 64 values.
                    tmp_pir_data = np.array(val).reshape(self.pir_res).T.reshape(self.pir_res[0]*self.pir_res[1])

                    # ----------------------------------------------------
                    # ONLY for sensor s1 which has two arrays misconnected.
                    if flip is True:
                        tmp = deepcopy(tmp_pir_data[0:64])
                        tmp_pir_data[0:64] = tmp_pir_data[64:]
                        tmp_pir_data[64:] = tmp
                    # ----------------------------------------------------

                    pir_data = list(tmp_pir_data)

                    # get the ultrasonic sensor data
                    ultra_data = float(item[2])
                    Tamb1 = float(item[3])
                    Tamb2 = float(item[4])

                    all_data.append(pir_data + [ultra_data, Tamb1, Tamb2])

        # save in dataframe
        df = pd.DataFrame(data=all_data, index=timestamps, columns=columns)

        return df

    # @profile
    def batch_normalization(self, raw_data, t_start=None, t_end=None,
                            p_outlier=0.01, stop_thres=(0.1,0.01), window_s=5.0, step_s=1.0):
        """
        This function runs a batch normalization of the raw_data
        :param raw_data: pd.DataFrame, (as reference)
        :param t_start: datetime, start time for the period
        :param t_end: datetime
        :param p_outlier: (1-p_outlier) for getting the noise distribution, see self._get_noise_distribution
        :param stop_thres:stop threshold, see self._get_noise_distribution
        :param window_s: seconds, the window size for computing the mean and std
        :param step_s: seconds, the time step for updating the mean and std
        :return:
        """
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # save the batch normalized data
        frames = raw_data.index[ (raw_data.index >= t_start) & (raw_data.index <= t_end) ]
        norm_data = deepcopy(raw_data.loc[frames,:])

        if window_s is None:
            # then use the entire period as the window
            _, _, noise_means, noise_stds = self._get_noise_distribution(norm_data, t_start=t_start, t_end=t_end,
                                                                         p_outlier=p_outlier, stop_thres=stop_thres,
                                                                         pixels=None)
            norm_data.iloc[:, 0:self.tot_pix].values[:,:] = \
                (norm_data.iloc[:, 0:self.tot_pix].values -
                 noise_means.T.reshape(self.tot_pix))/noise_stds.T.reshape(self.tot_pix)

        else:
            # The background noise distribution is updated at t at window [t-dw, t]
            dw = timedelta(seconds=window_s)
            last_update_t = norm_data.index[0]-timedelta(seconds=2*step_s)
            for i, cur_t in enumerate(norm_data.index):

                if cur_t - last_update_t >= timedelta(seconds=step_s):
                    # update the noise mean and std
                    _, _, noise_means, noise_stds = self._get_noise_distribution(raw_data,
                                                                             t_start=np.max([t_start, cur_t-dw]),
                                                                             t_end =np.min([t_end, cur_t]),
                                                                             p_outlier=p_outlier, stop_thres=stop_thres,
                                                                             pixels=None)
                    last_update_t = cur_t
                # normalize this frame data use computed mean and std
                norm_data.ix[cur_t, 0:self.tot_pix].values[:] = (norm_data.ix[cur_t, 0:self.tot_pix].values -
                                                                 noise_means.T.reshape(self.tot_pix))/\
                                                                noise_stds.T.reshape(self.tot_pix)

                print_loop_status('Normalizing frame:', i, len(norm_data))

            print('\n')

        return norm_data

    # @profile
    def subtract_background_KF(self, raw_data, t_start=None, t_end=None, init_s=300, veh_pt_thres=5, noise_pt_thres=5,
                            prob_int=0.95, pixels=None, debug=False):
        """
        This function models the background noise mean as a random walk process and uses a KF to track the mean.
        The std of PIR measurement was found to be constant, hence it suffices to just track the mean.
        :param raw_data: the raw DataFrame data, NOTE: the ultrasonic sensor data is also in there
        :param t_start: datetime, start time for subtracting background
        :param t_end: datetime, end time for subtracting background
        :param init_s: int, seconds, the time used for initializing the background noise distribution
        :param veh_pt_thres: the minimum number of point to be considered as a vehicle
        :param noise_pt_thres: the minimum number of noise point to close a detection cycle.
        :param prob_int: the confidence interval used to determine whether a point is noise or vehicle
        :param pixels: the list of pixels for performing the background subtraction
        :return:
        """

        _debug = debug
        if _debug:
            _debug_mu = []

        # only normalize those period of data
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # save the batch normalized data
        frames = raw_data.index[ (raw_data.index >= t_start) & (raw_data.index <= t_end) ]
        _raw_data = raw_data.loc[frames,:]
        # set the data to be all np.nan
        veh_data = deepcopy(_raw_data)
        veh_data.values[:, 0:self.tot_pix] = np.nan

        # Only subtracting background for the specified pixels
        if pixels is None:
            # all pixels
            pixels_to_process = np.arange(0, self.tot_pix)
            # _row, _col = np.meshgrid( np.arange(0, self.pir_res[0]), np.arange(0, self.pir_res[1]) )
            # pixels = zip(_row.flatten(), _col.flatten())
        else:
            pixels_to_process = []
            for pix_loc in pixels:
                pixels_to_process.append(pix_loc[1]*4+pix_loc[0])

        # ------------------------------------------------------------------------
        # Initialize the background noise distribution.
        t_init_end = t_start + timedelta(seconds=init_s)
        _, _, noise_means, noise_stds = self._get_noise_distribution(_raw_data, t_start=t_start,
                                                                     t_end=t_init_end, p_outlier=0.01,
                                                                     stop_thres=(0.001, 0.0001), pixels=pixels)

        # ------------------------------------------------------------------------
        # the current noise distribution
        n_mu_all = noise_means.T.reshape(self.tot_pix)
        n_sigma_all = noise_stds.T.reshape(self.tot_pix)
        sig_ratio = stats.norm.ppf(1-(1-prob_int)/2.0, 0, 1)    # the ratio of determining the veh or noise
        _t_init_end = _raw_data.index[np.where(_raw_data.index>t_init_end)[0][0]]
        _t_end = _raw_data.index[np.where(_raw_data.index<=t_end)[0][-1]]

        # Save the reference, which can speed up the code
        norm_data = _raw_data.ix[_t_init_end:_t_end]
        num_frames = len(norm_data.index)

        # ------------------------------------------------------------------------
        # For each pixel, run a FSM to subtract the background
        for pix in pixels_to_process:
            mu = n_mu_all[pix]      # mu will be updated using KF
            sig = 1.0*n_sigma_all[pix]  # sigma will be a constant

            # ------------------------------------------------------------------------
            # Finite State Machine
            # state: 0-background; 1:entering a vehicle detection cycle; -1:exiting a detectin cycle
            # v_buf = []    :vehicle buffer of the time of measurement
            # n_buf = []    :noise buffer of the time of measurement
            # v_counter     :the counter of vehicle points in this detection cycle
            # n_counter     :the counter of noise point in this detection cycle
            # noise         :a cache to prevent updating too frequently
            state = 0
            v_counter = 0
            n_counter = 0
            v_buf = []
            n_buf = []
            noise = []
            meas_buf = []

            # ------------------------------------------------------------------------
            # Kalman filter
            kf = KF_1d(1, self.Q, sig**2)
            kf.initialize_states(mu,sig**2)   # initialize state as current mu

            # ------------------------------------------------------------------------
            # start FSM and KF
            loop_c = 0
            for cur_t, sample_row in norm_data.iterrows():

                # update the mean using only the background noise
                if len(noise) >= self.KF_update_freq:
                    mu = kf.update_state_sequence(norm_data.ix[noise,pix])
                    noise = []
                    # keep track of the updated mean
                    if _debug: _debug_mu.append([cur_t, mu])

                # update the mean using all the measurements
                # if len(meas_buf) >= 16:
                #     mu = kf.update_state_sequence(norm_data.ix[meas_buf,pix])
                #     meas_buf = []
                #     # keep track of the updated mean
                #     if _debug: _debug_mu.append([cur_t, mu])

                meas_buf.append(cur_t)

                # check if current point is noise
                meas = sample_row.values[pix]
                is_noise = (meas>=mu-sig_ratio*sig) & (meas<=mu+sig_ratio*sig)

                # --------------------------------------------------------------------
                # State 0, in background mode
                if state == 0:
                    if is_noise:
                        # append the noise into noise buffer for updates
                        noise.append(cur_t)
                    else:
                        # enter the vehicle detection cycle
                        # put the current measurement in the vehicle buffer
                        v_buf.append(cur_t)
                        v_counter = 1
                        state = 1

                # --------------------------------------------------------------------
                # State 1, at the detection cycle of a vehicle
                if state == 1:
                    if is_noise:
                        # attempt to exiting a detection cycle
                        state = -1
                        n_buf.append(cur_t)
                        n_counter += 1
                    else:
                        # still in detection cycle
                        v_buf.append(cur_t)
                        v_counter += 1

                # --------------------------------------------------------------------
                # state -1, exiting the detection cycle of a vehicle
                if state == -1:
                    if is_noise:
                        # another noise measurement
                        n_buf.append(cur_t)
                        n_counter += 1

                        # now determine if should indeed exit the detection cycle
                        if n_counter >= noise_pt_thres:
                            # if number of consecutive noise data points is sufficient to close the detection cycle
                            if v_counter >= veh_pt_thres:
                                # if the number of points in this cycle indeed support one vehicle

                                # TODO: debug, if v_buf is greater than 2 s (128 samples), then consider as a
                                # stopped vehicle or moving cloud effect and only append the first 2 s
                                if len(v_buf) > self.KF_veh_width:
                                    v_buf = v_buf[0:32]

                                if not _debug:
                                    # return normalized data
                                    veh_data.ix[v_buf, pix] = np.abs((norm_data.ix[v_buf, pix].values-mu)/sig)
                                else:
                                    veh_data.ix[v_buf, pix] = norm_data.ix[v_buf, pix].values

                            else:
                                # the veh points detected in this cycle do not support a vehicle, then they are noise
                                n_buf += v_buf

                            # To update the noise distribution
                            noise += n_buf

                            # reset state
                            n_buf = []
                            v_buf = []
                            n_counter = 0
                            v_counter = 0

                    else:
                        # return to the vehicle detection cycle
                        v_buf.append(cur_t)
                        v_counter += 1

                        # clear the noise buffer, since those measurements are considered part of the vehicle or unidentified
                        # DO NOT reset the counter, otherwise could never exit the cycle in certain cases
                        n_buf = []
                        state = 1

                print_loop_status('Pixel {0}, Subtracting background for frame:'.format(pix), loop_c, num_frames)
                loop_c+=1

            if _debug:
                _debug_mu = np.array(_debug_mu)
                # plot a single pixel
                fig, ax = plt.subplots(figsize=(18,5))
                ax.plot(norm_data.index, norm_data['pir_{0}x{1}'.format(int(pix%4), int(pix/4))],
                        label='raw')
                ax.plot(veh_data.index, veh_data['pir_{0}x{1}'.format(int(pix%4), int(pix/4))],
                        label='veh', linewidth=3)
                # ax.scatter(veh_data.index, veh_data['pir_{0}x{1}'.format(int(pix%4), int(pix/4))])
                ax.plot( _debug_mu[:,0], _debug_mu[:,1], label='noise mu', marker='*', linewidth=2)
                ax.set_title('Pixel {0}'.format(pix))

                ax.legend()
                plt.draw()

        return veh_data



    def _get_noise_distribution(self, raw_data, t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.001,0.0001),
                                pixels=None):
        """
        This function computes the mean and std of noise distribution (normal) by iteratively fitting a normal
            distribution and throwing away points outsize of (1-p_outlier) confidence interval
        :param raw_data: the raw data, df, (as reference)
        :param t_start: datetime, start time of the period for getting the mean and std
        :param t_end: datetime
        :param p_outlier: the confidence interval is (1-p_outlier),
        :param stop_thres: (d_mean, d_std), stop iteration if the change from last distribution < stop_thres
        :param pixels: list of tuples, which pixel to compute
        :return: means, stds, noise_means, noise_stds; each is 4x32 array
        """
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start = raw_data.index[np.where(raw_data.index>=t_start)[0][0]]
        _t_end = raw_data.index[np.where(raw_data.index<=t_end)[0][-1]]

        means = np.ones(self.pir_res)*np.nan
        stds = np.ones(self.pir_res)*np.nan
        noise_means = np.ones(self.pir_res)*np.nan
        noise_stds = np.ones(self.pir_res)*np.nan

        if pixels is None:
            _row, _col = np.meshgrid( np.arange(0, self.pir_res[0]), np.arange(0, self.pir_res[1]) )
            pixels = zip(_row.flatten(), _col.flatten())

        # update each pixel
        for row, col in pixels:

            # get the time series in window
            time_series = raw_data.loc[_t_start:_t_end, 'pir_{0}x{1}'.format(row, col)].values

            # save the initial overall estimate
            means[row, col] = np.nanmean( time_series )
            stds[row, col] = np.nanstd( time_series )

            _pre_mean = means[row, col]
            _pre_std = stds[row, col]

            # converge to the true noise mean
            for i in range(0, 100):
                # if row ==2 and col == 24:
                #     print('updating noise {0}'.format(i))
                # =======================================================
                # throw out the outliers to get a new estimate of mean and std
                # Pr( x \in [-v_thres, v_thres] ) = 1-p_outlier
                v_thres_u = stats.norm.ppf(1-p_outlier/2.0, _pre_mean, _pre_std)
                v_thres_l = _pre_mean - (v_thres_u-_pre_mean)
                _idx = (v_thres_l <= time_series) & ( time_series <= v_thres_u)
                _mean = np.nanmean( time_series[_idx] )
                _std = np.nanstd( time_series[_idx] )

                if np.abs(_mean - _pre_mean) > stop_thres[0] or np.abs(_std - _pre_std) > stop_thres[1]:
                    # have NOT found the converged mean and std
                    _pre_mean = _mean
                    _pre_std = _std
                else:
                    # converged
                    break

            # save converged in the array
            noise_means[row, col] = _mean
            noise_stds[row, col] = _std


        return means, stds, noise_means, noise_stds

    @staticmethod
    def get_data_periods(load_dir, update=True, f_type='txt'):
        """
        This function returns the periods for all data collection experiments saved in a directory
        :param load_dir: the directory
        :param update: True of False, whether to update the periods file
        :param f_type: the type of the data files, txt or npy
        :return: save in file and return the periods
        """
        periods = OrderedDict()
        f_periods = load_dir + 'dataset_periods.cfg'

        if update is True:

            if f_type == 'npy':
                files = glob.glob(load_dir+'*.npy')
            elif f_type == 'txt':
                files = glob.glob(load_dir+'*.txt')
            else:
                raise Exception('Specify file type: txt or npy')

            for f in files:

                # get the sensor config
                sensor_id = f.split('/')[-1].replace('.{0}'.format(f_type), '')

                if f_type == 'npy':
                    d = np.load(f)
                    t_start = d[0][0]
                    t_end = d[-1][0]

                elif f_type == 'txt':
                    with open(f,'r') as fi:
                        first_line = fi.readline()

                        t_start = str2time( first_line.strip().split('|')[0] )
                        print('t_start: {0}'.format(t_start))

                        for line in fi:
                            pass

                        # last line
                        # print 'last' \
                        #       ' line:'
                        # print line
                        t_end = str2time( line.strip().split('|')[0] )
                else:
                    raise Exception('Specify file type: txt or npy')

                periods[sensor_id] = (t_start, t_end)

            # save in a file
            with open(f_periods, 'w') as f:
                for key in periods:
                    f.write('{0},{1},{2}\n'.format(key, time2str(periods[key][0]),
                                                   time2str(periods[key][1]) ))
            print('Updated dataset_periods.cfg.')
        else:

            # load previously extracted file if exists
            if exists(f_periods):
                print('Loading dataset_periods.cfg ...')
                with open(f_periods,'r') as fi:
                    for line in fi:
                        items = line.strip().split(',')
                        periods[items[0]] = ( str2time(items[1]), str2time(items[2]) )
                print('Loaded dataset_periods.cfg.')

            else:
                raise Exception('Previous f_periods not exit.')

        return periods

    @staticmethod
    def plot_heatmap_in_period(data, t_start=None, t_end=None, cbar=None, option='vec', nan_thres_p=None,
                               plot=False, save_dir=None, save_img=False, save_df=False, figsize=(18,8)):
        """
        This function plots the heatmap of the data
        :param data: a pandas dataframe, with index being datetime and columns being 'pir_0x0', ..., 'ultra',..
        :param t_start: datetime to plot
        :param t_end: datetime
        :param cbar: tuple, color bar limit
        :param option: 'vec', 'tworow', 'row_0', 'row_1', 'row_2', 'row_3',
        :param nan_thres_p: only keey data outside of nan_thres_p confidence interval
        :param plot: plot the figure or not
        :param save_dir: dictory for saving
        :param save_img: True, False, if should save the imag
        :param save_df: True, False, if save the DataFrame of this period
        :param figsize: tuple, the figure size
        :return:
        """
        if t_start is None: t_start = data.index[0]
        if t_end is None: t_end = data.index[-1]

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start = data.index[np.where(data.index>=t_start)[0][0]]
        _t_end = data.index[np.where(data.index<=t_end)[0][-1]]

        # parse the option to get the right heatmap
        if option == 'vec':
            columns = [i for i in data.columns if 'pir' in i]
        elif option == 'tworow':
            columns = [i for i in data.columns if 'pir_1x' in i or 'pir_2x' in i]
        elif 'row_' in option:
            row = 'pir_' + option.split('_')[1] + 'x'
            columns = [i for i in data.columns if row in i]
        else:
            raise Exception('Options: vec, tworow, row_0, row_1, row_2, row_3')

        # save the dataframe
        if save_df is True:
            data.ix[_t_start:_t_end, columns].to_csv(save_dir+'heatmap__{0}__{1}.csv'.format(time2str_file(_t_start),
                                                                                           time2str_file(_t_end)))
        heatmap = data.ix[_t_start:_t_end, columns].values.T

        # change all background white noise into nan
        if nan_thres_p is not None:
            print('Warning: Make sure data is normalized if use nan_thres_p')
            v_thres = stats.norm.ppf(1-(1-nan_thres_p)/2.0)
            heatmap[ (heatmap>=-v_thres) & (heatmap<=v_thres) ] = np.nan

        # ---------------------------------------------------------------------------------------------
        # Use imshow to plot the heatmap. Note, this assumes the frequency is constant.
        if cbar is None:
            cbar = (np.nanmin(heatmap), np.nanmax(heatmap))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('auto')
        im = ax.imshow(heatmap, cmap=plt.get_cmap('jet'), interpolation='nearest', aspect='auto',
                        vmin=cbar[0], vmax=cbar[1])
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_title('{0} to {1}'.format(_t_start, _t_end), fontsize=18)
        ax.set_xlabel('Frame count', fontsize=16)
        ax.set_ylabel('Frame pixels', fontsize=16)

        if save_img is True:
            plt.savefig('heatmap__{0}__{1}.png'.format(time2str_file(_t_start), time2str_file(_t_end)),
                        bbox_inches='tight')

        if plot is True:
            plt.draw()
        else:
            plt.clf()
            plt.close()

        return fig, ax

    def plot_noise_evolution(self, data, t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.01,0.1),
                             pixel=None, window_s=120, step_s=30):
        """
        Plot the evolution of the time series noise distribution for pixel
        :param data: pandas dataframe
        :param p_outlier: [0,1], points outside of (1-p_outlier) confidence interval will be considered as outlier
        :param stop_thres: (delta_mu, delta_sigma) degrees in temperature
        :param pixel: tuple, (row, col)
        :param window_s: seconds, the duration of the window for computing the mu and std
        :param step_s: seconds, the step for sliding the window
        :return: returns the axis handle
        """

        if t_start is None: t_start = data.index[0]
        if t_end is None: t_end = data.index[-1]

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start = data.index[np.where(data.index>=t_start)[0][0]]
        _t_end = data.index[np.where(data.index<=t_end)[0][-1]]

        data_to_analyze = data[_t_start:_t_end]

        dw = timedelta(seconds=window_s)
        dt = timedelta(seconds=step_s)
        len_data = len(data_to_analyze)

        timestamps = []
        mus = []
        sigmas = []

        last_update_t = data_to_analyze.index[0]
        for i, cur_t in enumerate(data_to_analyze.index):
            if cur_t - last_update_t >= dt:
                _, _, _mu, _sigma = self._get_noise_distribution(data_to_analyze, t_start=cur_t-dw, t_end=cur_t, p_outlier=p_outlier,
                                                                 stop_thres=stop_thres, pixels=[pixel])

                timestamps.append(cur_t)
                mus.append(_mu[pixel])
                sigmas.append(_sigma[pixel])
                last_update_t = cur_t

                print_loop_status('Evolving noise: ', i, len_data)

        print('\n')
        print('Range of means: {0}~{1}'.format(np.nanmin(mus), np.nanmax(mus)))
        print('Average sigma: {0}'.format(np.nanmean(sigmas)))

        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)

        # ------------------------------------------------------------------
        ax = self.plot_time_series_for_pixel(data_to_analyze, t_start=None, t_end=None, pixels=[pixel])
        # plot the noise mean and std
        ax.plot(timestamps, mus, color='r', linewidth=2)
        ax.plot(timestamps, mus+sigmas, color='r', linestyle='--', linewidth=2)
        ax.plot(timestamps, mus-sigmas, color='r', linestyle='--',linewidth=2)

        plt.draw()

        return ax

    @staticmethod
    def plot_time_series_for_pixel(data, t_start=None, t_end=None, pixels=list()):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param data: pandas dataframe
        :param t_start: datetime type
        :param t_end: datetime type
        :param pixels: list of tuples, [(1,1),(1,5)]
        :return: a figure with all the pixels time series
        """
        if t_start is None: t_start = data.index[0]
        if t_end is None: t_end = data.index[-1]

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start = data.index[np.where(data.index>=t_start)[0][0]]
        _t_end = data.index[np.where(data.index<=t_end)[0][-1]]

        data_to_plot = data.ix[_t_start:_t_end]

        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)

        for pixel in pixels:

            plt.plot(data_to_plot.index, data_to_plot.ix[:,'pir_{0}x{1}'.format(pixel[0], pixel[1])],
                     label='pixel {0}'.format(pixel))

        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.title('Time series')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time')

        plt.legend()
        plt.grid(True)
        plt.draw()

        return ax


    def plot_histogram_for_pixel(self, raw_data, pixels=list(),
                                 t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.001,0.0001)):
        """
        Statistic Analysis:
        This function plots the histogram of the raw data for a selected pixel, to better understand the noise
        :param raw_data: df, the raw data
        :param pixels: list of tuples, [(2,20),()]
        :param t_start: datetime type
        :param t_end: datetime type
        :param p_outlier: the confidence interval is (1-p_outlier)
        :param stop_thres: (d_mean, d_std), stop iteration if the change from last distribution < stop_thres
        :return: one figure for each pixel
        """

        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        _t_start = raw_data.index[np.where(raw_data.index>=t_start)[0][0]]
        _t_end = raw_data.index[np.where(raw_data.index<=t_end)[0][-1]]


        # compute the mean and std
        mu, sigma, noise_mu, noise_sigma = self._get_noise_distribution(raw_data,
                                                                       t_start=_t_start, t_end=_t_end,
                                                                       p_outlier=p_outlier, stop_thres=stop_thres)

        for row, col in pixels:

            # get the time series in window
            time_series = raw_data.loc[_t_start:_t_end, 'pir_{0}x{1}'.format(row, col)].values

            # the histogram of the data
            num_bins = 200
            fig = plt.figure(figsize=(8,5), dpi=100)
            n, bins, patches = plt.hist(time_series, num_bins,
                                        normed=1, facecolor='green', alpha=0.75)

            # add a 'best fit' line
            norm_fit_line = mlab.normpdf(bins, noise_mu[row, col],
                                               noise_sigma[row, col])
            l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1.5, label='Background noise')

            print('PIR pixel {0} x {1}:'.format(row, col))
            print('    All: ({0}, {1})'.format(mu[row, col], sigma[row, col]))
            print('    Noise: ({0}, {1})'.format(noise_mu[row, col], noise_sigma[row, col]))

            # norm_fit_line = mlab.normpdf(bins, mu[row, col],
            #                                    sigma[row, col])
            # l = plt.plot(bins, norm_fit_line, 'b--', linewidth=1.5, label='all')

            plt.legend()
            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Probability density')
            plt.title('Distribution of data from pixel {0} x {1}'.format(row, col))
            # plt.grid(True)

        plt.draw()



class KF_1d:
    """
    The system is a one-dimentional KF:
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
        self.Q = Q
        self.R = R

        # states. preallocate memory
        self.x = 0.0
        self.P = 0.0

    def initialize_states(self, x0, P0):
        """
        This function initializes the initial state
        :param x0: the initial state
        :param P0: the initial error covariance matrix
        :return: initialized into property
        """
        self.x = x0
        self.P = P0

    # @profile
    def update_state(self, z):
        """
        This function updates the current state given measurement z
        :param z: np array
        :return: the updated system state
        """

        # forward propagate the state
        P_f = self.P + self.Q

        # compute the innovation sequence and Kalman gain
        y = z - self.x  # x_f = x
        S = P_f + self.R
        K = P_f/S

        # update the state
        self.x += K*y
        self.P = (1-K)*P_f

        return self.x


    # def update_state_sequence(self, zs):
    #     """
    #     This function updates the current state given a sequence of measurements zs
    #     :param zs: a sequence of measurement z, num_meas x dim
    #     :return: the current state
    #     """
    #
    #     # update only uses the mean
    #     # the system changes as
    #     # x(k) = x(k-1) + wn    ; wn ~ N(0, n**2*Q)
    #     # z(k) = Ix(k) + vn     ; vn ~ N(0, n*R)    if z(k) is now mean(zs)
    #
    #     for z in zs:
    #         self.update_state(z)
    #
    #     return self.x


    # @profile
    def update_state_sequence(self, zs):
        """
        This function updates the current state given a sequence of measurements zs
        :param zs: a sequence of measurement z, num_meas x dim
        :return: the current state
        """

        # update only uses the mean
        # the system changes as
        # x(k) = x(k-1) + wn    ; wn ~ N(0, n**2*Q)
        # z(k) = Ix(k) + vn     ; vn ~ N(0, n*R)    if z(k) is now mean(zs)

        n = len(zs)
        # print('update mean with {0} pts'.format(n))
        Q = n**2*self.Q
        R = n*self.R

        avg_z = np.mean(zs)

        # forward propagate the state
        P_f = self.P +  Q

        # compute the innovation sequence and Kalman gain
        y = avg_z - self.x
        S = P_f + R
        K = P_f/S

        # update the state
        self.x += K*y
        self.P = (1-K)*P_f

        return self.x



