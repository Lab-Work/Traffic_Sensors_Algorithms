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
This class is used for vehicle detection and speed estimation.
"""

# ==================================================================================================================
# ==================================================================================================================
"""
Define some utility functions that will be shared among all classes.
"""
# ==================================================================================================================
# ==================================================================================================================
def time2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def str2time(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")

def time2str_file(dt):
    return dt.strftime("%Y%m%d_%H%M%S_%f")

def str2time_file(dt_str):
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
    v_thres_u = stats.norm.ppf(1-(1-prob)/2.0, mu, sigma)
    v_thres_l = mu - (v_thres_u - mu)

    return (data>=v_thres_l) & (data<=v_thres_u)

def in_2sigma(data, mu, sigma):
    return (data>=mu-2.0*sigma) & (data<=mu+2.0*sigma)

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
# ==================================================================================================================
"""
Classes:
 - TrafficSensorAlg: master level class, which handels the data collection and preprocessing
 - VehDet: vehicle detection class, which only handles vehicle detection
 - SpeedEst: speed estimation class, which estimates the speed
 - SensorData: the data class which preprocess, analyze, and visualize the data
"""
# ==================================================================================================================
# ==================================================================================================================
class TrafficSensorAlg:
    """
    This class is the top layer which handels the data collection and preprocessing, including:
        - reading data from source (e.g., file or streaming)
        - initialize the background distribution
        - normalize each frame as data streaming in
        - periodically update the background distribution
        - cache the past xx seconds cleaned data which will be passed to veh_det and speed_est classes
    """

    def __init__(self, pir_res=(4,32)):
        """
        Initialize the function with the data source and output options
        :return:
        """
        self.vehs = []
        self.pir_res=pir_res

    #TODO: to finish the online version
    def run(self, norm_data, buffer_s=1.5, step_s=0.5, t_start=None, t_end=None):

        # get the window and step in timedelta
        buffer_dt = timedelta(seconds=buffer_s)
        step_dt = timedelta(seconds=step_s)

        # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        if t_start is None: t_start = norm_data.index[0]
        if t_end is None: t_end=norm_data.index[-1]

        _t_start = norm_data.index[np.where(norm_data.index>=t_start+buffer_dt)[0][0]]
        _t_end = norm_data.index[np.where(norm_data.index<=t_end)[0][-1]]

        last_refresh_t = t_start
        for cur_t in norm_data.ix[_t_start:_t_end].index:

            if cur_t - last_refresh_t > step_dt:
                # should refresh
                buf = norm_data.ix[ (norm_data.index>=cur_t-buffer_dt) &
                                       (norm_data.index<=cur_t), :]
                last_refresh_t = cur_t

                # --------------------------------------------------------------------
                # run vehicle detection on this buffer


                # --------------------------------------------------------------------
                # run speed estimation on this buffer


                # --------------------------------------------------------------------
                # summarize and output


    def batch_run(self, norm_df, det_thres=600, window_s=2.0, step_s=1.0,
                  save_dir = '../workspace/1013/figs/speed_est_95_ff1/'):

        # --------------------------------------------------------------------
        # first detect the vehicle
        det = VehDet()
        windows = det.batch_detect_veh(norm_df, energy_thres=det_thres, window_s=window_s, step_s=step_s)

        # # plot detected vehicles
        # fig, ax =data.plot_heatmap_in_period(norm_df, t_start=None, t_end=None, cbar=(0,4),
        #                                       option='vec', nan_thres_p=None, plot=True, save_dir=save_dir, save_img=False,
        #                                       save_df=False, figsize=(18,8))
        # for veh in vehs:
        #     # t_start and t_end may not be in the index, hence replace by _t_start >= t_start and _t_end <= t_end
        #     _t_start_idx = np.where(norm_df.index>=veh[0])[0][0]
        #     _t_end_idx = np.where(norm_df.index<=veh[1])[0][-1]
        #     # ax.fill_between(norm_df.index, 0, 127, where=((norm_df.index>=veh[0]) & (norm_df.index<=veh[1])), facecolor='green',alpha=0.5  )
        #     rect = patches.Rectangle((_t_start_idx, 0), _t_end_idx-_t_start_idx, 128, linewidth=1, edgecolor='g',
        #                              facecolor=(0,1,0,0.2))
        #     ax.add_patch(rect)

        # --------------------------------------------------------------------
        # speed estimation
        save_dir = save_dir
        if not exists(save_dir): os.mkdir(save_dir)

        # print('Estimating speed for {0} vehs'.format(len(vehs)))
        for win in windows:
            est = SpeedEst(norm_df.ix[win[0]:win[1]], pir_res=(4,32), plot=False, save_dir=save_dir)
            vehs_in_win = est.estimate_speed(stop_tol=(0.002, 0.01), dist=3.5, r2_thres=0.8, min_num_pts=200)

            for veh in vehs_in_win:
                # register the vehicle to self.veh list
                self.register_veh(veh)

        # --------------------------------------------------------------------
        # save the final result in form
        np.save(save_dir+'detected_vehs.npy',self.vehs)


    def register_veh(self, veh):
        """
        This function checks if veh has been previously detected in earier time windows. If so, compare and only register
        the better one.
        :param veh: veh class object
        :return:
        """

        if len(self.vehs) == 0:
            self.vehs.append(veh)
        else:
            for v in self.vehs:
                if self.are_same_veh(v, veh):
                    # replace the old estimate if the new estimate is better
                    if v.det_perc < veh.det_perc:
                        # update the registered vehicles
                        print('######################## Updated vehicle entering at {0}\n'.format(veh.t_in))
                        self.vehs.remove(v)
                        self.vehs.append(veh)
                        return 0
                    else:
                        # duplicated vehicle, but old vehicle has better estimates, then ignore this new estimates
                        print('######################## Discarded duplicated vehicle entering at {0}\n'.format(veh.t_in))
                        return 0

            # if not returned yet, meaning no duplicated vehicle, than register.
            self.vehs.append(veh)


    def are_same_veh(self, v1, v2):
        if v1.t_out <= v2.t_in or v1.t_in >= v2.t_out:
            return False

        # use the amount of overlapping of supporting data point to determine if they are the same vehicle.
        thres = 0.5
        overlapping_pts = [p for p in set(v1.pts) & set(v2.pts)]

        # scatter and check overlapping
        # t_s = v1.pts[0][0]
        # v1.pts = np.asarray(v1.pts)
        # v2.pts = np.asarray(v2.pts)
        # v1_t = [(t-t_s).total_seconds() for t in v1.pts[:,0]]
        # v2_t = [(t-t_s).total_seconds() for t in v2.pts[:,0]]
        # plt.scatter(v1_t, v1.pts[:,1], color='b', alpha=0.5)
        # plt.scatter(v2_t, v2.pts[:,1], color='r', alpha=0.5)

        overlapping_perc = float(len(overlapping_pts))/np.min([len(set(v1.pts)), len(set(v2.pts))])

        if overlapping_perc >= thres:
            print('########## Found duplicated vehicles')
            print('                  v1: ({0}, {1})'.format(v1.t_in, v1.t_out))
            print('                  v2: ({0}, {1})'.format(v2.t_in, v2.t_out))
            return True
        else:
            return False


    def plot_detected_vehs(self, norm_df, ratio_tx=6.0):
        # plot the initial figure
        fig, ax = plt.subplots(figsize=(15,8))
        ax.set_aspect('auto')

        # ==============================================================================
        # plot all the data point, perform nonlinear transform
        t_start = norm_df.index[0]
        print('\n########################## Plotting the detected vehicles. t_start = {0}'.format(t_start))
        pts, t_grid, x_grid = self.nonlinear_trans(norm_df, ratio_tx=ratio_tx)
        ax.scatter(pts[:,0], pts[:,1], color='0.6')

        # ==============================================================================
        # plot the detected vehicles
        # --------------------------------------------------------------------
        # scatter the estimated converged models
        colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
        for i, veh in enumerate(self.vehs):

            # plot the supporting points
            sup_pts = np.array( [[(p[0]-t_start).total_seconds(), p[1]] for p in veh.pts] )
            ax.scatter(sup_pts[:,0], sup_pts[:,1], color=next(colors), alpha=0.75)

            # plot the fitted line
            t_in_s = (veh.t_in-t_start).total_seconds()
            t_out_s = (veh.t_out-t_start).total_seconds()
            ax.plot([t_in_s, t_out_s],[x_grid[0], x_grid[-1]], linewidth=2, color='k')

        ax.set_title('All detected vehicles', fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Space (x 6d = m)', fontsize=18)
        ax.set_xlim([t_grid[0], t_grid[-1]])
        ax.set_ylim([x_grid[-1], x_grid[0]])
        plt.draw()


    def nonlinear_trans(self, norm_df, ratio_tx):
        """
        This function performs the nonlinear transform of norm_df.
        :param norm_df: the dataframe
        :param ratio_tx: the ratio to normalize time and space dimensions. Speed (mph)=2.24*ratio_tx*distantce*slope
        :return: pts (np array), t_grid, x_grid
        """

        # ------------------------------------------------------------
        # initialize the space grid with nonlinear transformation
        # Slope * self.ratio * distance(m) = m/s
        _dup = self.pir_res[0]
        d_theta = (60.0 / 16) * np.pi / 180.0
        x_grid = []
        for i in range(-16, 16):
            for d in range(0, _dup):
                # duplicate the nonlinear operator for vec
                x_grid.append(np.tan(d_theta / 2 + i * d_theta))
        x_grid = -np.asarray(x_grid)/ratio_tx

        # ------------------------------------------------------------
        # initialize the time grid in seconds
        t_start = norm_df.index[0]
        t_end = norm_df.index[-1]
        t_grid = [(t-t_start).total_seconds() for t in norm_df.index]
        t_grid = np.asarray(t_grid)

        # ------------------------------------------------------------
        # convert the matrix to a list of data point tuples
        pt_time = []
        pt_space = []
        i = 0
        for cur_t, row in norm_df.iterrows():
            for col in range(0, self.pir_res[0]*self.pir_res[1]):
                if ~np.isnan(row.values[col]):
                    pt_time.append(t_grid[i])
                    pt_space.append(x_grid[col])
            i += 1

        pts = np.array(zip(pt_time, pt_space))

        return pts, t_grid, x_grid


    def plot_video_frames(self, video_file, raw_df, norm_df, det_vehs, save_dir):
        """
        This function plots the video frames. Use the following command to generate a video
        :param video_file:
        :param raw_df:
        :param norm_df:
        :param det_vehs:
        :return:
        """
        pass

# ==================================================================================================================
# vehicle detection class
# ==================================================================================================================
class VehDet:
    def __init__(self):
        pass

    def batch_detect_veh(self, norm_data, energy_thres=600, window_s=1.5, step_s=0.5):

        pir_data = deepcopy(norm_data.ix[:, [i for i in norm_data.columns if 'pir' in i]])
        # replace the nan value to 0.0
        pir_data.values[np.isnan(pir_data.values)] = 0.0

        ultra_data = norm_data.ix[:, 'ultra']

        # ---------------------------------------------------------------------------------
        # compute the energy in each 0.5 window
        dw = timedelta(seconds=window_s)
        ds = timedelta(seconds=step_s)
        t_s = pir_data.index[0]
        t_e = pir_data.index[0] + dw

        veh_t = []
        veh_e = []

        vehs = []
        while t_e <= pir_data.index[-1]:
            e = np.sum(pir_data.ix[(pir_data.index>=t_s) & (pir_data.index<=t_e)].values)
            veh_t.append(t_s)
            veh_e.append(e)
            t_s = t_s + ds
            t_e = t_s + dw

            if e>=energy_thres:
                # thresholding
                vehs.append((t_s, t_e))
        return vehs
        # plot the data
        # fig, ax = plt.subplots(figsize=(18,6))
        # ax.plot(pir_data.index, pir_data.mean(1), linewidth=1, label='pir')
        # # ax.plot(ultra_data.index, ultra_data.values, linewidth=1, label='ultra')
        # ax.step(veh_t, np.array(veh_e)/3000.0, where='post', linestyle='--', color='r')
        # ax.plot([veh_t[0], veh_t[-1]], [0.1, 0.1], 'g')
        # ax.legend()
        # plt.draw()


# ==================================================================================================================
# Vehicle class
# ==================================================================================================================
class Veh:
    """
    This class defines the basic attributes for each vehicle. Additional functions may be added.
    """
    def __init__(self, slope=None, t_in=None, t_out=None, pts=None, r2=None, sigma=None, frame_loc=None,
                 det_perc=None, det_window=None):

        self.slope = slope
        self.t_in = t_in
        self.t_out = t_out
        self.pts = pts
        self.r2 = r2
        self.sigma = sigma
        self.frame_loc = frame_loc
        self.det_perc = det_perc
        self.det_window = det_window



# ==================================================================================================================
# Speed estimation class
# ==================================================================================================================
class SpeedEst:
    def __init__(self, data_df, pir_res=(4,32), plot=False, save_dir=''):

        # ------------------------------------------------------------
        # initialize the space grid with nonlinear transformation
        # Slope * self.ratio * distance(m) = m/s
        self.ratio_tx = 6.0  # to make the space and time look equal
        _dup = pir_res[0]
        d_theta = (60.0 / 16) * np.pi / 180.0
        x_grid = []
        for i in range(-16, 16):
            for d in range(0, _dup):
                # duplicate the nonlinear operator for vec
                x_grid.append(np.tan(d_theta / 2 + i * d_theta))
        x_grid = -np.asarray(x_grid)/self.ratio_tx
        self.x_grid = x_grid

        # ------------------------------------------------------------
        # initialize the time grid in seconds
        self.init_dt = data_df.index[0]
        self.end_dt = data_df.index[-1]
        t_grid = [(t-data_df.index[0]).total_seconds() for t in data_df.index]
        self.t_grid = t_grid

        # ------------------------------------------------------------
        # convert the matrix to a list of data point tuples
        self.time = []
        self.space = []
        i = 0
        for cur_t, row in data_df.iterrows():
            for col in range(0, pir_res[0]*pir_res[1]):
                if ~np.isnan(row.values[col]):
                    self.time.append(t_grid[i])
                    self.space.append(x_grid[col])
            i += 1
        self.time = np.asarray(self.time)
        self.space = np.asarray(self.space)

        # ------------------------------------------------------------
        # other properties
        self.plot = plot
        self.save_dir = save_dir

        # each mdl: (line, sigma, inlier_idx, r2)
        self.all_mdls = []
        self.labeled_pts = np.array([]).astype(int)
        self.all_vehs = []

    def estimate_speed(self, stop_tol=(0.002, 0.01), dist=3.5, r2_thres=0.85, min_num_pts=150,
                       speed_range=(0,50)):

        # first use RANSAC to get the clusters
        clusters = self.get_clusters(db_radius=0.05, db_min_size=30, min_num_pts=min_num_pts)

        if len(clusters) == 0:
            # Meaning all pts in frame is < min_num_pts, then skip
            print('########################## Did not find vehicles in this frame starting at: {0}\n'.format(time2str_file(self.init_dt)))
            return []
        else:
            # for each cluster
            for counter, inlier_idx in enumerate(clusters):
                print('\n===========================================================================')
                print('-- Cluster {0}: {1} pts'.format(counter, len(inlier_idx)))

                self.fit_mixed_mdls(initial_idx=inlier_idx, stop_tol=stop_tol, dist=dist,
                                    counter=str(counter), r2_thres=r2_thres, min_num_pts=min_num_pts,
                                    speed_range=speed_range)

            # check remaining points
            if len(self.time) - len(self.labeled_pts) >= min_num_pts:
                # remaining points may still support one car
                rem_idx = np.arange(0, len(self.time), 1).astype(int)
                # removed labeled points
                rem_idx = np.delete(rem_idx, self.labeled_pts)
                print('\n===========================================================================')
                print('-- Checking remaining points: {0}'.format(len(rem_idx)))
                self.fit_mixed_mdls(initial_idx=rem_idx, stop_tol=stop_tol, dist=dist,
                                    counter='rem', r2_thres=r2_thres, min_num_pts=min_num_pts,
                                    speed_range=speed_range)

            # ================================================================================================
            # plot final estimates
            print('########################## Finished estimation for frame starting at: {0}\n'.format(time2str_file(self.init_dt)))
            self.plot_progress(cur_mdl=None, save_name='{0}'.format(time2str_file(self.init_dt)),
                               title='{0}'.format(self.init_dt),
                               sig_ratio=2.0, dist=dist)

            # ================================================================================================
            # clean up fitted models by merging models and removing wrong direction
            cleaned = self.clean_veh(dt_s=0.3,stop_tol=stop_tol)
            if cleaned:
                self.plot_progress(cur_mdl=None, save_name='{0}_cleaned'.format(time2str_file(self.init_dt)),
                               title='{0}'.format(self.init_dt),
                               sig_ratio=2.0, dist=dist)

            # ================================================================================================
            # convert to class
            for mdl in self.all_mdls:
                veh = self.mdl2veh(mdl)
                self.all_vehs.append(veh)

            return self.all_vehs


    def clean_veh(self, dt_s=0.3, stop_tol=(0.002, 0.01)):
        """
        This function clean the estimated vehicle models.
            - remove those that are at the wrong direction
            - merge vehicles that are too close ts1-ts2 <= dt_s and te1-te2<= dt_s
        :param dt_s:
        :return:
        """
        flag = False

        if len(self.all_mdls) == 1:
            # remove the wrong direction mdl
            if self.all_mdls[0]['line'][0] > 0:
                self.all_mdls = []
                flag = True

        elif len(self.all_mdls) >=2:
            # ========================================================================
            # first convert the models to a format to be returned except the datetime
            # [enter_time, exit_time, slope, points, sigma, r2]
            mdls = []
            for mdl in self.all_mdls:
                # only consider the correct direction
                if mdl['line'][0] > 0:
                    flag = True
                    print('########################## debug: slope removed: {0}'.format(mdl['line'][0]))
                    continue

                # compute the enter and exit time.
                x_s = (self.x_grid[0]-mdl['line'][1])/mdl['line'][0]
                x_e = (self.x_grid[-1]-mdl['line'][1])/mdl['line'][0]

                mdls.append([x_s, x_e, mdl['line'][0], mdl['inlier_idx'], mdl['sigma'], mdl['r2']])

            # ========================================================================
            # Use DBSCAN to find the models that to be merged
            ts_te = [i[0:2] for i in mdls]

            y_pre = DBSCAN(eps=dt_s, min_samples=1).fit_predict(ts_te)
            num_clusters = len(set(y_pre)) - (1 if -1 in y_pre else 0)
            y_pre = np.asarray(y_pre)
            print('########################## debug: y_pre: {0}'.format(y_pre))

            # ========================================================================
            # clean to final models in self.all_mdls format: (line, sigma, inlier_idx, r2)
            final_mdls = []
            for clus in range(0, num_clusters):
                n_mdls = sum(y_pre==clus)

                if n_mdls == 1:
                    # append the original model
                    idx = [i for i,x in enumerate(y_pre) if x==clus ]
                    mdl = mdls[idx[0]]

                    k = mdl[2]
                    c = self.x_grid[0] - k*mdl[0]

                    _mdl = {'line':(k,c), 'inlier_idx':mdl[3], 'sigma':mdl[4], 'r2':mdl[5]}
                    final_mdls.append(_mdl)
                else:
                    # merge the idx
                    _merge_idx = []
                    idx = [i for i,x in enumerate(y_pre) if x==clus ]
                    if len(idx) >1:
                        flag = True
                    for i in idx:
                        mdl = mdls[i]
                        _merge_idx += mdl[3].tolist()
                    _merge_idx = np.asarray(_merge_idx)

                    # fit a new model for each
                    # first remove labeled point from self.labeled points before fit models
                    _labeled_pts = list(self.labeled_pts)
                    for idx in _merge_idx:
                        _labeled_pts.remove(idx)
                    self.labeled_pts = np.array(_labeled_pts).astype(int)
                    _mdl = self.fit_mdl(_merge_idx, stop_tol=(np.inf, np.inf), dist=3.5, counter='0')

                    final_mdls.append(_mdl)

            # ========================================================================
            # replace self.all_mdls
            if flag is True:
                self.all_mdls = final_mdls

        return flag


    def mdl2veh(self, mdl):
        """
        This function converts the mdl dictionary to standard veh class
        :param mdl: dict
        :return: veh class object
        """
         # compute the enter and exit time.
        in_s = (self.x_grid[0]-mdl['line'][1])/mdl['line'][0]
        out_s = (self.x_grid[-1]-mdl['line'][1])/mdl['line'][0]

        t_in = self.init_dt + timedelta(seconds=in_s)
        t_out = self.init_dt + timedelta(seconds=out_s)

        _t = self.time[mdl['inlier_idx']]
        pts_t = [self.init_dt+timedelta(seconds=i) for i in _t]

        # determine the frame location
        # compute the percent of the trace in the detection window, which will be used as an indicator on how much the
        # estimated speed should be trusted.
        if in_s >=0 and out_s <= self.t_grid[-1]:
            frame_loc = 'full'
            det_perc = 1.0
        elif in_s >=0 and out_s > self.t_grid[-1]:
            frame_loc = 'head'
            det_perc = (self.t_grid[-1]-in_s)/(out_s-in_s)
        elif in_s <0 and out_s <= self.t_grid[-1]:
            frame_loc = 'head'
            det_perc = (out_s-self.t_grid[0])/(out_s-in_s)
        elif in_s <0 and out_s > self.t_grid[-1]:
            frame_loc = 'body'
            det_perc = (self.t_grid[-1]-self.t_grid[0])/(out_s-in_s)

        veh = Veh(slope=mdl['line'][0], t_in=t_in, t_out=t_out, pts=zip(pts_t, self.space[mdl['inlier_idx']]),
                  r2=mdl['r2'], sigma=mdl['sigma'], frame_loc=frame_loc, det_perc=det_perc,
                  det_window=(self.init_dt, self.end_dt))

        return veh


    def fit_mixed_mdls(self, initial_idx, stop_tol=(0.0001, 0.01),
                       dist=3.5, counter='0', r2_thres=0.8,
                       min_num_pts=100, top_n=8, min_init_pts=10, speed_range=(0,50)):

        mdl = self.fit_mdl(initial_idx, stop_tol=stop_tol,
                           dist=dist, counter=str(counter))

        if mdl is not None:

            r2 = mdl['r2']
            if r2 < r2_thres and len(mdl['inlier_idx']) >= min_num_pts:
                # a bad fitting but has a lot of points which could potentially be multiple vehicles
                # then split the cluster by projection
                print('\n$$$$ Bad fitting (r2 = {0}) with {1} pts.\n$$$$ Splitting to subclusters...'.format(r2, len(mdl['inlier_idx'])))

                # specify candidate slopes to speed up convergence.
                # They could be the current slope, and the slopes of finalized models
                candidate_slopes = [mdl['line'][0]]
                for m in self.all_mdls:
                    candidate_slopes.append(m['line'][0])

                # (lines, sigmas, weight, aic, bic)
                sub_clusters = self.split_cluster_exhaustive(pt_idx=mdl['inlier_idx'], min_num_pts=min_num_pts,
                                                             speed_range=(speed_range[0],speed_range[1],5),
                                                             dist=dist,
                                                             counter=counter, top_n=top_n,
                                                             candidate_slopes= [mdl['line'][0]])
                # sub_clusters = self.split_cluster(mdl[0], mdl[2], min_num_pts=min_num_pts, counter=counter)

                for i, (line, sigma, w, aic, bic) in enumerate(sub_clusters):

                    inlier_idx = self.update_inliers(line, tol=sigma)
                    print('---- Sub-cluster {0} with {1} pts'.format(i, len(inlier_idx)))

                    if len(inlier_idx) < min_init_pts:
                        # requires at least two points to fit a line
                        continue

                    sub_mdl = self.fit_mdl(inlier_idx=inlier_idx,
                                           stop_tol=stop_tol, dist=dist,
                                           counter='sub{0}_{1}'.format(counter, i))

                    if sub_mdl is None:
                        continue

                    # append the converged model only if r2>r2_thres and # pts > min_num_pts
                    r2 = sub_mdl['r2']
                    num_pts = len(sub_mdl['inlier_idx'])
                    if r2 >= r2_thres and num_pts >= min_num_pts:
                        self.all_mdls.append(sub_mdl)
                        # register labeled points
                        self.labeled_pts = np.concatenate([self.labeled_pts, sub_mdl['inlier_idx']])
                        print('$$$$ Good fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))
                    else:
                        print('$$$$ Discarding bad fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))

            elif r2 >= r2_thres and len(mdl['inlier_idx']) >= min_num_pts:
                print('$$$$ Good fitting of {0} pts with r2: {1}'.format(len(mdl['inlier_idx']), r2))
                # a good fitting with strong data support
                # append either the converged line or the last line
                self.all_mdls.append(mdl)
                # register labeled points
                self.labeled_pts = np.concatenate([self.labeled_pts, mdl['inlier_idx']])

            else:
                num_pts = len(mdl['inlier_idx'])
                print('$$$$ Discarding bad fitting of {0} pts with r2: {1}\n'.format(num_pts, r2))


    def fit_mdl(self, inlier_idx, stop_tol=(0.0001, 0.01), dist=3.5, counter='0'):
        """
        This function fit a linear line to the inlier points iteratively until convergence.
        :param inlier_idx: the inital inliers to fit
        :param stop_tol: (slope, intercept) differnce for convergence
        :param dist: the distance of the vehicle, only for illustration purpose
        :param counter: the counter title to be printed or used for saving figures.
        :return: a model dict:
                mdl['line'] = (k,c)
                mdl['sigma'] = sig
                mdl['inlier_idx'] = the final converged inlier index
                mdl['r2'] = the r2 score of the fitting de linear regression
        """

        if len(inlier_idx) < 5:
            return None

        try:

            _pre_line = None
            converged = False
            print('---------- Fitting a line, iterations: ')
            for i in range(0, 100):
                # sys.stdout.write('\r')
                # sys.stdout.write('..... {0},'.format(i))
                # sys.stdout.flush()
                print('               # {0}: {1} pts'.format(i, len(inlier_idx)))

                if len(inlier_idx) < 5:
                    return None

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
                    # # converged
                    # if stop_tol[0] == np.inf:
                    #     print('---------- #################################### inf stop_tol #######')
                    print('---------- Converged: y = {0:.03f}x + {1:.03f}, Speed {2:.01f} mph assuming dist={3:.01f} m'.format(line[0], line[1],
                                                                                                   -dist*2.24*self.ratio_tx * line[0],
                                                                                                                             dist))
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
                                       sig_ratio=sigma_ratio, dist=dist)

                if not converged:
                    # update cluster
                    inlier_idx = self.update_inliers(line, tol=sigma_ratio * sig)
                    _pre_line = deepcopy(line)
                else:
                    mdl={'line':line, 'sigma':sig, 'inlier_idx':inlier_idx, 'r2':r2}
                    return mdl

        except ValueError:
            return None

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

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.scatter(samples[:, 0], samples[:, 1], color='0.6')

        # colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
        #           'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
        #           'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
        #           'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
        #           'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w']
        for cluster_label in range(0, num_clusters):
            clus = (y_pre == cluster_label)
            # ax.scatter(samples[clus, 0], samples[clus, 1], color=colors[cluster_label])
            clusters.append([i for i, x in enumerate(clus) if x])

        # ax.set_title('DBSCAN clustering', fontsize=20)
        # plt.savefig(self.save_dir + 'DBSCAN_cluster.png', bbox_inches='tight')
        # plt.clf()
        # plt.close()

        # return
        return clusters

    def get_sigma(self, line, pt_idx=None):


        if pt_idx is None:
            # if not specified, project all points
            pt_idx = np.arange(0, len(self.time)).astype(int)

        # print('                                                        get_sigma:{0}'.format(len(pt_idx)))

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
                                 dist=3.5, counter='0', top_n=8, candidate_slopes=None):

        # -------------------------------------------------------------------------------------
        # explore all directions of lines with reference to the left bottom corner
        speeds = np.arange(speed_range[0], speed_range[1], speed_range[2]).astype(float)
        slopes = -speeds/(dist*2.24*self.ratio_tx)

        # also explore the candidate directions
        if candidate_slopes is not None:
            slopes = np.concatenate([slopes, np.asarray(candidate_slopes)])
            speeds = np.concatenate([speeds, -np.asarray(candidate_slopes)*(dist*2.24*self.ratio_tx)])

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

    def plot_progress(self, cur_mdl, save_name=None, title=None, sig_ratio=2.0, dist=3.5):

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
            for i, mdl in enumerate(self.all_mdls):
                line = mdl['line']
                sig = mdl['sigma']
                inlier_idx = mdl['inlier_idx']
                r2 = mdl['r2']
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
            line = cur_mdl['line']
            sig = cur_mdl['sigma']
            inlier_idx = cur_mdl['inlier_idx']
            r2 = cur_mdl['r2']
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
        ax0.set_xlim([self.t_grid[0], self.t_grid[-1]])
        ax0.set_ylim([self.x_grid[-1], self.x_grid[0]])

        # ===========================================================================
        # plot ax1: the analysis of the current model
        # - the distribution of the distance of points to the line
        # - the measures: r^2, num_old_inliers, num_new_inliers, total num_points, slope
        # plot histogram
        if cur_mdl is not None:
            # projection, density kernel segmentation, and GMM
            sig, dists = self.get_sigma(cur_mdl['line'], pt_idx=cur_mdl['inlier_idx'])

            sig = cur_mdl['sigma']
            bin_width = 0.005
            n, bins, patches = ax1.hist(dists, bins=np.arange(-3 * sig, 3 * sig, bin_width),
                                        normed=1, facecolor='green', alpha=0.75)
            # fill the one-sig space.
            x_fill = np.linspace(-sig, sig, 100)
            ax1.fill_between(x_fill, 0, mlab.normpdf(x_fill, 0, sig), facecolor='r', alpha=0.65)
            ax1.plot(bins, mlab.normpdf(bins, 0, sig), linewidth=2, c='r')
            text = ' R2: {0:.4f}; Speed: {1:.2f} mph'.format(cur_mdl['r2'],
                                                             -cur_mdl['line'][0] * dist * self.ratio_tx*2.24)
            ax1.annotate(text, xy=(0.05, 0.65), xycoords='axes fraction', fontsize=14)
            ax1.set_ylim([0, np.max(n) * 1.5])
            ax1.set_title('Analyzing current model', fontsize=16)

        elif len(self.all_mdls) != 0:
            # plot the final distribution of all clusters
            offset = 0
            y_lim = 0
            for i, mdl in enumerate(self.all_mdls):

                line = mdl['line']
                sig = mdl['sigma']
                inlier_idx = mdl['inlier_idx']
                r2 = mdl['r2']

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

                text = ' R2: {0:.4f}\n {1:.2f} mph\n #:{2}'.format(r2, -line[0] * dist*2.24*self.ratio_tx,
                                                                   len(inlier_idx))
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

        self._debug_MAP_data_len=[]

    def load_txt_data(self, data_file):
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
                    pir_data = list(np.array(val).reshape(self.pir_res).T.reshape(self.pir_res[0]*self.pir_res[1]))

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
    def subtract_background(self, raw_data, t_start=None, t_end=None, init_s=300, veh_pt_thres=5, noise_pt_thres=5,
                            prob_int=0.9, pixels=None):
        """
        This function subtracts the background in batch mode using MAP and FSM v1. Issues:
            - The cold tail effect. A detection cycle closes only when noise_pt_thres
              consecutive noise points are detected and those points are considered as part of the vehicle, which become
              the cold tail.
            - The long tail effect. The detection cycle won't properly close if every (noise_pt_thres-1) point, there is
              one veh point. The reason is the counter for the noise_pt is resetted each time it detects a vehicle point.
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

        _debug = False
        if _debug:
            _debug_pix = (0,0)
            _debug_mu = []

        # only normalize those period of data
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # save the batch normalized data
        frames = raw_data.index[ (raw_data.index >= t_start) & (raw_data.index <= t_end) ]
        _raw_data = raw_data.loc[frames,:]
        # set the data to be all 0
        veh_data = deepcopy(_raw_data)
        veh_data.values[:, 0:self.tot_pix] = np.nan

        if pixels is None:
            _row, _col = np.meshgrid( np.arange(0, self.pir_res[0]), np.arange(0, self.pir_res[1]) )
            pixels = zip(_row.flatten(), _col.flatten())

        # ------------------------------------------------------------------------
        # Initialize the background noise distribution.
        t_init_end = t_start + timedelta(seconds=init_s)
        _, _, noise_means, noise_stds = self._get_noise_distribution(_raw_data, t_start=t_start,
                                                                     t_end=t_init_end, p_outlier=0.01,
                                                                     stop_thres=(0.001, 0.0001), pixels=None)

        # ------------------------------------------------------------------------
        # the current noise distribution
        n_mu = noise_means.T.reshape(self.tot_pix)
        n_sigma = noise_stds.T.reshape(self.tot_pix)

        if _debug:
            print('initial mu and sigma: {0:.05f}, {1:.05f}'.format(n_mu[_debug_pix[1]*4+_debug_pix[0]],
                                                                    n_sigma[_debug_pix[1]*4+_debug_pix[0]]))

        # assuming the prior of the noise mean distribution is N(mu, sig), where mu noise_means and sig is noise_std*3
        prior_n_mu = deepcopy(n_mu)
        prior_n_sigma = deepcopy(n_sigma)*3.0

        sig_ratio = stats.norm.ppf(1-(1-prob_int)/2.0, 0, 1)

        # ------------------------------------------------------------------------
        # Now iterate through each frame
        _t_init_end = _raw_data.index[np.where(_raw_data.index>t_init_end)[0][0]]
        _t_end = _raw_data.index[np.where(_raw_data.index<=t_end)[0][-1]]

        # Save the reference, which can speed up the code
        norm_data = _raw_data.ix[_t_init_end:_t_end]
        num_frames = len(norm_data.index)

        # State definition:
        # 0: noise;
        # positive: number of positive consecutive vehicle pts;
        # negative: number of consecutive noise points from a vehicle state
        state = np.zeros(self.tot_pix)
        buf_is_veh = np.zeros(self.tot_pix)
        buf = {}
        for pix in range(0, self.tot_pix):
            buf[pix] = []

        i = 0
        for cur_t, sample_row in norm_data.iterrows():
            for pix in range(0, self.tot_pix):

                # check if current point is noise
                v = sample_row.values[pix]
                is_noise = (v>=n_mu[pix]-sig_ratio*n_sigma[pix]) & (v<=n_mu[pix]+sig_ratio*n_sigma[pix])

                if _debug and pix==_debug_pix[1]*4+_debug_pix[0]:
                    print('is noise: {0}'.format(is_noise))

                # for each pixel, run the state machine
                if state[pix] == 0:
                    if is_noise:
                        buf[pix].append(cur_t)
                    else:
                        # update noise distribution using buffer noise
                        (_n_mu, _n_sig), (_prior_mu, _prior_sig) = self._MAP_update(norm_data.ix[buf[pix], pix],
                                                                                    (n_mu[pix], n_sigma[pix]),
                                                 (prior_n_mu[pix], prior_n_sigma[pix]))
                        if _debug and pix== _debug_pix[1]*4+_debug_pix[0]:
                            _debug_mu.append([cur_t, _n_mu])
                            print('1st updating MAP:')
                            print('       last mu, sig: {0:.05f}, {1:.05f}'.format(n_mu[pix], n_sigma[pix]))
                            print('       prior mu, sig: {0:.05f}, {1:.05f}'.format(prior_n_mu[pix], prior_n_sigma[pix]))
                            print('       updated mu, sig: {0:.05f}, {1:.05f}'.format(_n_mu, _n_sig))
                            print('       updated prior mu, sig: {0:.05f}, {1:.05f}'.format(_prior_mu, _prior_sig))
                            print('       data: {0}'.format(norm_data.ix[buf[pix], pix].values))

                        # update the noise distribution
                        n_mu[pix] = _n_mu
                        n_sigma[pix] = _n_sig

                        # update the prior distribution of the background noise
                        prior_n_mu[pix] = _prior_mu
                        prior_n_sigma[pix] = _prior_sig

                        # clear buffer
                        buf[pix] = []
                        buf[pix].append(cur_t)
                        state[pix] = 1

                elif state[pix] > 0:
                    buf[pix].append(cur_t)
                    if is_noise:
                        state[pix] = -1
                    else:
                        state[pix] += 1
                        if state[pix] >= veh_pt_thres:
                            buf_is_veh[pix] = 1

                elif state[pix] < 0:
                    buf[pix].append(cur_t)
                    if is_noise:
                        state[pix] -= 1
                        if np.abs(state[pix]) >= noise_pt_thres:
                            # to dump the buffer
                            if buf_is_veh[pix] > 0:
                                # mark buffer as one vehicle point, normalize the data
                                if not _debug:
                                    veh_data.ix[buf[pix], pix] = np.abs((norm_data.ix[buf[pix], pix].values - n_mu[pix])/n_sigma[pix])
                                else:
                                    veh_data.ix[buf[pix], pix] = norm_data.ix[buf[pix], pix].values
                            else:
                                # update noise distribution using buffer noise
                                (_n_mu, _n_sig), (_prior_mu, _prior_sig) = self._MAP_update(norm_data.ix[buf[pix],pix].values,
                                                                                            (n_mu[pix], n_sigma[pix]),
                                                 (prior_n_mu[pix], prior_n_sigma[pix]))
                                 # update the noise distribution
                                n_mu[pix] = _n_mu
                                n_sigma[pix] = _n_sig

                                # update the prior distribution of the background noise
                                prior_n_mu[pix] = _prior_mu
                                prior_n_sigma[pix] = _prior_sig

                                if _debug and pix==_debug_pix[1]*4+_debug_pix[0]:
                                    _debug_mu.append([cur_t, _n_mu])
                                    print('2nd updated mu')
                                    print('       last mu, sig: {0:.05f}, {1:.05f}'.format(n_mu[pix], n_sigma[pix]))
                                    print('       prior mu, sig: {0:.05f}, {1:.05f}'.format(prior_n_mu[pix], prior_n_sigma[pix]))
                                    print('       updated mu, sig: {0:.05f}, {1:.05f}'.format(_n_mu, _n_sig))
                                    print('       updated prior mu, sig: {0:.05f}, {1:.05f}'.format(_prior_mu, _prior_sig))
                                    print('       data: {0}'.format(norm_data.ix[buf[pix], pix].values))

                            # reset the buffer and state
                            buf[pix] = []
                            state[pix] = 0
                            buf_is_veh[pix] = 0
                    else:
                        state[pix] = 1

            print_loop_status('Subtracting background for frame:', i, num_frames)
            i+=1

        if _debug:
            _debug_mu = np.array(_debug_mu)
            # plot a single pixel
            fig, ax = plt.subplots(figsize=(18,5))
            ax.plot(norm_data.index, norm_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))],
                    label='raw')
            ax.plot(veh_data.index, veh_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))],
                    label='veh', linewidth=3)
            ax.scatter(veh_data.index, veh_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))])
            ax.plot(_debug_mu[:,0], _debug_mu[:,1], label='_mu')
            ax.legend()
            plt.draw()

        return veh_data

    # @profile
    def subtract_background_v2(self, raw_data, t_start=None, t_end=None, init_s=300, veh_pt_thres=5, noise_pt_thres=5,
                            prob_int=0.9, pixels=None):
        """
        This function subtracts the background in batch mode using MAP and FSM v1
            - v2 fixes the cold tail problem by using two buffers: one for veh and one for noise
            - v2 fixed the long tail problem by not resetting the counter for noise. Note, in this way, a
              larger noise_pt_thres can prevent throwing the tail of vehicles away.
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

        _debug = False
        if _debug:
            _debug_pix = (0,0)
            _debug_mu = []

        # only normalize those period of data
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # save the batch normalized data
        frames = raw_data.index[ (raw_data.index >= t_start) & (raw_data.index <= t_end) ]
        _raw_data = raw_data.loc[frames,:]
        # set the data to be all 0
        veh_data = deepcopy(_raw_data)
        veh_data.values[:, 0:self.tot_pix] = np.nan

        if pixels is None:
            _row, _col = np.meshgrid( np.arange(0, self.pir_res[0]), np.arange(0, self.pir_res[1]) )
            pixels = zip(_row.flatten(), _col.flatten())

        # ------------------------------------------------------------------------
        # Initialize the background noise distribution.
        t_init_end = t_start + timedelta(seconds=init_s)
        _, _, noise_means, noise_stds = self._get_noise_distribution(_raw_data, t_start=t_start,
                                                                     t_end=t_init_end, p_outlier=0.01,
                                                                     stop_thres=(0.001, 0.0001), pixels=None)

        # ------------------------------------------------------------------------
        # the current noise distribution
        n_mu = noise_means.T.reshape(self.tot_pix)
        n_sigma = noise_stds.T.reshape(self.tot_pix)

        if _debug:
            print('initial mu and sigma: {0:.05f}, {1:.05f}'.format(n_mu[_debug_pix[1]*4+_debug_pix[0]],
                                                                    n_sigma[_debug_pix[1]*4+_debug_pix[0]]))

        # assuming the prior of the noise mean distribution is N(mu, sig), where mu noise_means and sig is noise_std*3
        prior_n_mu = deepcopy(n_mu)
        prior_n_sigma = deepcopy(n_sigma)*3.0

        sig_ratio = stats.norm.ppf(1-(1-prob_int)/2.0, 0, 1)

        # ------------------------------------------------------------------------
        # Now iterate through each frame
        _t_init_end = _raw_data.index[np.where(_raw_data.index>t_init_end)[0][0]]
        _t_end = _raw_data.index[np.where(_raw_data.index<=t_end)[0][-1]]

        # Save the reference, which can speed up the code
        norm_data = _raw_data.ix[_t_init_end:_t_end]
        num_frames = len(norm_data.index)

        # ------------------------------------------------------------------------
        # Finite State Machine
        # state: 0-background; 1:vehicle; -1:noise
        # buf_v = {}    :vehicle buffer
        # buf_n = {}    :noise buffer
        # v_counter     :the counter of vehicle points in this detection cycle
        # n_counter     :the counter of noise point in this detection cycle
        state = np.zeros(self.tot_pix)
        v_counter = np.zeros(self.tot_pix)
        n_counter = np.zeros(self.tot_pix)
        buf_v = {}
        buf_n = {}
        noise = {}
        for pix in range(0, self.tot_pix):
            buf_v[pix] = []
            buf_n[pix] = []
            noise[pix] = []

        i = 0
        for cur_t, sample_row in norm_data.iterrows():
            for pix in range(0, self.tot_pix):

                # check if current point is noise
                v = sample_row.values[pix]
                is_noise = (v>=n_mu[pix]-sig_ratio*n_sigma[pix]) & (v<=n_mu[pix]+sig_ratio*n_sigma[pix])

                if _debug and pix==_debug_pix[1]*4+_debug_pix[0]:
                    print('is noise: {0}'.format(is_noise))

                # for each pixel, run the state machine
                if state[pix] == 0:
                    if is_noise:
                        buf_n[pix].append(cur_t)
                    else:
                        # update noise distribution using buffer noise
                        noise[pix] = noise[pix] + buf_n[pix]
                        # update at most every 0.5 s = 32 data point
                        if len(noise[pix]) >= 1:
                            (_n_mu, _n_sig), (_prior_mu, _prior_sig) = self._MAP_update(norm_data.ix[noise[pix], pix].values,
                                                                                    (n_mu[pix], n_sigma[pix]),
                                                                                (prior_n_mu[pix], prior_n_sigma[pix]))
                            noise[pix] = []
                            if _debug and pix== _debug_pix[1]*4+_debug_pix[0]:
                                _debug_mu.append([cur_t, _n_mu])
                                print('1st updating MAP:')
                                print('       last mu, sig: {0:.05f}, {1:.05f}'.format(n_mu[pix], n_sigma[pix]))
                                print('       prior mu, sig: {0:.05f}, {1:.05f}'.format(prior_n_mu[pix], prior_n_sigma[pix]))
                                print('       updated mu, sig: {0:.05f}, {1:.05f}'.format(_n_mu, _n_sig))
                                print('       updated prior mu, sig: {0:.05f}, {1:.05f}'.format(_prior_mu, _prior_sig))
                                print('       data: {0}'.format(norm_data.ix[buf_n[pix], pix].values))

                            # update the noise distribution
                            n_mu[pix] = _n_mu
                            n_sigma[pix] = _n_sig

                            # update the prior distribution of the background noise
                            prior_n_mu[pix] = _prior_mu
                            prior_n_sigma[pix] = _prior_sig

                        # clear noise buffer
                        buf_n[pix] = []
                        # append vehicle point to vehicle buffer
                        buf_v[pix].append(cur_t)
                        v_counter[pix] = 1
                        state[pix] = 1

                elif state[pix] > 0:
                    # at detection cycle at vehicle
                    if is_noise:
                        state[pix] = -1
                        buf_n[pix].append(cur_t)
                        n_counter[pix] += 1
                    else:
                        buf_v[pix].append(cur_t)
                        v_counter[pix] += 1

                elif state[pix] < 0:
                    # at detection cycle at noise
                    if is_noise:
                        buf_n[pix].append(cur_t)
                        n_counter[pix] += 1

                        if n_counter[pix] >= noise_pt_thres:
                            # closing a detection cycle
                            if v_counter[pix] > veh_pt_thres:
                                # vehicle detected
                                if not _debug:
                                    # add normalized vehicle data
                                    veh_data.ix[buf_v[pix], pix] = np.abs((norm_data.ix[buf_v[pix], pix].values -
                                                                           n_mu[pix])/n_sigma[pix])
                                else:
                                    # unnormalized for easier comarison
                                    veh_data.ix[buf_v[pix], pix] = norm_data.ix[buf_v[pix], pix].values
                            else:
                                # update noise distribution using both the buf_n and buf_v since buf_v is just noise
                                # update noise distribution using buffer noise
                                noise[pix] = noise[pix] + buf_n[pix] + buf_v[pix]
                                # update at most every 0.5 s = 32 data point
                                if len(noise[pix]) >= 1:
                                    (_n_mu, _n_sig), (_prior_mu, _prior_sig) = self._MAP_update(norm_data.ix[noise[pix], pix].values,
                                                                                            (n_mu[pix], n_sigma[pix]),
                                                                                        (prior_n_mu[pix], prior_n_sigma[pix]))
                                    # clear noise data cache
                                    noise[pix] = []

                                    if _debug and pix==_debug_pix[1]*4+_debug_pix[0]:
                                        _debug_mu.append([cur_t, _n_mu])
                                        print('2nd updated mu')
                                        print('       last mu, sig: {0:.05f}, {1:.05f}'.format(n_mu[pix], n_sigma[pix]))
                                        print('       prior mu, sig: {0:.05f}, {1:.05f}'.format(prior_n_mu[pix], prior_n_sigma[pix]))
                                        print('       updated mu, sig: {0:.05f}, {1:.05f}'.format(_n_mu, _n_sig))
                                        print('       updated prior mu, sig: {0:.05f}, {1:.05f}'.format(_prior_mu, _prior_sig))
                                        print('       data: {0}'.format(norm_data.ix[buf_n[pix]+buf_v[pix], pix].values))

                                    # update the noise distribution
                                    n_mu[pix] = _n_mu
                                    n_sigma[pix] = _n_sig

                                    # update the prior distribution of the background noise
                                    prior_n_mu[pix] = _prior_mu
                                    prior_n_sigma[pix] = _prior_sig

                            # clear the buffers
                            buf_v[pix] = []
                            buf_n[pix] = []
                            v_counter[pix] = 0
                            n_counter[pix] = 0
                            state[pix] = 0
                    else:
                        buf_v[pix].append(cur_t)
                        v_counter[pix] += 1
                        buf_n[pix] = [] # clear noise buffer
                        state[pix] = 1

            print_loop_status('Subtracting background for frame:', i, num_frames)
            i+=1

        if _debug:
            _debug_mu = np.array(_debug_mu)
            # plot a single pixel
            fig, ax = plt.subplots(figsize=(18,5))
            ax.plot(norm_data.index, norm_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))],
                    label='raw')
            ax.plot(veh_data.index, veh_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))],
                    label='veh', linewidth=3)
            ax.scatter(veh_data.index, veh_data['pir_{0}x{1}'.format(int(_debug_pix[0]), int(_debug_pix[1]))])
            ax.plot(_debug_mu[:,0], _debug_mu[:,1], label='_mu')
            ax.legend()
            plt.draw()

        return veh_data

    # @profile
    def _MAP_update(self, data, paras, prior):
        """
        This function updates the noise distribution and the prior of the noise mean distribution
        :param data: the noise data point
        :param paras: (mu, sigma) the current noise distribution
        :param prior: (prior_mu, prior_sigma) the prior distribution of mu
        :return:
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


    def _get_noise_distribution(self, raw_data, t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.001,0.0001),
                                pixels=None):
        """
        This function computes the mean and std of noise distribution (normal) by iteratively fitting a normal
            distribution and throwing away points outsize of (1-p_outlier) confidence interval
        :param raw_data: the raw data, (as reference)
        :param t_start: datetime, start time of the period for getting the mean and std
        :param t_end: datetime
        :param p_outlier: the confidence interval is (1-p_outlier),
        :param stop_thres: (d_mean, d_std), stop iteration if the change from last distribution < stop_thres
        :param pixels: list of tuples, which pixel to compute
        :return:
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

        dw = timedelta(seconds=window_s)
        dt = timedelta(seconds=step_s)
        len_data = len(data)

        timestamps = []
        mus = []
        sigmas = []

        last_update_t = data.index[0]
        for i, cur_t in enumerate(data.index):
            if cur_t - last_update_t >= dt:
                _, _, _mu, _sigma = self._get_noise_distribution(data, t_start=cur_t-dw, t_end=cur_t, p_outlier=p_outlier,
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
        ax = self.plot_time_series_for_pixel(data, t_start=None, t_end=None, pixels=[pixel])
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


# ==================================================================================================================
# Video processing class
# ==================================================================================================================
class VideoData:

    def __init__(self):
        pass

    @staticmethod
    def crop_video(input_video, output_video, rotation=None, x_coord=None, y_coord=None, frame_lim=None):

        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # print out information
        print('Loaded video {0}:'.format(input_video))
        print('    Current timestamp: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
        print('    Current index: {0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        print('    Resolution: {0} x {1}'.format(res[0], res[1]))
        print('    FPS: {0}'.format(fps))
        print('    Frame count: {0}'.format(total_frames))

        # --------------------------------------------------------------
        # set the crop locations
        if x_coord is None: x_coord = (0, res[0])
        if y_coord is None: y_coord = (0, res[1])
        if frame_lim is None: frame_lim = (0, int(total_frames))

        # --------------------------------------------------------------
        # define the output video
        # fourcc = cv2.cv.CV_FOURCC('m','p','4','v')    # for opencv 2
        fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
        out = cv2.VideoWriter(output_video, fourcc, fps,
                              (x_coord[1]-x_coord[0],y_coord[1]-y_coord[0]))

        # --------------------------------------------------------------
        # crop video
        for i in range(frame_lim[0], frame_lim[1]):

            with suppress_stdout():
                ret, frame = cap.read()
                if ret is True:

                    if rotation is not None:
                        frame = imutils.rotate(frame, rotation)

                    out.write(frame[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :])
                else:
                    raise Exception('fail to read frame')

            print_loop_status('Status: saving frame: ', i, int(frame_lim[1]))

        print('\nFinished saving cropped video.')
        cap.release()
        out.release()

    # TODO: untested
    @staticmethod
    def trim_video(input_video, output_video, offset=None, trim_period=None):
        """
        This function trims the video
        :param input_video: input video file
        :param output_video: output video file
        :param offset: timedelta, the offset of the video
        :param trim_period: tuple, the period to be trimed.
        :return:
        """

        # --------------------------------------------------------------
        # if not specified, assume no time shift
        if offset is None: offset = timedelta(seconds=0)

        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # get the timestamp
        print('Loaded video {0}:'.format(input_video))
        print('    Resolution: {0} x {1}'.format(res[0], res[1]))
        print('    FPS: {0}'.format(fps))
        print('    Frame count: {0}'.format(total_frames))
        print('    Current timestamp: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
        print('    Current index: {0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # --------------------------------------------------------------
        # compute the index of frames to trim
        index_start = int( (trim_period[0]-offset).total_seconds()*fps )
        index_end = int( (trim_period[1]-offset).total_seconds()*fps )
        num_frames = index_end-index_start

        print('Trimming video...')
        # --------------------------------------------------------------
        # set the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_start)
        # create writerCV
        fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
        out = cv2.VideoWriter(output_video, fourcc, fps, res)

        for i in range(0, num_frames):
            ret, frame = cap.read()

            # time_str = time2str( trim_period[0] + timedelta(seconds=i/fps) )
            # #
            # cv2.putText(frame, time_str, (150, 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255))

            if ret is True:
                out.write(frame)
                sys.stdout.write('\r')
                sys.stdout.write('Status: filtering step {0}/{1}'.format(i, num_frames))
                sys.stdout.flush()
            else:
                raise Exception('fail to read frame')

        cap.release()
        out.release()
        # cv2.destroyAllWindows()

    @staticmethod
    def play_video(input_video, rotation=None, x_coord=None, y_coord=None):
        """
        This function plays the video. Press q to quit
        :param input_video: input video file
        :return:
        """

        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # print out information
        print('Loaded video {0}:'.format(input_video))
        print('    Resolution: {0} x {1}'.format(res[0], res[1]))
        print('    FPS: {0}'.format(fps))
        print('    Frame count: {0}'.format(total_frames))
        print('    Current timestamp: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
        print('    Current index: {0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # --------------------------------------------------------------
        if x_coord is None: x_coord = (0, res[0])
        if y_coord is None: y_coord = (0, res[1])
        for i in range(0, int(total_frames)):
            ret, frame = cap.read()

            if rotation is not None:
                frame = imutils.rotate(frame, rotation)

            # play in gray
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # # draw a line to find out the lane bound
            # cv2.line(gray, (0,0), res, (255,255,255), 5)
            # cv2.imshow('{0}'.format(input_video),gray)

            # play in color
            cv2.line(frame, (x_coord[0], y_coord[0]), (x_coord[1], y_coord[0]), (0,255,0), 1)
            cv2.line(frame, (x_coord[0], y_coord[1]), (x_coord[1], y_coord[1]), (0,255,0), 1)
            cv2.imshow('{0}'.format(input_video), frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break

    @staticmethod
    def generate_heatmap(input_video, save_npy=None):

        # ----------------------------------------------------------------------------------------
        # Load video
        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # print out information
        print('Loaded video {0}:'.format(input_video))
        print('    Resolution: {0} x {1}'.format(res[0], res[1]))
        print('    FPS: {0}'.format(fps))
        print('    Frame count: {0}'.format(total_frames))
        print('    Current timestamp: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
        print('    Current index: {0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # ----------------------------------------------------------------------------------------
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        # ----------------------------------------------------------------------------------------
        heatmap = []
        for i in range(0, int(total_frames)):
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)

            # ------------------------------------------------------------------------------
            # plot if necessary
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # stack_img = np.vstack([gray, fgmask])
            # cv2.imshow('subtracted background', stack_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # ------------------------------------------------------------------------------
            # sum up each column and normalize to [0,1]
            heatmap.append(np.sum(fgmask, 0)/(255.0*res[1]))
            print_loop_status('Status: vectorizing ', i, total_frames)

        heatmap = np.asarray(heatmap).T

        if save_npy is not None:
            np.save(save_npy, heatmap)

        cap.release()

        return heatmap

    @staticmethod
    def plot_heatmap(heatmap, figsize=(18,8), plot=False, save_img=None, title=''):

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('auto')
        im = ax.imshow(heatmap, cmap=plt.get_cmap('jet'), interpolation='nearest', aspect='auto',
                        vmin=0, vmax=1)
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_title('{0}'.format(title), fontsize=18)
        x_ticks = ax.get_xticks().tolist()
        # ax.set_xticklabels( np.round(np.array(x_ticks)/60.0).astype(int) )
        ax.set_xlabel('Time (seconds)', fontsize=16)
        ax.set_ylabel('Frame pixels', fontsize=16)

        if save_img is not None:
            plt.savefig('{0}'.format(save_img), bbox_inches='tight')

        if plot is True:
            plt.draw()
        else:
            plt.clf()
            plt.close()

        return heatmap

    @staticmethod
    def generate_1d_signal(input_video, rotation=None, x_coord=None, y_coord=None):
        """
        This function plays the video. Press q to quit
        :param input_video: input video file
        :return:
        """

        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # print out information
        print('Loaded video {0}:'.format(input_video))
        print('    Resolution: {0} x {1}'.format(res[0], res[1]))
        print('    FPS: {0}'.format(fps))
        print('    Frame count: {0}'.format(total_frames))
        print('    Current timestamp: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
        print('    Current index: {0}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        if x_coord is None: x_coord = (0, res[0])
        if y_coord is None: y_coord = (0, res[1])

        # --------------------------------------------------------------
        # define the output video
        # fourcc = cv2.cv.CV_FOURCC('m','p','4','v')    # for opencv 2
        fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
        output_video = input_video.replace('.mp4', '_cropped.mp4')
        out = cv2.VideoWriter(output_video, fourcc, fps,
                              (x_coord[1]-x_coord[0],y_coord[1]-y_coord[0]))

        # --------------------------------------------------------------
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        signal = []
        for i in range(0, int(total_frames)):
            ret, frame = cap.read()

            if rotation is not None:
                frame = imutils.rotate(frame, rotation)

            out.write(frame[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :])

            # subtract background
            fgmask = fgbg.apply(frame)

            # play in gray
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # # draw a line to find out the lane bound
            # cv2.line(gray, (0,0), res, (255,255,255), 5)
            # cv2.imshow('{0}'.format(input_video),gray)

            # summarize the energy within x_coord and y_coord and normalize to [0,1]
            energy = np.sum( fgmask[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]] )/\
                     (255*(x_coord[1]-x_coord[0])*(y_coord[1]-y_coord[0]))
            signal.append(energy)

            print_loop_status('Status: processing frame  ', i, total_frames)

        cap.release()
        out.release()

        return signal
