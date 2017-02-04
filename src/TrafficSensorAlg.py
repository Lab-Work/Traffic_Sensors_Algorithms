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
from datetime import datetime
from datetime import timedelta
import numpy as np
from scipy import stats
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
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

    def __init__(self, data_source=None, pir_res=(4,32), save_dir='', plot=False):
        """
        Initialize the function with the data source and output options
        :return:
        """
        pass


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


# ==================================================================================================================
# vehicle detection class
# ==================================================================================================================
class VehDet:
    def __init__(self):
        pass

    def detect_veh(self, buf, window=0.5):

        pir_data = buf.ix[:, [i for i in buf.columns if 'pir' in i]]
        ultra_data = buf.ix[:, 'ultra']

        # ----------------------------------------------------
        # Use a 1 s hanning window (64 points) to smooth the entropy over time.
        pir_entropy = pir_data.mean(1)


        # plot the data
        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(pir_entropy.index, pir_entropy.values, linewidth=1, label='pir')
        ax.plot(ultra_data.index, ultra_data.values, linewidth=1, label='ultra')
        ax.legend()
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


    def subtract_background(self, raw_data, t_start=None, t_end=None, init_s=300, veh_pt_thres=5, noise_pt_thres=5,
                            prob_int=0.9, pixels=None):

        # only normalize those period of data
        if t_start is None: t_start = raw_data.index[0]
        if t_end is None: t_end = raw_data.index[-1]

        # save the batch normalized data
        frames = raw_data.index[ (raw_data.index >= t_start) & (raw_data.index <= t_end) ]
        norm_data = raw_data.loc[frames,:]
        # set the data to be all 0
        veh_data = deepcopy(norm_data)
        veh_data.values[:, 0:self.tot_pix] = 0.0

        if pixels is None:
            _row, _col = np.meshgrid( np.arange(0, self.pir_res[0]), np.arange(0, self.pir_res[1]) )
            pixels = zip(_row.flatten(), _col.flatten())

        # ------------------------------------------------------------------------
        # Initialize the background noise distribution.
        t_init_end = t_start + timedelta(seconds=init_s)
        _, _, noise_means, noise_stds = self._get_noise_distribution(norm_data, t_start=t_start,
                                                                     t_end=t_init_end, p_outlier=0.01,
                                                                     stop_thres=(0.1, 0.01), pixels=None)

        # ------------------------------------------------------------------------
        # the current noise distribution
        n_mu = noise_means.T.reshape(self.tot_pix)
        n_sigma = noise_stds.T.reshape(self.tot_pix)

        # assuming the prior of the noise mean distribution is N(mu, sig), where mu noise_means and sig is noise_std*3
        prior_n_mu = deepcopy(n_mu)
        prior_n_sigma = deepcopy(n_sigma)*3.0

        # ------------------------------------------------------------------------
        # Now iterate through each frame
        _t_init_end = raw_data.index[np.where(raw_data.index>t_init_end)[0][0]]
        _t_end = raw_data.index[np.where(raw_data.index<=t_end)[0][-1]]
        idxs = norm_data.ix[_t_init_end:_t_end].index
        num_frames = len(idxs)

        # State definition:
        # 0: noise;
        # positive: number of positive consecutive vehicle pts;
        # negative: number of consecutive noise points from a vehicle state
        state = np.zeros(self.tot_pix)
        buf_is_veh = np.zeros(self.tot_pix)
        buf = {}
        for pix in range(0, self.tot_pix):
            buf[pix] = []

        for i, cur_t in enumerate(idxs):

            # check if current point is noise
            is_noise = in_conf_intrvl(norm_data.ix[cur_t, 0:self.tot_pix], prob_int, n_mu, n_sigma)

            for pix in range(0, self.tot_pix):
                # for each pixel, run the state machine
                if state[pix] == 0:
                    if is_noise[pix]:
                        # update noise distribution using buffer noise
                        self._MAP_update([norm_data.ix[cur_t, pix]], (n_mu[pix], n_sigma[pix]),
                                                 (prior_n_mu[pix], prior_n_sigma[pix]))
                    else:
                        buf[pix].append(cur_t)
                        state[pix] = 1

                elif state[pix] > 0:
                    buf[pix].append(cur_t)
                    if is_noise[pix]:
                        state[pix] = -1
                    else:
                        state[pix] += 1
                        if state[pix] >= veh_pt_thres:
                            buf_is_veh[pix] = 1

                elif state[pix] < 0:
                    buf[pix].append(cur_t)
                    if is_noise[pix]:
                        state[pix] -= 1
                        if np.abs(state[pix]) >= noise_pt_thres:
                            # to dump the buffer
                            if buf_is_veh[pix] > 0:
                                # mark buffer as one vehicle point
                                veh_data.ix[buf[pix], pix] = norm_data.ix[buf[pix], pix].values
                            else:
                                # update noise distribution using buffer noise
                                self._MAP_update(norm_data.ix[buf[pix],pix].values, (n_mu[pix], n_sigma[pix]),
                                                 (prior_n_mu[pix], prior_n_sigma[pix]))

                            # reset the buffer and state
                            buf[pix] = []
                            state[pix] = 0
                            buf_is_veh[pix] = 0
                    else:
                        state[pix] = 1

            print_loop_status('Processing frame: ', i, num_frames)

        return veh_data


    def _MAP_update(self, data, paras, prior):
        mu, sig = paras
        prior_mu, prior_sig = prior
        if type(data) is int or type(data) is float:
            len_data = 1
        else:
            len_data = len(data)

        post_var = np.sqrt(1.0/(1.0/prior_sig**2 + len_data/sig**2))
        post_mu = post_var*(np.sum(data)/sig**2 + prior_mu/prior_sig**2)

        return post_mu, np.sqrt( sig**2 +  post_var)


    def _get_noise_distribution(self, raw_data, t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.1,0.01),
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
