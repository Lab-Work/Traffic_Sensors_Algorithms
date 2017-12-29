__author__ = 'Yanning Li (yli171@illinois.edu), Fangyu Wu (fwu10@illinois.edu)'

"""
The classes interface with the data files. It reads the data, conducts statistic analysis of the data, and visualizes
the data.

The data input should be one single csv file with the format specified in the Data_Collection_Manual.pdf.
This class can handle different format of the data, e.g. PIR data with different number of pixels, or data missing the
ultrasonic sensor measurement.

The structure of each class:
# - Visualization
# - Statistic analysis

"""


import numpy as np
import matplotlib
from scipy import stats
matplotlib.use('TkAgg')
import bisect
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import matplotlib.patches as patches
from datetime import datetime
from datetime import timedelta
from os.path import exists
from collections import OrderedDict
import sys
import time
import glob
from sklearn import mixture
import cv2



class TrafficData:
    """
    This class is used for data preprocessing, including visualize the data, and exploring potential algorithms.
    """

    def __init__(self):

        # self.PIR[data_key] = {'time': [], 'data':[]}
        self.PIR = {}

        # ultrasonic sensor data
        # USON['time'] = narray; USON['data'] = narray
        self.USON = {}

        # IMU data
        # IMU = {'time':narray, 'accel':3xn array, 'mag':3xn array, 'gyro':3xn array}
        self.IMU = {}

        # LABEL data
        # LABEL = {'time':narray, 'count':narray, 'speed':narray}
        self.LABEL = {}

    def load_npy_data(self, file_name_str=None, dataset=None, data_key='raw_data'):
        """
        This function loads the data npy file. each entry is [timestamp(datetime), 4x16 pir array, ptat, cp]
        :param file_name_str: a string of the file name,
        :param dataset: the name of the dataset file
        :param data_key: the data key, any name: 'raw' 'temperature'...
        :return:
        """
        if file_name_str is not None and exists(file_name_str):
            data = np.load(file_name_str)

            if dataset is None:
                # default dataset name is the file name
                dataset = file_name_str.strip().split('/')[-1].replace('.npy', '')

            self.dataset = dataset

            # check if read other types of data previously
            if data_key not in self.PIR.keys():
                self.PIR[data_key] = {}

            if 'time' not in self.PIR[data_key].keys():
                # get the time stamp
                self.PIR[data_key]['time'] = []
                for entry in data:
                    self.PIR[data_key]['time'].append(entry[0])

                self.PIR[data_key]['time'] = np.asarray(self.PIR[data_key]['time'])

            if 'data' not in self.PIR[data_key].keys():
                # get the data
                self.PIR[data_key]['data'] = []
                for entry in data:
                    self.PIR[data_key]['data'].append( entry[1] )
                self.PIR[data_key]['data'] = np.asarray(self.PIR[data_key]['data'])

        else:
            raise Exception('Dataset file not exists: {0}'.format(file_name_str))

    # moved
    def load_txt_data(self, file_name_str=None, dataset=None, data_key='raw_data'):
        """
        This function loads the txt data file
        :param file_name_str:
        :param dataset:
        :param data_key:
        :return:
        """
        if file_name_str is not None and exists(file_name_str):

            if dataset is None:
                # default dataset name is the file name
                dataset = file_name_str.strip().split('/')[-1].replace('.npy', '')

            self.dataset = dataset

            if data_key not in self.PIR.keys():
                    self.PIR[data_key] = {}

            if 'time' not in self.PIR[data_key].keys():
                # get the time stamp
                self.PIR[data_key]['time'] = []

            if 'data' not in self.PIR[data_key].keys():
                # get the data
                self.PIR[data_key]['data'] = []

            with open(file_name_str, 'r') as f:
                for line in f:
                    item = line.strip().split('|')

                    self.PIR[data_key]['time'].append( self.video_string_to_time(item[0]) )

                    # convert string to list
                    val = [float(i) for i in item[1].split(',')]

                    self.PIR[data_key]['data'].append( np.array(val).reshape((4,32)) )    # row by row
                    # self.PIR[data_key]['data'].append( np.array(val).reshape((32,4)).T )    # col by col

            self.PIR[data_key]['time'] = np.asarray(self.PIR[data_key]['time'])
            self.PIR[data_key]['data'] = np.\
                asarray(self.PIR[data_key]['data'])

    # moved
    def get_data_periods(self, dir, update=True):
        """
        This function returns the periods for all data collection experiments saved in a directory
        :param dir: the directory
        :return: save in file and return the periods
        """
        periods = OrderedDict()
        f_periods = dir.replace('*.npy', '')+'dataset_periods.cfg'

        if update is True:
            files = glob.glob(dir)

            for f in files:
                # get the sensor config
                sensor_id = f.split('/')[-1].replace('.npy', '')

                d = np.load(f)
                t_start = d[0][0]
                t_end = d[-1][0]

                periods[sensor_id] = (t_start, t_end)

            # save in a file
            with open(f_periods, 'w') as f:
                for key in periods:
                    f.write('{0},{1},{2}\n'.format(key, self.time_to_string(periods[key][0]),
                                                   self.time_to_string(periods[key][1]) ))
            print('Updated dataset_periods.cfg.')
        else:

            # load previously extracted file if exists
            if exists(f_periods):
                print('Loading dataset_periods.cfg ...')
                with open(f_periods,'r') as fi:
                    for line in fi:
                        items = line.strip().split(',')
                        periods[items[0]] = ( self.string_to_time(items[1]), self.string_to_time(items[2]) )
                print('Loaded dataset_periods.cfg.')

            else:
                raise Exception('Previous f_periods not exit')

        return periods

    # moved
    def get_txt_data_periods(self, dir, update=True):
        """
        This function returns the periods for all data collection experiments saved in a directory
        :param dir: the directory
        :return: save in file and return the periods
        """
        periods = OrderedDict()
        f_periods = dir.replace('*.txt', '')+'dataset_periods.cfg'

        if update is True:
            files = glob.glob(dir)

            for f in files:
                # get the sensor config
                sensor_id = f.split('/')[-1].replace('.txt', '')

                with open(f,'r') as fi:
                    first_line = fi.readline()
                    # print 'first line:'
                    # print first_line
                    t_start = self.video_string_to_time( first_line.strip().split('|')[0] )
                    print('t_start: {0}'.format(t_start))

                    for line in fi:
                        pass

                    # last line
                    # print 'last' \
                    #       ' line:'
                    # print line
                    t_end = self.video_string_to_time( line.strip().split('|')[0] )

                periods[sensor_id] = (t_start, t_end)

            # save in a file
            with open(f_periods, 'w') as f:
                for key in periods:
                    f.write('{0},{1},{2}\n'.format(key, self.time_to_string(periods[key][0]),
                                                   self.time_to_string(periods[key][1]) ))
            print('Updated dataset_periods.cfg.')
        else:

            # load previously extracted file if exists
            if exists(f_periods):
                print('Loading dataset_periods.txt ...')
                with open(f_periods,'r') as fi:
                    for line in fi:
                        items = line.strip().split(',')
                        periods[items[0]] = ( self.string_to_time(items[1]), self.string_to_time(items[2]) )
                print('Loaded dataset_periods.txt.')

            else:
                raise Exception('Previous f_periods not exit')

        return periods

    # moved
    def get_noise_distribution(self, data_key=None, t_start=None, t_end=None,
                               p_outlier=0.01, stop_thres=(0.1, 0.01), pixels=None, suppress_output=False):
        """
        This function returns the noise distribution. It iteratively throw away the data points that has a probability
        below p_outlier and refit a gaussian distribution. It stops when the change fo the mean and std are within the
        specified stop criteria.
        :param data_key: the key of the dataset. Processed data will be saved in the same dataset but with differnt keys
        :param t_start: datetime type
        :param t_end: datatime type
        :param p_outlier: [0,1], a point is considered as outlier if it is below p_outlier
        :param stop_thres: (delta_mu, delta_sigma) degrees in temperature
        :param pixels: a list of tuples row [0,3], col [0,15] or [0,31]
        :return: mu, sigma, noise_mu, noise_sigma
        """

        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        mean = np.empty((num_rows, num_cols))
        mean.fill(np.nan)
        std = np.empty((num_rows, num_cols))
        std.fill(np.nan)

        noise_mean = np.empty((num_rows, num_cols))
        noise_mean.fill(np.nan)
        noise_std = np.empty((num_rows, num_cols))
        noise_std.fill(np.nan)

        if pixels is None:
            # compute all pixels
            _row, _col = np.meshgrid( np.arange(0, num_rows), np.arange(0, num_cols) )
            pixels = zip(_row.flatten(), _col.flatten())

        for row, col in pixels:

            time_series = self.PIR[data_key]['data'][index_start:index_end, row, col]

            # save the initial estimate
            mean[row, col] = np.nanmean( time_series )
            std[row, col] = np.nanstd( time_series )

            _pre_mean = mean[row, col]
            _pre_std = std[row, col]

            # =======================================================
            # This segment is the incorrect way of computing outliers
            # compute the mean and std of the noise
            # p = mlab.normpdf(time_series, _pre_mean, _pre_std)
            # _mean = np.nanmean( time_series[ p>=p_outlier] )
            # _std = np.nanstd( time_series[ p>=p_outlier] )
            # =======================================================
            # Pr( x \in [-v_thres, v_thres] ) = 1-p_outlier
            v_thres_u = stats.norm.ppf(1-p_outlier/2.0, _pre_mean, _pre_std)
            v_thres_l = _pre_mean - (v_thres_u-_pre_mean)
            _idx = (v_thres_l <= time_series) & ( time_series <= v_thres_u)
            _mean = np.nanmean( time_series[_idx] )
            _std = np.nanstd( time_series[_idx] )

            while np.abs(_mean - _pre_mean) > stop_thres[0] or np.abs(_std - _pre_std) > stop_thres[1]:
                if suppress_output is False:
                    print('updating noise distribution for pixel {0}'.format((row, col)))
                _pre_mean = _mean
                _pre_std = _std

                # =======================================================
                # This segment is the incorrect way of computing outliers
                # p = mlab.normpdf(time_series, _pre_mean, _pre_std)
                # _mean = np.nanmean( time_series[ p>=p_outlier] )
                # _std = np.nanstd( time_series[ p>=p_outlier] )
                # =======================================================
                # Pr( x \in [-v_thres, v_thres] ) = 1-p_outlier
                v_thres_u = stats.norm.ppf(1-p_outlier/2.0, _pre_mean, _pre_std)
                v_thres_l = _pre_mean - (v_thres_u-_pre_mean)
                _idx = (-v_thres_l <= time_series) & ( time_series <= v_thres_u)
                _mean = np.nanmean( time_series[_idx] )
                _std = np.nanstd( time_series[_idx] )

            # save the final noise distribution
            noise_mean[row, col] = _mean
            noise_std[row, col] = _std

        return mean, std, noise_mean, noise_std

    # moved
    def normalize_data(self, data_key=None, norm_data_key=None, t_start=None, t_end=None,
                       p_outlier=0.01, stop_thres=(0.1, 0.01), window_s=5, step_s=1, fps=64):
        """
        This function normalizes the data in the period [t_start, t_end], and save the normalized data in class
            - The background is normalized within each window_s, which assumes background temperature changes.
              If window_s is None, then normalize within entire duration.
            - The background noise is proven to be normal. Hence it is separated by iteratively throw out outliers that
              have probability less than p_outliers.
            - The iteration stops when mean and std of new fitted normal distribution parameter change is less than
              stop_thres (celcius)
        :param dataset: the dataset identifier
        :param data_key: the key of the data to be normalized
        :param norm_data_key: the key of the normalized data
        :param t_start: datetime type
        :param t_end: datetime type
        :param p_outlier: the probability threshold for determining outliers.
        :param stop_thres: (d_mean, d_std) in Celcius. Stop iteration if new distribution parameter change is smaller.
        :param window_s: second. The window for normalization. If None, normalize in the entire period.
        :param step_s: second. The incremental step for moving the window.
        :param fps: the average framerate of the data.
        :return:
        """

        if norm_data_key is None:
            norm_data_key = 'norm_' + data_key

        if t_start is None:
            t_start = self.PIR[data_key]['time'][0]

        if t_end is None:
            t_end = self.PIR[data_key]['time'][-1]

        frame_start, frame_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)
        num_frames = frame_end - frame_start

        # save the normalized data to those properties
        self.PIR[norm_data_key] = {}
        self.PIR[norm_data_key]['time'] = self.PIR[data_key]['time'][frame_start:frame_end]
        self.PIR[norm_data_key]['data'] = np.zeros((num_frames, 4, 32))

        if window_s is None:

            _, _, noise_mu, noise_sigma = self.get_noise_distribution(data_key=data_key, t_start=t_start, t_end=t_end,
                                        p_outlier=p_outlier, stop_thres=stop_thres, suppress_output=True)


            self.PIR[norm_data_key]['data'] = \
                (self.PIR[data_key]['data'][frame_start:frame_end] - noise_mu)/noise_sigma

        else:
            dt_hw = timedelta(seconds=window_s/2.0)

            # compute the number of frames that corresponds to the duration in seconds
            d_frame = int(fps*step_s)

            print('Normalizing data: frame {0} ~ {1}'.format(frame_start, frame_end))

            # frame counter
            frame_counter = d_frame
            noise_mu, noise_sigma = None, None
            for frame in range(frame_start, frame_end):

                # reset counter and update mean and std
                if frame_counter == d_frame:
                    frame_counter = 0

                    # update the distribution
                    if self.PIR[data_key]['time'][frame]-dt_hw < t_start:
                        _t_start = t_start
                    else:
                        _t_start = self.PIR[data_key]['time'][frame]-dt_hw

                    if self.PIR[data_key]['time'][frame]+dt_hw > t_end:
                        _t_end = t_end
                    else:
                        _t_end = self.PIR[data_key]['time'][frame]+dt_hw

                    _, _, noise_mu, noise_sigma = self.get_noise_distribution(data_key=data_key,
                                                                    t_start=_t_start, t_end=_t_end,
                                                                    p_outlier=p_outlier, stop_thres=stop_thres,
                                                                    suppress_output=True)

                # save normalized frame
                self.PIR[norm_data_key]['data'][frame-frame_start] = \
                    (self.PIR[data_key]['data'][frame] - noise_mu)/noise_sigma
                frame_counter += 1

                # print status
                self.print_loop_status('Normalizing frames: ', frame-frame_start, num_frames)



    # @ deprecated
    def train_gmm(self, t_start_str=None, t_end_str=None, pixel_list=list(), data_type=None):
        """
        This function fits a gmm model to the nosie and vehicle for each pixel
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param pixel_list: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :param data_type: the options for the data, 'raw_data', 'temp_data',...
        :return:
        """
        # For each PIR configuration
        for pir_mxn_list in pixel_list:

            sensor_id = pir_mxn_list[0]
            pixels = pir_mxn_list[1]
            if sensor_id not in self.PIR.keys():
                raise Exception('Incorrect sensor id')

            # find the data in the time interval
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start_str=t_start_str, t_end_str=t_end_str)
            for pixel in pixels:

                time_series = self.PIR[sensor_id][data_type][index_start:index_end,
                              pixel[0], pixel[1]]
                time_series = time_series.reshape((len(time_series), 1))

                # train a gmm classifier
                g = mixture.GMM(n_components=1, n_iter=100000)
                g.fit(time_series)
                mu = g.means_.squeeze()
                cov = g.means_.squeeze()


                # the histogram of the data
                num_bins = 200
                fig = plt.figure(figsize=(8,5), dpi=100)
                n, bins, patches = plt.hist(time_series, num_bins,
                                            normed=1, facecolor='green', alpha=0.75)

                # Need tuple for indexing. Make sure is tuple
                if not isinstance(pixel, tuple):
                    pixel = tuple(pixel)

                # add a 'best fit' line
                norm_fit_line = mlab.normpdf(bins, mu, np.sqrt(cov))
                l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1)
                # norm_fit_line = mlab.normpdf(bins, mu[1], np.sqrt(cov[1]))
                # l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1)

                plt.xlabel('Temperature ($^{\circ}C$)')
                plt.ylabel('Probability')
                # plt.title(r'{0} pixel {1}; $\mu, \sigma$=({2:.2f}, {3:.2f}), ({4:.2f}, {5:.2f})'.format(sensor_id, pixel,
                #                                                                                  mu[0], np.sqrt(cov[0]),
                #                                                                                  mu[1], np.sqrt(cov[1])))
                plt.grid(True)

        plt.draw()

    # deprecated
    def check_duplicates(self, data_key=None, t_start=None, t_end=None):
        """
        This function checks if there is duplicated data
        :param dataset:
        :param data_key:
        :param t_start:
        :param t_end:
        :return:
        """
        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        # find the duplicated frames
        num_duplicated_frames = 0
        for i in range(index_start, index_end-1):
            if (self.PIR[data_key]['data'][i+1]- self.PIR[data_key]['data'][i]==0).all():
                num_duplicated_frames+=1
        print('percent of duplicated frames: {0}'.format(num_duplicated_frames/num_frames))


        num_duplicates = np.empty((num_rows, num_cols))
        num_duplicates.fill(np.nan)

        for row in range(0, num_rows):
            for col in range(0, num_cols):

                time_series = self.PIR[data_key]['data'][index_start:index_end, row, col]

                num_duplicates[row, col] = sum( (time_series[1:]-time_series[:-1]==0) )

        num_duplicates = num_duplicates/num_frames

        print('percent of duplicates: \n{0}'.format(num_duplicates))

        self.cus_imshow(num_duplicates, cbar_limit=(0,1),title='Duplicate {0}, {1}'.format(self.dataset, data_key),
                        annotate=True, save_name='{0}_{1}'.format(self.dataset, data_key))


    # deprecated
    def down_sample(self, data_key=None, from_freq=128, to_freq=64, save_name=None):
        """
        Reduce the frequece by taking the average of consecutive frames.
        :param from_freq: original frequence
        :param to_freq: desired frequency
        :return: will be saved in a separate file
        """
        d_frame = int(from_freq/to_freq)

        data = []

        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        for i in range(d_frame, num_frames, d_frame):

            idx_start = i - d_frame
            idx_end = i

            data.append([self.PIR[data_key]['time'][i],
                         np.nanmean( self.PIR[data_key]['data'][idx_start:idx_end] , 0)])

        np.save(save_name, data)


    # =========================================================================
    # static methods for processing
    # =========================================================================
    @staticmethod
    def string_to_time(datetime_str=None):
        """
        This funciton parses the datestr in format %Y%m%d_%H:%M:%S_%f
        :param datetime_str: The datetime string, %Y%m%d_%H:%M:%S_%f
        :return: datetime type
        """
        if datetime_str is None:
            print 'Error: invalid date time string for string_to_time'
            return None
        else:
            return datetime.strptime(datetime_str, "%Y%m%d_%H:%M:%S_%f")

    @staticmethod
    def time_to_string(dt):
        """
        This function returns a string in format %Y%m%d_%H:%M:%S_%f
        :param dt: datetime type
        :return: str
        """
        return dt.strftime("%Y%m%d_%H:%M:%S_%f")

    @staticmethod
    def file_time_to_string(dt):
        """
        This function returns a string in format %Y%m%d_%H:%M:%S_%f
        :param dt: datetime type
        :return: str
        """
        return dt.strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def video_string_to_time(datetime_str=None):
        """
        This funciton parses the datestr in format %Y%m%d_%H:%M:%S_%f
        :param datetime_str: The datetime string, %Y%m%d_%H:%M:%S_%f
        :return: datetime type
        """
        if datetime_str is None:
            print 'Error: invalid date time string for string_to_time'
            return None
        else:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def video_time_to_string(dt):
        """
        This function returns a string in format %Y%m%d_%H:%M:%S_%f
        :param dt: datetime type
        :return: str
        """
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def get_index_in_period(timestamps=None, t_start=None, t_end=None):
        """
        This function returns the start and end index of entries between t_start and t_end.
        :param timestamps: a list of timestamps in datetime format
        :param t_start: datetime type
        :param t_end: datetime type
        :return: index_start, index_end   timestamps[index_start:index_end] gives the data in period
        """
        if t_start is None:
            index_start = 0
        else:
            index_start = bisect.bisect( timestamps, t_start )

        if t_end is None:
            index_end = len(timestamps)
        else:
            index_end = bisect.bisect( timestamps, t_end)

        if index_start == index_end:
            raise Exception('Error, no data found in period: {0} ~ {1}'.format(t_start, t_end))
        else:
            return index_start, index_end


    @staticmethod
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

    # =========================================================================
    # visualization methods
    # =========================================================================
    def plot_histogram_for_pixel(self, data_key=None, pixels=list(),
                                 t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.1,0.01)):
        """
        Statistic Analysis:
        This function plots the histogram of the raw data for a selected pixel, to better understand the noise
        :param dataset: the dataset identifier
        :param data_key: the options for the data, 'raw_data', 'temp_data',...
        :param pixels: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :param t_start: datetime type
        :param t_end: datetime type
        :return: one figure for each pixel
        """

        if data_key not in self.PIR.keys():
            raise Exception('Invalid data key')

        # compute the mean and std
        mu, sigma, noise_mu, noise_sigma = self.get_noise_distribution(data_key=data_key,
                                                                       t_start=t_start, t_end=t_end,
                                                                       p_outlier=p_outlier, stop_thres=stop_thres)

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                      t_start=t_start, t_end=t_end)

        for pixel in pixels:

            print('{0}: overall {1}, noise {2}'.format(pixel, (mu[pixel], sigma[pixel]),
                                                       (noise_mu[pixel], noise_sigma[pixel])))

            time_series = self.PIR[data_key]['data'][index_start:index_end, pixel[0], pixel[1]]

            # the histogram of the data
            num_bins = 200
            fig = plt.figure(figsize=(8,5), dpi=100)
            n, bins, patches = plt.hist(time_series, num_bins,
                                        normed=1, facecolor='green', alpha=0.75)

            # Need tuple for indexing. Make sure is tuple
            if not isinstance(pixel, tuple):
                pixel = tuple(pixel)

            # add a 'best fit' line
            norm_fit_line = mlab.normpdf(bins, noise_mu[pixel],
                                               noise_sigma[pixel])
            l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1.5, label='noise')
            norm_fit_line = mlab.normpdf(bins, mu[pixel],
                                               sigma[pixel])
            l = plt.plot(bins, norm_fit_line, 'b--', linewidth=1.5, label='all')

            plt.legend()
            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Probability density')
            plt.title(r'{0} pixel {1}; $\mu$= {2:.2f}, $\sigma$={3:.2f}'.format(self.dataset, pixel,
                                                                                         noise_mu[pixel],
                                                                                         noise_sigma[pixel]))
            plt.grid(True)

        plt.draw()

    def plot_sample_timing(self, data_key=None, save_name=None):
        """
        This function plots the timing of the samples
        :param dataset: the dataset
        :param save_name: if None, will plot but not save; otherwise will save in file
        :return:
        """
        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)

        print(self.PIR[data_key]['time'][101] - self.PIR[data_key]['time'][100]).total_seconds()

        dt = [ (self.PIR[data_key]['time'][i+1] - self.PIR[data_key]['time'][i]).total_seconds()
               for i in range(0, len(self.PIR[data_key]['time'])-1) ]

        # print dt
        ax.plot(dt)
        ax.set_title('The sampling timing for dataset {0}'.format(data_key), fontsize=20)
        ax.set_xlabel('Samples', fontsize=16)
        ax.set_ylabel('dt (seconds)', fontsize=16)

        print('mean: {0}'.format(np.mean(dt)))
        print('std: {0}'.format(np.std(dt)))

        if save_name is None:
            plt.draw()
        else:
            plt.savefig('../figs/{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()

        # plot the histogram of the sampling time
        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)
        num_bins = 200

        # remove outliers
        dt = np.asarray(dt)
        dt[dt>=1] = 1

        print(max(dt))
        n, bins, patches = plt.hist(dt, num_bins, normed=1, facecolor='green', alpha=0.75)
        ax.set_title('The sampling timing distribution for dataset {0}'.format(data_key), fontsize=20)
        ax.set_xlabel('dt (seconds)', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)

        if save_name is None:
            plt.draw()
        else:
            plt.savefig('../figs/{0}.png'.format(save_name+'_hist'), bbox_inches='tight')
            plt.clf()
            plt.close()


    # moved
    def plot_time_series_for_pixel(self, data_key=None,
                                   t_start=None, t_end=None, pixels=list()):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param dataset: the dataset identifier
        :param data_key: the data key identifier, 'raw_data'
        :param t_start: datetime type
        :param t_end: datetime type
        :param pixels: list of tuples, [(1,1),(1,5)]
        :return: a figure with all the pixels time series
        """

        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)

        if data_key not in self.PIR.keys():
            raise Exception('In correct data key definition: {0}'.format(data_key))

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                      t_start=t_start, t_end=t_end)

        for pixel in pixels:

            timestamps = self.PIR[data_key]['time'][index_start:index_end]
            time_series = self.PIR[data_key]['data'][index_start:index_end, pixel[0], pixel[1]]

            print('length of data to plot:{0}'.format(len(timestamps)))

            plt.plot(timestamps, time_series, label='{0} pixel {1}'.format(self.dataset, pixel))

        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.title('Time series')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time')

        plt.legend()
        plt.grid(True)
        plt.draw()

        return ax

    # moved
    def plot_heat_map_in_period(self, data_key=None, t_start=None, t_end=None,
                                cbar_limit=None, option='vec', nan_thres_p=None,
                                plot=False, folder=None, save_img=False, save_npy=False):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each vec(frame)
        :param data_key: the data identifier, 'raw', 'cleaned'...
        :param t_start: datetime type
        :param t_end: datetime type
        :param cbar_limit: the min and max value for the plot
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :param nan_thres_p: replace the data that within nan_thres_p probability by nan to highlight vehicles
        :return: a figure with 16 x n color map for n frame
        """

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        print('\nPlotting heat map for {0}, index {1} ~{2}'.format(data_key, index_start, index_end))

        # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        print('{0}x{1}x{2}'.format(num_frames, num_rows, num_cols))

        map = []
        if option == 'vec':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t].T.reshape(1, num_rows*num_cols).squeeze() )
        elif option == 'tworow':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
        elif option == '2nd':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][1,:].T.reshape(1, num_cols).squeeze() )
        elif option == '3rd':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][2,:].T.reshape(1, num_cols).squeeze() )
        elif option == 'max':
            for t in range(index_start, index_end):
                map.append( np.max(self.PIR[data_key]['data'][t], 0) )
        elif option == 'mean':
            for t in range(index_start, index_end):
                map.append( np.mean(self.PIR[data_key]['data'][t], 0) )

        map = np.array(map).T

        if nan_thres_p is not None:
            # Pr{ v \in [-v_thres, v_thres] } = nan_thres_p
            print('Warning: assuming dataset {0} is normalized.'.format(data_key))
            v_thres = stats.norm.ppf(1-(1-nan_thres_p)/2.0)
            map[ map <= v_thres ] = np.nan

        if save_img is True:
            t_start_str = self.file_time_to_string(t_start)
            _save_name = folder+'{0}__{1}'.format(t_start_str, int(np.sum(map)))
        else:
            _save_name = None

        self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit, title='{0} to {1}'.format(t_start, t_end),
                        annotate=False, patch=False,
                        plot=plot, save_name=_save_name)

        if save_npy is True:
            t_start_str = self.file_time_to_string(t_start)
            np.save(folder+'{0}_{1}.npy'.format(t_start_str, int(np.sum(map))), map)

    def plot_detected_veh_in_period(self, data_key=None, t_start=None, t_end=None,
                                cbar_limit=None, option='vec', nan_thres_p=None, det_thres=20,
                                plot=False, folder=None, save_img=False, save_npy=False):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each vec(frame)
        :param data_key: the data identifier, 'raw', 'cleaned'...
        :param t_start: datetime type
        :param t_end: datetime type
        :param cbar_limit: the min and max value for the plot
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :param nan_thres_p: replace the data that within nan_thres_p probability by nan to highlight vehicles
        :return: a figure with 16 x n color map for n frame
        """

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        print('\nPlotting heat map for {0}, index {1} ~{2}'.format(data_key, index_start, index_end))

        # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        print('{0}x{1}x{2}'.format(num_frames, num_rows, num_cols))

        map = []
        if option == 'vec':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t].T.reshape(1, num_rows*num_cols).squeeze() )
        elif option == 'tworow':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
        elif option == '2nd':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][1,:].T.reshape(1, num_cols).squeeze() )
        elif option == '3rd':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][2,:].T.reshape(1, num_cols).squeeze() )
        elif option == 'max':
            for t in range(index_start, index_end):
                map.append( np.max(self.PIR[data_key]['data'][t], 0) )
        elif option == 'mean':
            for t in range(index_start, index_end):
                map.append( np.mean(self.PIR[data_key]['data'][t], 0) )

        map = np.array(map).T

        # vehicle detection and plot
        vehs = self.det_veh_in_period(data_key=data_key, t_start=t_start, t_end=t_end, option='vec',
                                   det_thres=det_thres)

        print('detected vehs: {0}'.format(vehs))

        if nan_thres_p is not None:
            # Pr{ v \in [-v_thres, v_thres] } = nan_thres_p
            print('Warning: assuming dataset {0} is normalized.'.format(data_key))
            v_thres = stats.norm.ppf(1-(1-nan_thres_p)/2.0)
            map[ map <= v_thres ] = np.nan

        # save the entire heat map
        if save_img is True:
            t_start_str = self.file_time_to_string(t_start)
            _save_name = folder+'{0}_heatmap'.format(t_start_str)
        else:
            _save_name = None

        self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit, title='{0} to {1}'.format(t_start, t_end),
                        annotate=False, patch=True, patch_intervals=vehs,
                        plot=plot, save_name=_save_name)

        # save the heatmap for each vehicle
        if save_npy is True:
            for veh in vehs:
                # save at least one second clip
                if veh[1] - veh[0] < 64:
                    dt = 64 - (veh[1]-veh[0])
                    veh = (np.max([0, veh[0]-int(dt/2)]),
                           np.min([map.shape[1], veh[1]+int(dt/2)]) )

                # save the clip
                t_start_str = self.file_time_to_string(self.PIR[data_key]['time'][veh[0]])
                veh_data = {}
                veh_data['time'] = self.PIR[data_key]['time'][veh[0]:veh[1]]
                veh_data['data'] = map[:,veh[0]:veh[1]]
                np.save(folder+'{0}.npy'.format(t_start_str), veh_data)

                if save_img is True:
                    # save the image
                    self.cus_imshow(data_to_plot=map[:, veh[0]:veh[1]], cbar_limit=cbar_limit,
                                    title='{0} to {1}'.format(self.PIR[data_key]['time'][veh[0]],
                                                              self.PIR[data_key]['time'][veh[1]]),
                        annotate=False, patch=False, patch_intervals=None,
                        plot=False, save_name=folder+'{0}.png'.format(t_start_str))


    def det_veh_in_period(self, data_key=None, t_start=None, t_end=None, option='vec', det_thres=20):
        """
        This function detects the vehicles for data_key during time t_start to t_end
        :param data_key:
        :param t_start:
        :param t_end:
        :param option:
        :return:
        """
        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        print('\nPlotting heat map for {0}, index {1} ~{2}'.format(data_key, index_start, index_end))

        # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape

        print('{0}x{1}x{2}'.format(num_frames, num_rows, num_cols))

        map = []
        if option == 'vec':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t].T.reshape(1, num_rows*num_cols).squeeze() )
        elif option == 'tworow':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
        elif option == 'max':
            for t in range(index_start, index_end):
                map.append( np.max(self.PIR[data_key]['data'][t], 0) )
        elif option == 'mean':
            for t in range(index_start, index_end):
                map.append( np.mean(self.PIR[data_key]['data'][t], 0) )

        map = np.array(map).T

        # ----------------------------------------------------
        # TRY 1:
        #   - Use a 1 s hanning window (64 points) to smooth the entropy over time.
        #   - Apply a threshold in the smoothed entropy signal and determine the vehicle

        entropy = np.sum(map, 0)
        print('entropy mean: {0}'.format(np.mean(entropy)))
        # smooth the entropy
        window_len = 32
        hanning_w = np.hanning(window_len)
        rectangle_w = np.ones(window_len)/window_len
        entropy_smoothed = np.convolve(entropy, rectangle_w, 'valid')
        print('smoothed entropy mean: {0}'.format(np.mean(entropy_smoothed)))
        # pad zero to both side
        entropy_smoothed = np.concatenate((np.zeros(int(window_len/2)), entropy_smoothed, np.zeros(int(window_len/2))))

        # plot entropy
        print('Imshow map dimension:{0}'.format(map.shape))
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(entropy, linewidth=2, color='b')
        ax.plot(entropy_smoothed, linewidth=2, color='r')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Entropy', fontsize=18)

        # use threshold to determine the pass of each vehicle
        flag = False    # detected
        vehs = []
        for head in range(0, len(entropy_smoothed)):
            if flag is False:
                if entropy_smoothed[head] >= det_thres:
                    tail = head
                    flag = True
            else:
                if entropy_smoothed[head] < det_thres:
                    # TODO: found one interval, to apply a simple filter
                    # - if head-pre_tail <= 20 (0.3s headway), combine
                    # - if total still <= 0.5 s, discard
                    vehs.append((tail, head))
                    flag = False
                    ax.plot([tail, head], [det_thres, det_thres], linewidth=2, color='g')

        plt.draw()

        return vehs


    # moved
    def nonlinear_transform(self, data, ratio=6.0):
        """
        This function applies a nonlinear transform to the data and save in image
        :param data: a dict; 'time': datetime timestamps, 'data': 128xnum_frames
        :param ratio: the space time ratio
        :return:
        """
        # 120 degree field of view
        _dup = data['data'].shape[0]/32
        d_theta = (60.0/16)*np.pi/180.0
        spaces = []
        for i in range(-16, 16):
            for d in range(0, _dup):
                # duplicate the nonlinear operator for vec
                spaces.append( np.tan(d_theta/2 + i*d_theta) )
        spaces = -np.asarray(spaces)/ratio

        # plot the original
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data['data'], cmap=plt.get_cmap('jet'),
                        interpolation='nearest', aspect='auto',
                        vmin=2.0, vmax=6.0)
        ax.set_title('Original', fontsize=20)
        ax.set_xlabel('Time')
        ax.set_ylabel('space')
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        fig.colorbar(im, cax=cax, orientation='vertical')

        plt.draw()

        fig, ax = plt.subplots(figsize=(10, 10))
        _dt = data['time'] - data['time'][0]
        dt = np.asarray([i.total_seconds() for i in _dt])
        X, Y = np.meshgrid(dt, spaces)
        sc = ax.scatter(X, Y, c=data['data'], vmin=2.0, vmax=6.0, cmap=plt.get_cmap('jet'))
        ax.set_title('Nonlinear transform', fontsize=20)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('space')
        ax.set_xlim([np.min(dt), np.max(dt)])
        ax.set_ylim([np.min(spaces), np.max(spaces)])
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        plt.colorbar(sc, cax=cax, orientation='vertical')
        plt.draw()


    # deprecated
    # to include nonlinear transformation
    def get_img_in_period(self, data_key=None, t_start=None, t_end=None,
                                cbar_limit=(0,1), option='vec', nan_thres=None, plot=False,
                                folder=None):
        """
        Visualization:
        This function returns the heat image during time.
        :param dataset: the dataset identifier, default the name
        :param data_key: the data identifier, 'raw', 'cleaned'...
        :param t_start: datetime type
        :param t_end: datetime type
        :param cbar_limit: the min and max value for the plot
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :param dur: seconds, into 1 seconds segments
        :param folder: save the img as png to folder associated with npy data.
        :return: a figure with 16 x n color map for n frame
        """

         # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[data_key]['data'].shape


        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_key]['time'],
                                                          t_start=t_start, t_end=t_end)

        map = []
        if option == 'vec':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][:,:,t].T.reshape(1, num_rows*num_cols).squeeze() )
        elif option == 'tworow':
            for t in range(index_start, index_end):
                map.append( self.PIR[data_key]['data'][:,:,t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
        elif option == 'max':
            for t in range(index_start, index_end):
                map.append( np.max(self.PIR[data_key]['data'][:,:,t], 0) )
        elif option == 'mean':
            for t in range(index_start, index_end):
                map.append( np.mean(self.PIR[data_key]['data'][:,:,t], 0) )

        map = np.array(map).T

        if nan_thres is not None:
            map[ map<=nan_thres ] = 0
            map[ map>nan_thres ] = 1

        t_start_str = self.file_time_to_string(t_start)

        self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit,
                        title='{0}'.format(t_start_str),
                        annotate=False,save_name=folder+'{0}__{1}'.format(t_start_str, int(np.sum(map))))

        np.save(folder+'{0}_{1}.npy'.format(t_start_str, int(np.sum(map))), map)


        # times = self.PIR[dataset]['time'][index_start:index_end]
        # spaces = []
        # if num_cols == 16:
        #     # 60 degree FOV
        #     d_theta = (60.0/16)*np.pi/180.0
        #     for i in range(-7, 1):
        #         spaces.append( np.tan(-d_theta/2 + i*d_theta) )
        #     for i in range(0, 8):
        #         spaces.append( np.tan(d_theta/2 + i*d_theta) )

        # plot and compare the nonlinear transform
        # if plot is True:
        #     self.cus_imshow(map, cbar_limit, '{0} to {1}'.format(t_start, t_end))
            #
            # fig, ax = plt.subplots(figsize=(18, 8))
            #
            # x = []
            # y = []
            #
            # img_res = map.shape
            # for i in range(0, img_res[0]):
            #     for j in range(0, img_res[1]):
            #         val = map[i,j]
            #         if not np.isnan(val):
            #             x.append(times[j])
            #             y.append(spaces[i])
            #
            # print len(x)
            # print len(y)
            #
            # plt.scatter(x,y)
            # plt.draw()

        # return map

    # deprecated
    # To include nonlinear transformation and vehicle detection
    def get_veh_img_in_period(self, dataset=None, data_key=None, t_start=None, t_end=None,
                                cbar_limit=(0,1), option='vec', nan_thres=None, plot=False,
                               dur=1, folder=None, fps=None):
        """
        Visualization:
        This function aims at plotting the heatmap of only the cars.
        :param dataset: the dataset identifier, default the name
        :param data_key: the data identifier, 'raw', 'cleaned'...
        :param t_start: datetime type
        :param t_end: datetime type
        :param cbar_limit: the min and max value for the plot
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :param dur: seconds, into 1 seconds segments
        :param folder: save the img as png to folder associated with npy data.
        :return: a figure with 16 x n color map for n frame
        """

         # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[dataset][data_key].shape

        index_start_p, index_end_p = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                          t_start=t_start+timedelta(seconds=dur),
                                                          t_end=t_end)

        d_frame = dur*fps

        var = []
        for i in range(index_start_p, index_end_p):

            index_start = i - d_frame
            index_end = i

            map = []
            if option == 'vec':
                for t in range(index_start, index_end):
                    map.append( self.PIR[dataset][data_key][t].T.reshape(1, num_rows*num_cols).squeeze() )
            elif option == 'tworow':
                for t in range(index_start, index_end):
                    map.append( self.PIR[dataset][data_key][t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
            elif option == 'max':
                for t in range(index_start, index_end):
                    map.append( np.max(self.PIR[dataset][data_key][t], 0) )
            elif option == 'mean':
                for t in range(index_start, index_end):
                    map.append( np.mean(self.PIR[dataset][data_key][t], 0) )

            map = np.array(map).T

            var.append( np.mean(map) )

            # if np.mean(map) >- 0.25:
            #     t_1_str = self.file_time_to_string( self.PIR[dataset]['time'][index_start] )
            #
            #     if nan_thres is not None:
            #         map[ map<=nan_thres ] = 0
            #         map[ map>nan_thres ] = 1
            #
            #     self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit,
            #                     title='{0}'.format(t_1_str),
            #                     annotate=False,save_name=folder+'{0}__{1}'.format(t_1_str, int(np.sum(map))))
            #
            #     np.save(folder+'{0}__{1}.npy'.format(t_1_str, int(np.sum(map))), map)

        fig, ax = plt.subplots(figsize=(18, 8))
        plt.plot(var)
        plt.draw()


        # times = self.PIR[dataset]['time'][index_start:index_end]
        # spaces = []
        # if num_cols == 16:
        #     # 60 degree FOV
        #     d_theta = (60.0/16)*np.pi/180.0
        #     for i in range(-7, 1):
        #         spaces.append( np.tan(-d_theta/2 + i*d_theta) )
        #     for i in range(0, 8):
        #         spaces.append( np.tan(d_theta/2 + i*d_theta) )

        # plot and compare the nonlinear transform
        # if plot is True:
        #     self.cus_imshow(map, cbar_limit, '{0} to {1}'.format(t_start, t_end))
            #
            # fig, ax = plt.subplots(figsize=(18, 8))
            #
            # x = []
            # y = []
            #
            # img_res = map.shape
            # for i in range(0, img_res[0]):
            #     for j in range(0, img_res[1]):
            #         val = map[i,j]
            #         if not np.isnan(val):
            #             x.append(times[j])
            #             y.append(spaces[i])
            #
            # print len(x)
            # print len(y)
            #
            # plt.scatter(x,y)
            # plt.draw()

        # return map

    # moved
    def cus_imshow(self, data_to_plot=None, cbar_limit=None, title=None, annotate=False,
                   patch=False, patch_intervals=None,
                   plot=False, save_name=None):
        """
        Visualization:
        This function is a universal function for plotting a 2d color map.
        :param data_to_plot: a 2D float array with values to be plotted
        :param cbar_limit: the color bar limits
        :param title: string, the title for the figure
        :param annotate: if true, will overlay the values on the imshow image
        :param save_name: if not None, will save figure with the name
        :return: a 2d colormap figure
        """

        if cbar_limit is None:
            v_min = np.min(data_to_plot)
            v_max = np.max(data_to_plot)
        else:
            v_min = cbar_limit[0]
            v_max = cbar_limit[1]

        print 'Temperature range: {0} ~ {1}\n'.format(np.nanmin(data_to_plot), np.nanmax(data_to_plot))

        # adjust the figure width and height to best represent the data matrix
        # Best figure size (,12):(480,192)
        row, col = data_to_plot.shape
        fig_height = 8
        fig_width = 18

        # fig_width = int(0.8*col/row)*12
        # if fig_width >= 2000:
        #     # two wide
        #     fig_width = 2000

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('auto')
        # pos1 = ax.get_position() # get the original position
        # pos2 = [pos1.x0 - 0.12, pos1.y0 ,  pos1.width*1.272, pos1.height*1.25]
        # ax.set_position(pos2)

        im = ax.imshow(data_to_plot,
                        cmap=plt.get_cmap('jet'),
                        interpolation='nearest', aspect='auto',
                        vmin=v_min, vmax=v_max)

        if annotate is True:
            # add values to pixels
            for (j, i), label in np.ndenumerate(data_to_plot):
                if label < 10:
                    label = round(label,2)
                else:
                    label = round(label,1)

                ax.text(i,j,label,ha='center',va='center')

        # patch semi-opaque boxes to img
        if patch is True:
            for interval in patch_intervals:
                rect = patches.Rectangle((interval[0], 0), interval[1]-interval[0], 128, linewidth=1, edgecolor='r',
                                         facecolor=(1,0,0,0.5))
                ax.add_patch(rect)

        plt.title('{0}'.format(title))
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        fig.colorbar(im, cax=cax, orientation='vertical')

        if plot is True:
            plt.draw()

        if save_name is not None:
            plt.savefig('{0}.png'.format(save_name), bbox_inches='tight')

            # do not close the window
            if plot is not True:
                plt.clf()
                plt.close()


    def play_thermal_video(self, data_keys=None, cbar_limits=None,
                    t_start=None, t_end=None, speed=1):
        """
        This function plays the heat map video
            It can also stack multiple data_key in the same frame. E.g., raw data and normalized data.
        :param data_keys: the keys in self.PIR
        :param cbar_limits: [(v_min, v_max),(),...], color bar limits corresponding to each data key
        :param t_start: datetime
        :param t_end: datetime
        :param speed: Speed of video. 1: realtime; 2: 2x faster; 0.5: 2x slower
        :return: A video plotting
        """
        num_plots = len(data_keys)

        if num_plots == 1:

            # initialize figure
            fig, ax = plt.subplots(figsize=(15,10))

            ax.set_aspect('auto')
            ax.set_xlim(-0.5, 15.5)
            ax.set_ylim(-0.5, 3.5)
            ax.hold(True)
            # cache the background
            background = fig.canvas.copy_from_bbox(ax.bbox)

            # Initialize with 4x48 full pixel frame. Will change over the time
            num_frames, num_rows, num_cols = self.PIR[data_keys[0]]['data'].shape
            T_init = np.zeros((num_rows, num_cols)) + cbar_limits[0][0]
            image = ax.imshow(T_init, cmap=plt.get_cmap('jet'),
                                interpolation='nearest', vmin=cbar_limits[0][0], vmax=cbar_limits[0][1])

            # add some initial figure properties
            # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            # fig.colorbar(image, cax=cax)
            ax.set_title('{0}'.format(data_keys[0]))
            # ax.set_xlabel('Column')
            # ax.set_ylabel('Row')

        else:
            fig, ax = plt.subplots(num_plots, 1, figsize=(15,10))

            background = []
            image = []
            for i in range(0, num_plots):
                ax[i].set_aspect('auto')
                ax[i].set_xlim(-0.5, 15.5)
                ax[i].set_ylim(-0.5, 3.5)
                ax[i].hold(True)
                background.append( fig.canvas.copy_from_bbox(ax[i].bbox) )

                # Initialize with 4x48 full pixel frame. Will change over the time
                num_frames, num_rows, num_cols = self.PIR[data_keys[i]]['data'].shape

                T_init = np.zeros((num_rows, num_cols)) + cbar_limits[i][0]
                image.append(ax[i].imshow(T_init, cmap=plt.get_cmap('jet'),
                                    interpolation='nearest', vmin=cbar_limits[i][0], vmax=cbar_limits[i][1]) )

                # add some initial figure properties
                # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
                # fig.colorbar(image[i], cax=cax)
                ax[i].set_title('{0}'.format(data_keys[i]))
                # ax.set_xlabel('Column')
                # ax.set_ylabel('Row')

        plt.show(False)
        plt.draw()

        # play all frames in the data options
        if num_plots == 1:
            # find the data in the time interval
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_keys[0]]['time'],
                                                          t_start=t_start, t_end=t_end)

            t0 = None
            for t in range(index_start, index_end-1):
                t1 = time.time()
                if t0 is not None:
                    print('fps: {0}'.format(1/(t1-t0)))
                t0 = t1

                image.set_data(self.PIR[data_keys[0]]['data'][t, :, :])
                fig.canvas.restore_region(background)
                ax.draw_artist(image)
                fig.canvas.blit(ax.bbox)

        else:
            # Use the time scale from the first data set
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[data_keys[0]]['time'],
                                                          t_start=t_start, t_end=t_end)

            # t0 = None
            for frame_idx in range(index_start, index_end-1):
                # t1 = time.time()
                # if t0 is not None:
                #     print('fps: {0}'.format(1/(t1-t0)))
                # t0 = t1

                # plot the first dataset
                image[0].set_data(self.PIR[data_keys[0]]['data'][frame_idx, :, :])
                fig.canvas.restore_region(background[0])
                ax[0].draw_artist(image[0])
                fig.canvas.blit(ax[0].bbox)

                # for other dataset, find the corresponding frame
                for i in range(1, num_plots):

                    # current time
                    t = self.PIR[data_keys[0]]['time'][frame_idx]
                    idx = self.PIR[data_keys[i]]['time'].index(t)

                    image[i].set_data(self.PIR[data_keys[i]]['data'][idx, :, :])
                    fig.canvas.restore_region(background[i])
                    ax[i].draw_artist(image[i])
                    fig.canvas.blit(ax[i].bbox)


    def trim_video(self, input_video=None, input_timestamp=None, trim_period=None, output_video=None):
        """
        This function trims the video and saves the trimmed video in period
        :param input_video: str, the input video file
        :param input_timestamp: datetime
        :param trim_period: (start, end), datetime
        :param output_video: str, output video file name
        :return:
        """
        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

        # get the timestamp
        print('Loaded video {0}:'.format(input_video))
        print('-- Current timestamp: {0}'.format(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)))
        print('-- Current index: {0}'.format(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
        print('-- Resolution: {0} x {1}'.format(res[0], res[1]))
        print('-- FPS: {0}'.format(fps))
        print('-- Frame count: {0}'.format(total_frames))

        # compute the index of frames to trim
        index_start = int( (trim_period[0]-input_timestamp).total_seconds()*fps )
        index_end = int( (trim_period[1]-input_timestamp).total_seconds()*fps )
        num_frames = index_end-index_start

        print('Extracting video...')

        # set the current frame
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index_start)

        # create writerCV
        fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
        # fourcc = cv2.cv.CV_FOURCC('a', 'v', 'c', '1')   # use H.264 compression
        out = cv2.VideoWriter(output_video, fourcc, fps, res)

        for i in range(0, num_frames):
            ret, frame = cap.read()

            time_str = self.video_time_to_string( trim_period[0] + timedelta(seconds=i/fps) )
            #
            cv2.putText(frame, time_str, (150,50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255))

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


    def play_video(self, input_video=None, input_timestamp=None, trim_period=None):
        """
        This function trims the video and saves the trimmed video in period
        :param input_video: str, the input video file
        :param input_timestamp: datetime
        :param trim_period: (start, end), datetime
        :return:
        """
        cap = cv2.VideoCapture(input_video)

        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        res = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

        # get the timestamp
        print('Loaded video {0}:'.format(input_video))
        print('-- Current timestamp: {0}'.format(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)))
        print('-- Current index: {0}'.format(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
        print('-- Resolution: {0} x {1}'.format(res[0], res[1]))
        print('-- FPS: {0}'.format(fps))
        print('-- Frame count: {0}'.format(total_frames))

        # compute the index of frames to trim
        index_start = int( (trim_period[0]-input_timestamp).total_seconds()*fps )
        index_end = int( (trim_period[1]-input_timestamp).total_seconds()*fps )
        num_frames = index_end-index_start

        # set the current frame
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index_start)

        for i in range(0, num_frames):
            ret, frame = cap.read()

            time_str = self.video_time_to_string( trim_period[0] + timedelta(seconds=i/fps) )
            #
            cv2.putText(frame, time_str, (150,50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def save_thermal_to_avi(self, data_key='norm_temp_data', fps=64):
        """
        This function saves the selected data in avi file
        :param dataset: the sensor id
        :param data_key: 'raw_data' or 'temp_data'
        :return:
        """
        scale  = 64

        fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
        out = cv2.VideoWriter( self.dataset + ".avi", fourcc, fps, (16*scale,4*scale))
        pir_cam = []

        for frame in self.PIR[data_key]['data'][:]:
            #print frame
            img = np.asarray(frame).astype(np.uint8)
            img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            out.write(img)

            cv2.imshow("PIR Cam", img)
            cv2.waitKey(10)

        out.release()
        cv2.destroyAllWindows()

    # moved
    def plot_noise_evolution(self, data_key=None, p_outlier=0.05, stop_thres=(0.5, 0.1),
                             pixel=None, window_s=120, step_s=30, fps=64):
        """
        This function plots the evolution of the
        :param dataset: the file name
        :param data_key: the data identifier
        :param p_outlier: [0,1], points with probability below the value is considered as outlier
        :param stop_thres: (delta_mu, delta_sigma) degrees in temperature
        :param pixel: a list of tuples [0,3] x [0,15]
        :param window_s: seconds, the duration of the window for computing the mu and std
        :return:
        """
        dt_w = timedelta(seconds=window_s)

        times = []
        mu = []
        sigma = []

        # compute the number of frames that corresponds to the duration in seconds
        d_frame = int(fps*step_s)
        start_frame = int(fps*window_s)

        num_frames= len(self.PIR[data_key]['time'])

        print start_frame
        print num_frames
        print d_frame

        for i in range(start_frame, num_frames, d_frame):
            _, _, _mu, _sigma = self.get_noise_distribution(data_key=data_key,
                                                             t_start= self.PIR[data_key]['time'][i]-dt_w,
                                                             t_end=self.PIR[data_key]['time'][i],
                                                             p_outlier=p_outlier,stop_thres=stop_thres,
                                                             pixels=[pixel], suppress_output=True)
            # see from the center of the window
            times.append(self.PIR[data_key]['time'][i] - timedelta(seconds=window_s/2.0))
            mu.append(_mu[pixel])
            sigma.append(_sigma[pixel])

            self.print_loop_status('Evolving noise ', i, num_frames)
        print('\n')
        print('mu: {0}~{1}'.format(np.nanmin(mu), np.nanmax(mu)))
        print('Average sigma: {0}'.format(np.nanmean(sigma)))


        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        # plot the time series
        ax = self.plot_time_series_for_pixel(data_key=data_key, t_start=None, t_end=None, pixels=[pixel])

        ax.plot(times, mu, color='r', linewidth=2)
        ax.plot(times, mu+sigma, color='r', linestyle='--', linewidth=2)
        ax.plot(times, mu-sigma, color='r', linestyle='--',linewidth=2)

        plt.draw()