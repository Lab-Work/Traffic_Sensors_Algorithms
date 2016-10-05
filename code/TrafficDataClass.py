__author__ = 'Yanning Li (yli171@illinois.edu), Fangyu Wu (fwu10@illinois.edu)'

"""
The classes interface with the data files. It reads the data, conducts statistic analysis of the data, and visualizes
the data.

The data input should be one single csv file with the format specified in the Data_Collection_Manual.pdf.
This class can handle different format of the data, e.g. PIR data with different number of pixels, or data missing the
ultrasonic sensor measurement.

The structure of each class:
# - Visualization:
# - Statistic analysis:

"""


import numpy as np
import matplotlib
from scipy import stats
matplotlib.use('TkAgg')
import bisect
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
from sklearn import mixture
import cv2



class TrafficData:
    """
    This class is a universal class which deals with different types of data specified in the data collection manual.
    """

    def __init__(self):

        # There may be multiple PIR data formats: 1x16, 4x48.... Use key-value store
        # PIR['1x16'] = {'time':narray, 'Ta':narray, 'raw_data':narray, 'cleaned_data':narray}
        # raw_data and cleaned_data are 3d array, first index is the frames
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

            # check if read other types of data previously
            if dataset not in self.PIR.keys():
                self.PIR[dataset] = OrderedDict()

            if 'time' not in self.PIR[dataset].keys():
                # get the time stamp
                self.PIR[dataset]['time'] = []
                for entry in data:
                    self.PIR[dataset]['time'].append(entry[0])

                self.PIR[dataset]['time'] = np.array(self.PIR[dataset]['time'])

            if data_key not in self.PIR[dataset].keys():
                # get the data
                self.PIR[dataset][data_key] = []
                for entry in data:
                    self.PIR[dataset][data_key].append( entry[1] )
                self.PIR[dataset][data_key] = np.array(self.PIR[dataset][data_key])

        else:
            raise Exception('Dataset file not exists: {0}'.format(file_name_str))


    def get_data_periods(self, dir, update=True):
        """
        This function returns the periods for all data collection experiments saved in a directory
        :param dir: the directory
        :return: save in file and return the periods
        """
        periods = OrderedDict()
        f_periods = dir.replace('*.npy', '')+'dataset_periods.txt'

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
            print('Updated dataset_periods.txt.')
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


    def get_noise_distribution(self, dataset=None, data_key=None, t_start=None, t_end=None,
                               p_outlier=0.01, stop_thres=(0.1, 0.01), pixels=None, suppress_output=False):
        """
        This function returns the noise distribution. It iteratively throw away the data points that has a probability
        below p_outlier and refit a gaussian distribution. It stops when the change fo the mean and std are within the
        specified stop criteria.
        :param dataset: the dataset
        :param data_key: the key of the dataset. Processed data will be saved in the same dataset but with differnt keys
        :param t_start: datetime type
        :param t_end: datatime type
        :param p_outlier: [0,1], a point is considered as outlier if it is below p_outlier
        :param stop_thres: (delta_mu, delta_sigma) degrees in temperature
        :param pixels: a list of tuples row [0,3], col [0,15] or [0,31]
        :return: save in self.PIR[dataset]['mean'], self.PIR[dataset]['std'],
                         self.PIR[dataset]['noise_mean'], self.PIR[dataset]['noise_std'],
        """

        num_frames, num_rows, num_cols = self.PIR[dataset][data_key].shape

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
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

            time_series = self.PIR[dataset][data_key][index_start:index_end, row, col]

            # save the initial estimate
            mean[row, col] = np.nanmean( time_series )
            std[row, col] = np.nanstd( time_series )

            _pre_mean = mean[row, col]
            _pre_std = std[row, col]

            # compute the mean and std of the noise
            p = mlab.normpdf(time_series, _pre_mean, _pre_std)

            _mean = np.nanmean( time_series[ p>=p_outlier] )
            _std = np.nanstd( time_series[ p>=p_outlier] )

            while np.abs(_mean - _pre_mean) > stop_thres[0] or np.abs(_std - _pre_std) > stop_thres[1]:
                if suppress_output is False:
                    print('updating noise distribution for pixel {0}'.format((row, col)))
                _pre_mean = _mean
                _pre_std = _std
                p = mlab.normpdf(time_series, _pre_mean, _pre_std)
                _mean = np.nanmean( time_series[ p>=p_outlier] )
                _std = np.nanstd( time_series[ p>=p_outlier] )

            # save the final noise distribution
            noise_mean[row, col] = _mean
            noise_std[row, col] = _std

        self.PIR[dataset]['mean'] = mean
        self.PIR[dataset]['std'] = std
        self.PIR[dataset]['noise_mean'] = noise_mean
        self.PIR[dataset]['noise_std'] = noise_std

        return mean, std, noise_mean, noise_std


    def normalize_data(self, dataset=None, data_key=None, norm_data_key=None, t_start=None, t_end=None):
        """
        This function normalizes the data in the period
        :param dataset: the dataset identifier
        :param data_key: the key of the data to be normalized
        :param norm_data_key: the key of the normalized data
        :param t_start: datetime type
        :param t_end: datetime type
        :return:
        """

        if norm_data_key is None:
            norm_data_key = 'norm_' + data_key

        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                          t_start=t_start, t_end=t_end)

        timestamps = norm_data_key + 'time'
        self.PIR[dataset][timestamps] = self.PIR[dataset]['time'][index_start:index_end]

        if 'noise_mean' not in self.PIR[dataset].keys() or 'noise_std' not in self.PIR[dataset].keys():
            print('Warning: using default values to compute the noise mean and std')
            self.get_noise_distribution(dataset=dataset, data_key=data_key, t_start=t_start, t_end=t_end)

        noise_mu = self.PIR[dataset]['noise_mean']
        noise_std = self.PIR[dataset]['noise_std']

        self.PIR[dataset][norm_data_key] = (self.PIR[dataset][data_key][index_start:index_end] - noise_mu)/noise_std


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


    def check_duplicates(self, dataset=None, data_key=None, t_start=None, t_end=None):
        """
        This function checks if there is duplicated data
        :param dataset:
        :param data_key:
        :param t_start:
        :param t_end:
        :return:
        """
        num_frames, num_rows, num_cols = self.PIR[dataset][data_key].shape

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                          t_start=t_start, t_end=t_end)

        # find the duplicated frames
        num_duplicated_frames = 0
        for i in range(index_start, index_end-1):
            if (self.PIR[dataset][data_key][i+1]- self.PIR[dataset][data_key][i]==0).all():
                num_duplicated_frames+=1
        print('percent of duplicated frames: {0}'.format(num_duplicated_frames/num_frames))


        num_duplicates = np.empty((num_rows, num_cols))
        num_duplicates.fill(np.nan)

        for row in range(0, num_rows):
            for col in range(0, num_cols):

                time_series = self.PIR[dataset][data_key][index_start:index_end, row, col]

                num_duplicates[row, col] = sum( (time_series[1:]-time_series[:-1]==0) )

        num_duplicates = num_duplicates/num_frames

        print('percent of duplicates: \n{0}'.format(num_duplicates))

        self.cus_imshow(num_duplicates, cbar_limit=(0,1),title='Duplicates {0}'.format(dataset),
                        annotate=True, save_name='{0}'.format(dataset))




    def down_sample(self, dataset=None, data_key=None, from_freq=128, to_freq=64, save_name=None):
        """
        Reduce the frequece by taking the average of consecutive frames.
        :param from_freq: original frequence
        :param to_freq: desired frequency
        :return: will be saved in a separate file
        """
        d_frame = int(from_freq/to_freq)

        data = []

        num_frames, num_rows, num_cols = self.PIR[dataset][data_key].shape

        for i in range(d_frame, num_frames, d_frame):

            idx_start = i - d_frame
            idx_end = i

            data.append([self.PIR[dataset]['time'][i],
                         np.nanmean( self.PIR[dataset][data_key][idx_start:idx_end] , 0)])

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
    def plot_histogram_for_pixel(self, dataset=None, data_key=None, pixels=list(),
                                 t_start=None, t_end=None):
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

        if dataset not in self.PIR.keys() or data_key not in self.PIR[dataset].keys():
            raise Exception('Invalid dataset of key identifier')

        # compute the mean and std
        if 'noise_mean' not in self.PIR[dataset].keys() or 'noise_std' not in self.PIR[dataset].keys():
            print('Warning: using default values to compute the noise mean and std')
            mu, sigma, noise_mu, noise_sigma = self.get_noise_distribution(dataset=dataset, data_key=data_key,
                                                                       t_start=t_start, t_end=t_end)
        mu = self.PIR[dataset]['mean']
        sigma = self.PIR[dataset]['std']
        noise_mu = self.PIR[dataset]['noise_mean']
        noise_sigma = self.PIR[dataset]['noise_std']

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                      t_start=t_start, t_end=t_end)

        for pixel in pixels:

            print('{0}: overall {1}, noise {2}'.format(pixel, (mu[pixel], sigma[pixel]),
                                                       (noise_mu[pixel], noise_sigma[pixel])))

            time_series = self.PIR[dataset][data_key][index_start:index_end, pixel[0], pixel[1]]

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
            l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1.5)
            norm_fit_line = mlab.normpdf(bins, mu[pixel],
                                               sigma[pixel])
            l = plt.plot(bins, norm_fit_line, 'b--', linewidth=1.5)

            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Probability')
            plt.title(r'{0} pixel {1}; $\mu$= {2:.2f}, $\sigma$={3:.2f}'.format(dataset, pixel,
                                                                                         noise_mu[pixel],
                                                                                         noise_sigma[pixel]))
            plt.grid(True)

        plt.draw()

    def plot_sample_timing(self, dataset=None, save_name=None):
        """
        This function plots the timing of the samples
        :param dataset: the dataset
        :param save_name: if None, will plot but not save; otherwise will save in file
        :return:
        """
        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)

        print(self.PIR[dataset]['time'][101] - self.PIR[dataset]['time'][100]).total_seconds()

        dt = [ (self.PIR[dataset]['time'][i+1] - self.PIR[dataset]['time'][i]).total_seconds()
               for i in range(0, len(self.PIR[dataset]['time'])-1) ]

        # print dt
        ax.plot(dt)
        ax.set_title('The sampling timing for dataset {0}'.format(dataset), fontsize=20)
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


    def plot_time_series_for_pixel(self, dataset=None, data_key=None,
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

        if dataset not in self.PIR.keys():
            print 'Error: incorrect pixel definition.'
            return -1

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                      t_start=t_start, t_end=t_end)

        for pixel in pixels:

            timestamps = self.PIR[dataset]['time'][index_start:index_end]
            time_series = self.PIR[dataset][data_key][index_start:index_end, pixel[0], pixel[1]]

            print('length of data to plot:{0}'.format(len(timestamps)))

            plt.plot(timestamps, time_series, label='{0} pixel {1}'.format(dataset, pixel))

        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.title('Time series')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time')

        plt.legend()
        plt.grid(True)
        plt.draw()

        return ax


    def plot_heat_map_in_period(self, dataset=None, data_key=None, t_start=None, t_end=None,
                                cbar_limit=None, option='vec', nan_thres=None):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each vec(frame)
        :param dataset: the dataset identifier, default the name
        :param data_key: the data identifier, 'raw', 'cleaned'...
        :param t_start: datetime type
        :param t_end: datetime type
        :param cbar_limit: the min and max value for the plot
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :return: a figure with 16 x n color map for n frame
        """

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                          t_start=t_start, t_end=t_end)

        print index_start, index_end
        # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[dataset][data_key].shape

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

        if nan_thres is not None:
            map[ map<=nan_thres ] = np.nan

        self.cus_imshow(map, cbar_limit, '{0} to {1}'.format(t_start, t_end))


    def get_img_in_period(self, dataset=None, data_key=None, t_start=None, t_end=None,
                                cbar_limit=(0,1), option='vec', nan_thres=None, plot=False,
                                folder=None):
        """
        Visualization:
        This function returns the heat image along with a nonlinear transform
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


        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                          t_start=t_start, t_end=t_end)

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

        if nan_thres is not None:
            map[ map<=nan_thres ] = 0
            map[ map>nan_thres ] = 1

        t_start_str = self.file_time_to_string(t_start)

        self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit,
                        title='{0}'.format(t_start_str),
                        annotate=False,save_name=folder+'{0}__{1}'.format(t_start_str, int(np.sum(map))))

        np.save(folder+'{0}__{1}.npy'.format(t_start_str, int(np.sum(map))), map)


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


    def get_heat_img_in_period(self, dataset=None, data_key=None, t_start=None, t_end=None,
                                cbar_limit=(0,1), option='vec', nan_thres=None, plot=False,
                               dur=1, folder=None):
        """
        Visualization:
        This function returns the heat image along with a nonlinear transform
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

        t_1 = t_start
        t_2 = t_1 + timedelta(seconds=dur)

        while t_2 <= t_end-timedelta(seconds=dur):
            # find the data in the time interval
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[dataset]['time'],
                                                              t_start=t_1, t_end=t_2)

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

            if nan_thres is not None:
                map[ map<=nan_thres ] = 0
                map[ map>nan_thres ] = 1

            t_1_str = self.file_time_to_string(t_1)

            self.cus_imshow(data_to_plot=map, cbar_limit=cbar_limit,
                            title='{0}'.format(self.time_to_string(t_1)),
                            annotate=False,save_name=folder+'{0}__{1}'.format(t_1_str, int(np.sum(map))))

            np.save(folder+'{0}__{1}.npy'.format(t_1_str, int(np.sum(map))), map)

            # update the time
            t_1 = t_2
            t_2 = t_1 + timedelta(seconds=dur)


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


    def get_veh_img_in_period(self, dataset=None, data_key=None, t_start=None, t_end=None,
                                cbar_limit=(0,1), option='vec', nan_thres=None, plot=False,
                               dur=1, folder=None, fps=None):
        """
        Visualization:
        This function returns the heat image along with a nonlinear transform
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


    def cus_imshow(self, data_to_plot=None, cbar_limit=None, title=None, annotate=False, save_name=None):
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

        print 'Temperature range: {0} ~ {1}\n'.format(np.min(data_to_plot), np.max(data_to_plot))

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

        plt.title('{0}'.format(title))
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.65])
        fig.colorbar(im, cax=cax, orientation='vertical')

        if save_name is None:
            plt.draw()
        else:
            plt.savefig('../figs/{0}.png'.format(save_name), bbox_inches='tight')
            plt.clf()
            plt.close()


    def play_thermal_video(self, sensor_id=None, data_option=('raw_data'), colorbar_limits=None,
                    t_start=None, t_end=None, speed=1):
        """
        This function plays the heat map video
        :param sensor_id: the key in self.PIR
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param colorbar_limits: a list of tuples corresponding to the data options
        :param speed: Speed of video. 1: realtime; 2: 2x faster; 0.5: 2x slower
        :param data_option: 'raw_data', 'cleaned data'
        :return: A video plotting
        """
        num_plots = len(data_option)

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
            num_frames, num_rows, num_cols = self.PIR[sensor_id][data_option[0]].shape
            T_init = np.zeros((num_rows, num_cols)) + colorbar_limits[0][0]
            image = ax.imshow(T_init, cmap=plt.get_cmap('jet'),
                                interpolation='nearest', vmin=colorbar_limits[0][0], vmax=colorbar_limits[0][1])

            # add some initial figure properties
            # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            # fig.colorbar(image, cax=cax)
            ax.set_title('{0}'.format(data_option[0]))
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
                num_frames, num_rows, num_cols = self.PIR[sensor_id][data_option[i]].shape

                T_init = np.zeros((num_rows, num_cols)) + colorbar_limits[i][0]
                image.append(ax[i].imshow(T_init, cmap=plt.get_cmap('jet'),
                                    interpolation='nearest', vmin=colorbar_limits[i][0], vmax=colorbar_limits[i][1]) )

                # add some initial figure properties
                # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
                # fig.colorbar(image[i], cax=cax)
                ax[i].set_title('{0}'.format(data_option[i]))
                # ax.set_xlabel('Column')
                # ax.set_ylabel('Row')

        plt.show(False)
        plt.draw()

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start=t_start, t_end=t_end)

        # play all frames in the data options
        if num_plots == 1:
            t0 = None
            for t in range(index_start, index_end-1):
                t1 = time.time()
                if t0 is not None:
                    print('fps: {0}'.format(1/(t1-t0)))
                t0 = t1

                image.set_data(self.PIR[sensor_id][data_option[0]][t, :, :])
                fig.canvas.restore_region(background)
                ax.draw_artist(image)
                fig.canvas.blit(ax.bbox)

        else:
            t0 = None
            num_data_options = len(data_option)
            for t in range(index_start, index_end-1):
                t1 = time.time()
                if t0 is not None:
                    print('fps: {0}'.format(1/(t1-t0)))
                t0 = t1

                for i in range(0, num_data_options):
                    image[i].set_data(self.PIR[sensor_id][data_option[i]][t, :, :])
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


    def save_thermal_to_avi(self, dataset=None, data_key='norm_temp_data', fps=64):
        """
        This function saves the selected data in avi file
        :param dataset: the sensor id
        :param data_key: 'raw_data' or 'temp_data'
        :return:
        """
        scale  = 64

        fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
        out = cv2.VideoWriter( dataset + ".avi", fourcc, fps, (16*scale,4*scale))
        pir_cam = []
        for frame in self.PIR[dataset][data_key]:
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


    def plot_noise_evolution(self, dataset=None, data_key=None, p_outlier=0.05, stop_thres=(0.5, 0.1),
                             pixel=None, window_s=120, step_s=30, fps=128):
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
        start_frame = fps*window_s

        num_frames= len(self.PIR[dataset]['time'])

        print start_frame
        print num_frames
        print d_frame

        for i in range(start_frame, num_frames, d_frame):
            _, _, _mu, _sigma = self.get_noise_distribution(dataset=dataset, data_key=data_key,
                                                             t_start= self.PIR[dataset]['time'][i]-dt_w,
                                                             t_end=self.PIR[dataset]['time'][i],
                                                             p_outlier=p_outlier,stop_thres=stop_thres,
                                                             pixels=[pixel], suppress_output=True)
            # see from the center of the window
            times.append(self.PIR[dataset]['time'][i] - timedelta(seconds=window_s/2.0))
            mu.append(_mu[pixel])
            sigma.append(_sigma[pixel])

            self.print_loop_status('Evolving noise ', i, num_frames)
        print('\n')
        print('mu: {0}~{1}'.format(np.nanmin(mu), np.nanmax(mu)))
        print('Average sigma: {0}'.format(np.nanmean(sigma)))

        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        # plot the time series
        ax = self.plot_time_series_for_pixel(dataset=dataset, data_key=data_key, t_start=None, t_end=None, pixels=[pixel])

        ax.plot(times, mu, color='r', linewidth=2)
        ax.plot(times, mu+sigma, color='r', linestyle='--', linewidth=2)
        ax.plot(times, mu-sigma, color='r', linestyle='--',linewidth=2)

        plt.draw()