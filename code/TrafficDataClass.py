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

    def load_npy_data(self, file_name_str=None, sensor_id=None, data_type='raw_data'):
        """
        This function loads the data npy file. each entry is [timestamp(datetime), 4x16 pir array, ptat, cp]
        :param file_name_str: a string of the file name,
        :param sensor_id: the name of the data saved in self.PIR
        :param data_type: the data type, 'raw data', 'temperature', 'cleaned_temperature'
        :return:
        """
        if file_name_str is not None and exists(file_name_str):
            data = np.load(file_name_str)

            # check if read other types of data previously
            if sensor_id not in self.PIR.keys():
                self.PIR[sensor_id] = OrderedDict()

            if 'time' not in self.PIR[sensor_id].keys():
                # get the time stamp
                self.PIR[sensor_id]['time'] = []
                for entry in data:
                    self.PIR[sensor_id]['time'].append(entry[0])

                self.PIR[sensor_id]['time'] = np.array(self.PIR[sensor_id]['time'])

            if data_type not in self.PIR[sensor_id].keys():
                # get the data
                self.PIR[sensor_id][data_type] = []
                for entry in data:
                    self.PIR[sensor_id][data_type].append(entry[1])
                self.PIR[sensor_id][data_type] = np.array(self.PIR[sensor_id][data_type])

        else:
            print('Warning: no data was read')

    def parse_time(self, datetime_str=None):
        """
        This funciton parses the datestr in format %Y%m%d_%H:%M:%S_%f
        :param datetime_str: The datetime string, %Y%m%d_%H:%M:%S_%f
        :return: datetime type
        """
        if datetime_str is None:
            print 'Error: invalid date time string for parse_time'
            return None
        else:
            return datetime.strptime(datetime_str, "%Y%m%d_%H:%M:%S_%f")

    def time_to_string(self, dt):
        """
        This function returns a string in format %Y%m%d_%H:%M:%S_%f
        :param dt: datetime type
        :return: str
        """
        return dt.strftime("%Y%m%d_%H:%M:%S_%f")

    def subtract_PIR_background(self, pir_mxn, background_duration=100, save_in_file_name=None, data_type='raw_data'):
        """
        This function subtracts the background of the PIR sensor data; the background is defined as the median of the
        past background_duration samples
        :param background_duration: int, the number of samples regarded as the background
        :param save_in_file_name: string, the file name which saves the cleaned data; Put None if not save
        :param data_type: string, subtract background for raw data or temperature data
        :return: data saved in self.pir_data_background_removed
        """

        # row is number of pixels, and col is the number of samples
        frame, row, col = self.PIR[pir_mxn][data_type].shape

        # initialize the cleaned data
        cleaned_data = 'cleaned_' + data_type
        self.PIR[pir_mxn][cleaned_data] = []
        for i in range(0, frame):
            self.PIR[pir_mxn][cleaned_data].append(np.zeros((row,col)))

        for f in range(0, frame):

            for pixel_row in range(0,row):
                for pixel_col in range(0, col):

                    # if history is shorter than background_duration, simply copy
                    # otherwise subtract background
                    if f < background_duration:
                        self.PIR[pir_mxn][cleaned_data][f][pixel_row, pixel_col] = \
                            self.PIR[pir_mxn][data_type][f, pixel_row, pixel_col]
                    else:
                        self.PIR[pir_mxn][cleaned_data][f][pixel_row, pixel_col] = \
                            self.PIR[pir_mxn][data_type][f, pixel_row, pixel_col] - \
                            stats.nanmedian(self.PIR[pir_mxn][data_type][f-background_duration:f, pixel_row, pixel_col] )


            self.PIR[pir_mxn][cleaned_data] = np.array(self.PIR[pir_mxn][cleaned_data])

        print 'Background subtracted in PIR data, saved in self.PIR[{0}][{1}]\n'.format(pir_mxn, cleaned_data)

    def normalize_data(self, temp_data_files=list(), periods=None, skip_dt=1):
        """
        This function normalize the temperature PIR frames and save into a
        :param temp_data_files: the data file list
        :param periods: the periods of starting times
        :param skip_dt: seconds, skip the first a few seconds where data is nan
        :return: save in files.
        """

        dt = timedelta(seconds=skip_dt)

        for f in temp_data_files:

            # get the name and load data
            sensor_id = f.strip().split('/')[-1].replace('.npy', '')

            self.load_npy_data(file_name_str=f, sensor_id=sensor_id, data_type='temp_data')

            # compute the mean and std of the sensor
            mu, std, noise_mu, noise_std = self.calculate_std(sensor_id=sensor_id, data_type='temp_data',
                                                              t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
                                                              t_end_str= self.time_to_string(periods[sensor_id][1]) )

            self.PIR[sensor_id]['norm_temp_data'] = (self.PIR[sensor_id]['temp_data'] - noise_mu)/noise_std*32

            self.PIR[sensor_id]['norm_temp_data'][ self.PIR[sensor_id]['norm_temp_data'] > 255 ] = 255
            self.PIR[sensor_id]['norm_temp_data'][ self.PIR[sensor_id]['norm_temp_data'] < 0 ] = 0

            # visualize the data
            # self.plot_histogram_for_pixel(t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
            #                               t_end_str= self.time_to_string( periods[sensor_id][1]),
            #                           pixel_list=[(sensor_id, [(0,4), (1,8), (2, 12), (3, 15)])], data_type='norm_temp_data')

            # self.plot_heat_map_in_period(sensor_id=sensor_id, data_type='norm_temp_data',
            #                              t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
            #                              t_end_str= self.time_to_string(periods[sensor_id][1]),
            #                              T_min=1, T_max=15, option='vec')
            # self.plot_heat_map_in_period(sensor_id=sensor_id, data_type='norm_temp_data',
            #                              t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
            #                              t_end_str= self.time_to_string(periods[sensor_id][1]),
            #                              T_min=1, T_max=None, option='max')
            # self.plot_heat_map_in_period(sensor_id=sensor_id, data_type='norm_temp_data',
            #                              t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
            #                              t_end_str= self.time_to_string(periods[sensor_id][1]),
            #                              T_min=1, T_max=8, option='mean')
            # self.plot_heat_map_in_period(sensor_id=sensor_id, data_type='norm_temp_data',
            #                              t_start_str= self.time_to_string(periods[sensor_id][0] + dt),
            #                              t_end_str= self.time_to_string(periods[sensor_id][1]),
            #                              T_min=1, T_max=None, option='tworow')


    def calculate_std(self, sensor_id=None, data_type=None, t_start_str=None, t_end_str=None):
        """
        Statistic Analysis:
        The goal of this function is to analyze the PIR data background noise. The data should be measuring background.
        This function calculates the mean and standard deviation of each pixel, to better understand the noise.
        :param sensor_id: the sensor id in self.PIR
        :param data_type: the data type
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :return: mean (dict) and std (dict), keys are same as the self.PIR
        """

        num_frames, num_rows, num_cols = self.PIR[sensor_id][data_type].shape

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start_str=t_start_str, t_end_str=t_end_str)

        mean = np.zeros((num_rows, num_cols))
        std = np.zeros((num_rows, num_cols))

        noise_mean = np.zeros((num_rows, num_cols))
        noise_std = np.zeros((num_rows, num_cols))

        for row in range(0, num_rows):
            for col in range(0, num_cols):

                time_series = self.PIR[sensor_id][data_type][index_start:index_end, row, col]
                mean[row, col] = np.nanmean( time_series )
                std[row, col] = np.nanstd( time_series )

                # compute the mean and std of the noise
                p = mlab.normpdf(time_series, mean[row, col], std[row, col])

                noise_mean[row, col] = np.nanmean( time_series[ p>=0.05] )
                noise_std[row, col] = np.nanstd( time_series[ p>=0.05] )

        return mean, std, noise_mean, noise_std


    def plot_histogram_for_pixel(self, t_start_str=None, t_end_str=None, pixel_list=list(), data_type=None):
        """
        Statistic Analysis:
        This function plots the histogram of the raw data for a selected pixel, to better understand the noise
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param pixel_list: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :param data_type: the options for the data, 'raw_data', 'temp_data',...
        :return: one figure for each pixel
        """

        # For each PIR configuration
        for pir_mxn_list in pixel_list:

            sensor_id = pir_mxn_list[0]
            pixels = pir_mxn_list[1]
            if sensor_id not in self.PIR.keys():
                raise Exception('Incorrect sensor id')

            # compute the mean and std
            mu, sigma, noise_mu, noise_sigma = self.calculate_std(sensor_id=sensor_id, data_type=data_type,
                                                                  t_start_str=t_start_str, t_end_str=t_end_str)

            # find the data in the time interval
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start_str=t_start_str, t_end_str=t_end_str)

            for pixel in pixels:

                print('{0}: overall {1}, noise {2}'.format(pixel, (mu[pixel], sigma[pixel]),
                                                           (noise_mu[pixel], noise_sigma[pixel])))

                time_series = self.PIR[sensor_id][data_type][index_start:index_end, pixel[0], pixel[1]]

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
                plt.title(r'{0} pixel {1}; $\mu$= {2:.2f}, $\sigma$={3:.2f}'.format(sensor_id, pixel,
                                                                                                 noise_mu[pixel],
                                                                                                 noise_sigma[pixel]))
                plt.grid(True)

        plt.draw()


    def plot_time_series_for_pixel(self, t_start_str=None, t_end_str=None,
                                   pixel_list=list(), data_type=None):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param t_start_str: str str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param pixel_list: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :param data_type: string, option for data, 'raw_data' or 'temp_data'
        :return: a figure with all the pixels time series
        """

        fig = plt.figure(figsize=(16,8), dpi=100)
        ax = fig.add_subplot(111)

        for pir_mxn_list in pixel_list:

            sensor_id = pir_mxn_list[0]
            pixels = pir_mxn_list[1]
            if sensor_id not in self.PIR.keys():
                print 'Error: incorrect pixel definition.'
                return -1

            # find the data in the time interval
            index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start_str=t_start_str, t_end_str=t_end_str)

            for pixel in pixels:

                timestamps = self.PIR[sensor_id]['time'][index_start:index_end]
                time_series = self.PIR[sensor_id][data_type][index_start:index_end, pixel[0], pixel[1]]

                print('length of data to plot:{0}'.format(len(timestamps)))

                plt.plot(timestamps, time_series, label='{0} pixel {1}'.format(sensor_id, pixel))

        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.title('Time series')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time')

        plt.legend()
        plt.grid(True)
        plt.draw()


    def plot_time_series(self, fig_handle=None, time_series_list=list(),
                         t_start_str=None, t_end_str=None):
        """
        Visualization:
        This function plots all the time series in the time_series_list, which offers more flexibility.
        :param fig_handle: fig or ax handle to plot on; create new figure if None
        :param time_series_list: list, [time_series_1, time_series_2,...];
                                 time_series_n: dict; time_series_n['info'] = string or label in figure
                                                      time_series_n['time'] = list of float
                                                      time_series_n['data'] = list of float
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :return: a (new) figure with all time series
        """
        # plot in new figure window if not specified
        if fig_handle is None:
            fig = plt.figure(figsize=(16,8), dpi=100)
            ax = fig.add_subplot(111)
        else:
            ax = fig_handle.subplot(111)

        for time_series in time_series_list:

            if t_start_str is None or t_end_str is None:
                # if not specified, then plot all data
                time_to_plot = time_series['time']
                data_to_plot = time_series['data']
            else:
                t_start = self.parse_time(t_start_str)
                t_end = self.parse_time(t_end_str)

                index_start = time_series['time'] >= t_start
                index_end = time_series['time'] <= t_end
                index = index_start & index_end

                time_to_plot = time_series['time'][index]
                data_to_plot = time_series['data'][index]

            plt.plot(time_to_plot, data_to_plot, label=time_series['info'])

        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.title('Time series')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time')

        plt.legend()
        plt.grid(True)
        plt.draw()


    def plot_heat_map_in_period(self, sensor_id=None, data_type=None, t_start_str=None, t_end_str=None,
                                T_min=None, T_max=None, option='vec'):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each vec(frame)
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :param data_option: string 'raw' or 'cleaned'; 'cleaned' removed the background
        :param option: options for stack each column: 'vec', 'mean', 'max', 'tworow'
        :return: a figure with 16 x n color map for n frame
        """

        # find the data in the time interval
        index_start, index_end = self.get_index_in_period(timestamps=self.PIR[sensor_id]['time'],
                                                          t_start_str=t_start_str, t_end_str=t_end_str)

        # reshape the data by vec(frame)
        num_frames, num_rows, num_cols = self.PIR[sensor_id][data_type].shape
        map = []
        if option == 'vec':
            for t in range(index_start, index_end):
                map.append( self.PIR[sensor_id][data_type][t].T.reshape(1, num_rows*num_cols).squeeze() )
        elif option == 'tworow':
            for t in range(index_start, index_end):
                map.append( self.PIR[sensor_id][data_type][t][1:3,:].T.reshape(1, num_rows*num_cols/2).squeeze() )
        elif option == 'max':
            for t in range(index_start, index_end):
                map.append( np.max(self.PIR[sensor_id][data_type][t], 0) )
        elif option == 'mean':
            for t in range(index_start, index_end):
                map.append( np.mean(self.PIR[sensor_id][data_type][t], 0) )

        map = np.array(map).T
        self.plot_2d_colormap(map, T_min, T_max, '{0} to {1}'.format(
                                  t_start_str, t_end_str))


    def plot_2d_colormap(self, data_to_plot=None, v_min=None, v_max=None, title=None):
        """
        Visualization:
        This function is a universal function for plotting a 2d color map.
        :param data_to_plot: a 2D float array with values to be plotted
        :param v_min: the min value for the color bar; if None, will use min(data_to_plot)
        :param v_max: the max value for the color bar; if None, will use max(data_to_plot)
        :param title: string, the title for the figure
        :return: a 2d colormap figure
        """

        if v_min is None:
            v_min = np.min(data_to_plot)
        if v_max is None:
            v_max = np.max(data_to_plot)/2

        print 'Temperature range: {0} ~ {1}\n'.format(np.min(data_to_plot), np.max(data_to_plot))

        # adjust the figure width and height to best represent the data matrix
        # Best figure size (,12):(480,192)
        row,col = data_to_plot.shape
        fig_height = 8
        fig_width = 15

        # fig_width = int(0.8*col/row)*12
        # if fig_width >= 2000:
        #     # two wide
        #     fig_width = 2000

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('auto')
        # pos1 = ax.get_position() # get the original position
        # pos2 = [pos1.x0 - 0.12, pos1.y0 ,  pos1.width*1.272, pos1.height*1.25]
        # ax.set_position(pos2)

        im = plt.imshow(data_to_plot,
                        cmap=plt.get_cmap('jet'),
                        interpolation='nearest', aspect='auto',
                        vmin=v_min, vmax=v_max)

        plt.title('{0}'.format(title))
        # cax = fig.add_axes([0.02, 0.25, 0.01, 0.4])
        # fig.colorbar(im, cax=cax, orientation='vertical')
        plt.draw()


    def play_video(self, sensor_id=None, data_option=('raw_data'), colorbar_limits=None,
                    t_start_str=None, t_end_str=None, speed=1):
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
                                                          t_start_str=t_start_str, t_end_str=t_end_str)

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


    def get_index_in_period(self, timestamps=None, t_start_str='20151131_15:23:33_123456',
                                                    t_end_str='20151131_15:23:34_123456'):
        """
        This function returns the start and end index of entries between t_start and t_end.
        :param timestamps: a list of timestamps in datetime format
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :return: index_start, index_end             timestamps[index_start:index_end] gives the data in period
        """
        if t_start_str is None:
            index_start = 0
        else:
            index_start = bisect.bisect( timestamps, self.parse_time(t_start_str) )

        if t_end_str is None:
            index_end = len(timestamps)
        else:
            index_end = bisect.bisect( timestamps, self.parse_time(t_end_str) )


        return index_start, index_end


    def get_data_periods(self, dir):
        """
        This function returns the periods for all data collection experiments
        :param dir: the directory
        :return: save in file and return the periods
        """
        periods = OrderedDict()
        f_periods = dir.replace('*.npy', '')+'dataset_periods.txt'

        # load previously extracted file if exists
        if exists(f_periods):
            print('Loading dataset_periods.txt ...')
            with open(f_periods,'r') as fi:
                for line in fi:
                    items = line.strip().split(',')
                    periods[items[0]] = ( self.parse_time(items[1]), self.parse_time(items[2]) )

        else:
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

        return periods


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


    def save_as_avi(self, sensor_id=None, data_type='norm_temp_data', fps=64):
        """
        This function saves the selected data in avi file
        :param sensor_id: the sensor id
        :param data_type: 'raw_data' or 'temp_data'
        :return:
        """
        scale  = 64

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter( sensor_id + ".avi",fourcc, fps, (16*scale,4*scale))
        pir_cam = []
        for frame in self.PIR[sensor_id][data_type]:
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












