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
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from datetime import datetime
from os.path import exists
import sys
import time



class TrafficData:
    """
    This class is a universal class which deals with different types of data specified in the data collection manual.
    """

    def __init__(self):

        # There may be multiple PIR data formats: 1x16, 4x48.... Use key-value store
        # PIR['1x16'] = {'time':narray, 'Ta':narray, 'raw_data':narray, 'cleaned_data':narray}
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


    def read_data_file(self, file_name_str=None):
        """
        This function reads and parses the file according to the data format specified in data collection manual
        :param file_name_str: string, the file name string: e.g. dataset_20151204_145234.csv
        :return: 0 if correctly read. Data are saved in key-value stores
        """

        if not exists(file_name_str):
            print 'Error: file {0} does not exists'.format(file_name_str)
            return 1

        f = open(file_name_str, 'r')

        for line in f:
            self.line_parser(line)

        # change all arrays into np.array for easy access
        # print out the dataset information
        print 'Finished reading dataset {0}'.format(file_name_str)
        print '--PIR dataset:'
        for pir_mxn in self.PIR.keys():
            for key in self.PIR[pir_mxn].keys():
                self.PIR[pir_mxn][key] = np.array(self.PIR[pir_mxn][key])
            print '----{0}: {1} samples from {2} to {3}'.format(pir_mxn,
                                                                len(self.PIR[pir_mxn]['time']),
                                                                self.PIR[pir_mxn]['time'][0].time(),
                                                                self.PIR[pir_mxn]['time'][-1].time())
        for key in self.USON.keys():
            self.USON[key] = np.array(self.USON[key])
        print '--USON dataset: {0} sampels from {1} to {2}'.format(len(self.USON['time']),
                                                                   self.USON['time'][0].time(),
                                                                   self.USON['time'][-1].time())
        print '--IMU dataset:'
        for key in self.IMU.keys():
            self.IMU[key] = np.array(self.IMU[key])
        print '----mag: {0} sampels from {1} to {2}'.format(len(self.IMU['mag'][0]),
                                                                   self.IMU['time'][0].time(),
                                                                   self.IMU['time'][-1].time())
        print '----accel: {0} sampels from {1} to {2}'.format(len(self.IMU['accel'][0]),
                                                                   self.IMU['time'][0].time(),
                                                                   self.IMU['time'][-1].time())
        print '----gyro: {0} sampels from {1} to {2}'.format(len(self.IMU['gyro'][0]),
                                                                   self.IMU['time'][0].time(),
                                                                   self.IMU['time'][-1].time())
        for key in self.LABEL.keys():
            self.LABEL[key] = np.array(self.LABEL[key])
        print '--LABEL: {0} labels from {1} to {2}\n'.format(len(self.LABEL['time']),
                                                                   self.LABEL['time'][0].time(),
                                                                   self.LABEL['time'][-1].time())

        f.close()

    def line_parser(self, line=None):
        """
        This function parses each line of data and save in corresponding class property. Need to be updated if new data
        is in a different format.
        :param line: The line in the data file;
        :return: 0 if correctly parsed.
        """
        line = line.strip()
        items = line.split(',')

        # Each row is a data slice. Parse according to data collection manual
        if 'invalid_read' in line:
            return 0

        elif 'PIR_1x16_2nd' in line:
            # check if the data set created
            if 'pir_1x16_2nd' not in self.PIR.keys():
                self.PIR['pir_1x16_2nd'] = {'time':[], 'Ta':[], 'raw_data':[]}
                for i in range(0,16):
                    self.PIR['pir_1x16_2nd']['raw_data'].append([])
            else:
                self.PIR['pir_1x16_2nd']['time'].append( self.parse_time(items[1]) )
                self.PIR['pir_1x16_2nd']['Ta'].append( float(items[2]) )

                for i in range(0, 16):
                    self.PIR['pir_1x16_2nd']['raw_data'][i].append( float(items[i+3]) )

        elif 'PIR_2x16' in line:
            # check if the data set created
            if 'pir_2x16' not in self.PIR.keys():
                self.PIR['pir_2x16'] = {'time':[], 'Ta':[], 'raw_data':[]}
                for i in range(0,32):
                    self.PIR['pir_2x16']['raw_data'].append([])
            else:
                self.PIR['pir_2x16']['time'].append( self.parse_time(items[1]) )
                self.PIR['pir_2x16']['Ta'].append( float(items[2]) )

                for i in range(0, 32):
                    self.PIR['pir_2x16']['raw_data'][i].append( float(items[i+3]) )

        elif 'PIR_4x48' in line:
            # check if the data set created
            if 'pir_4x48' not in self.PIR.keys():
                self.PIR['pir_4x48'] = {'time':[], 'Ta':[], 'raw_data':[]}
                for i in range(0,192):
                    self.PIR['pir_4x48']['raw_data'].append([])
            else:
                self.PIR['pir_4x48']['time'].append( self.parse_time(items[1]) )

                self.PIR['pir_4x48']['Ta'].append( float(items[2]) )
                for i in range(0, 192):
                    self.PIR['pir_4x48']['raw_data'][i].append( float(items[i+3]) )

        elif 'USON' in line:
            if len(self.USON.keys()) == 0:
                self.USON = {'time':[],'distance':[]}
            else:
                self.USON['time'].append( self.parse_time(items[1]) )
                self.USON['distance'].append(float(items[2]))

        elif 'IMU' in line:
            if len(self.IMU.keys()) == 0:
                self.IMU = {'time':[], 'mag':[], 'accel':[], 'gyro':[]}
                for i in range(0,3):
                    self.IMU['mag'].append([])
                    self.IMU['accel'].append([])
                    self.IMU['gyro'].append([])
            else:
                self.IMU['time'].append(self.parse_time(items[1]))
                for i in range(0,3):
                    self.IMU['mag'][i].append(float(items[i+2]))
                    self.IMU['accel'][i].append(float(items[i+5]))
                    self.IMU['gyro'][i].append(float(items[i+8]))

        elif 'LABEL' in line:
            if len(self.LABEL.keys()) == 0:
                self.LABEL = {'time':[],'count':[], 'speed':[]}
            else:
                self.LABEL['time'].append(self.parse_time(items[1]))
                self.LABEL['count'].append(float(items[2]))
                self.LABEL['speed'].append(float(items[3]))

        elif len(line) == 1:
            # This may simply be a blank line
            return 0

        else:
            print 'Error: unrecognized data format.'
            return -1

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


    def subtract_PIR_background(self, background_duration=100, save_in_file_name=None):
        """
        This function subtracts the background of the PIR sensor data; the background is defined as the median of the
        past background_duration samples
        :param background_duration: int, the number of samples regarded as the background
        :param save_in_file_name: string, the file name which saves the cleaned data; Put None if not save
        :return: data saved in self.pir_data_background_removed
        """

        # For each pir dataset: 1x16 or 4x48
        for pir_mxn in self.PIR.keys():

            # row is number of pixels, and col is the number of samples
            row, col = self.PIR[pir_mxn]['raw_data'].shape

            # initialize the cleaned data
            self.PIR[pir_mxn]['cleaned_data'] = []
            for i in range(0, row):
                self.PIR[pir_mxn]['cleaned_data'].append([])

            for pixel in range(0,row):

                for sample in range(0, col):

                    # if history is shorter than background_duration, simply copy
                    # otherwise subtract background
                    if sample < background_duration:
                        self.PIR[pir_mxn]['cleaned_data'][pixel].append(self.PIR[pir_mxn]['raw_data'][pixel][sample])
                    else:
                        self.PIR[pir_mxn]['cleaned_data'][pixel].append(
                            self.PIR[pir_mxn]['raw_data'][pixel][sample] -
                            np.median(self.PIR[pir_mxn]['raw_data'][pixel][sample-background_duration:sample] )
                        )

            self.PIR[pir_mxn]['cleaned_data'] = np.array(self.PIR[pir_mxn]['cleaned_data'])

        print 'Background subtracted in PIR data, saved in self.PIR["cleaned_data"]\n'

        # save cleaned data in file in the same format specified in Data Collection Manual
        if save_in_file_name is not None:
            self.save_data_in_file(save_in_file_name)

    def save_data_in_file(self, file_name_str='cleaned_data.csv'):
        """
        Saves the PIR, IMU, USON, LABEL data from the key-value store to files in Data Collection Manual format
        :param file_name_str: the file name to save in
        :return: 0 if successful
        """

        f = open(file_name_str, 'w')

        # For all the PIR data
        for pir_mxn in self.PIR.keys():

            for sample in range(0, len(self.PIR[pir_mxn]['time'])):

                f.write('PIR_{0},{1},{2},{3}\n'.format(pir_mxn.split('_')[1],
                                                       self.time_to_string( self.PIR[pir_mxn]['time'][sample] ),
                                                       self.PIR[pir_mxn]['Ta'][sample],
                                                       ','.join( str(i) for i in self.PIR[pir_mxn]['cleaned_data'][:,sample])))

        # For all the IMU data
        for sample in range(0, len(self.IMU['time'])):

            f.write('IMU,{0},{1},{2},{3}\n'.format(self.time_to_string( self.IMU['time'][sample] ),
                                                   ','.join( str(i) for i in self.IMU['mag'][:,sample]),
                                                   ','.join( str(i) for i in self.IMU['accel'][:,sample]),
                                                   ','.join( str(i) for i in self.IMU['gyro'][:,sample]) ) )
        # For all the Ultrasonic data
        for sample in range(0, len(self.USON['time'])):
            f.write('USON,{0},{1}\n'.format( self.time_to_string( self.USON['time'][sample] ),
                                           self.USON['distance'][sample] ) )

        # For all the labled data
        for sample in range(0, len(self.LABEL['time'])):
            f.write('LABEL,{0},{1},{2}\n'.format( self.time_to_string( self.LABEL['time'][sample] ),
                                                  self.LABEL['count'][sample],
                                                  self.LABEL['speed'][sample]) )




    def calculate_std(self):
        """
        Statistic Analysis:
        The goal of this function is to analyze the PIR data background noise. The data should be measuring background.
        This function calculates the mean and standard deviation of each pixel, to better understand the noise.
        :return: mean (dict) and std (dict), keys are same as the self.PIR
        """

        mean = {}
        std = {}

        for pir_mxn in self.PIR.keys():

            mean[pir_mxn] = []
            std[pir_mxn] = []

            num_pixels, num_samples = self.PIR[pir_mxn]['raw_data'].shape
            for pixel in range(0, num_pixels):

                # For each pixel
                if len(self.PIR[pir_mxn]['raw_data'][pixel]) != 0:
                    mean[pir_mxn].append( np.mean(self.PIR[pir_mxn]['raw_data'][pixel] ) )
                    std[pir_mxn].append( np.std(self.PIR[pir_mxn]['raw_data'][pixel] ) )

            print 'Average Std: {0}'.format(np.mean(std[pir_mxn]))

            # convert the 1xn array in to a matrix which corresponds to the pixel position

            mean[pir_mxn] = np.array(mean[pir_mxn])
            std[pir_mxn] = np.array(std[pir_mxn])

            row, col = pir_mxn.split('_')[1].split('x')
            row = int(row)
            col = int(col)
            mean[pir_mxn] = mean[pir_mxn].reshape((col,row)).T
            std[pir_mxn] = std[pir_mxn].reshape((col,row)).T

        return mean, std

    def plot_histogram_for_pixel(self, t_start_str=None, t_end_str=None, pixel_list=list()):
        """
        Statistic Analysis:
        This function plots the histogram of the raw data for a selected pixel, to better understand the noise
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param pixel_list: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :return: one figure for each pixel
        """

        mu, sigma = self.calculate_std()

        # For each PIR configuration
        for pir_mxn_list in pixel_list:

            pir_mxn = pir_mxn_list[0]
            pixels = pir_mxn_list[1]
            if pir_mxn not in self.PIR.keys():
                print 'Error: incorrect pixel definition.'
                return -1

            # get the size of this pir data set
            row, col = pir_mxn.split('_')[1].split('x')
            row = float(row)
            col = float(col)

            for pixel in pixels:

                pixel_index = (pixel[1]-col)*row + (pixel[0]-1)

                # grab the data
                if t_start_str is None or t_end_str is None:
                    time_series = self.PIR[pir_mxn]['raw_data'][pixel_index, :]
                else:
                    t_start = self.parse_time(t_start_str)
                    t_end = self.parse_time(t_end_str)

                    index_start = self.PIR[pir_mxn]['time'] >= t_start
                    index_end = self.PIR[pir_mxn]['time'] <= t_end
                    index = index_start & index_end

                    time_series = self.PIR[pir_mxn]['raw_data'][pixel_index, index]

                # the histogram of the data
                num_bins = 50
                fig = plt.figure(figsize=(16,8), dpi=100)
                n, bins, patches = plt.hist(time_series, num_bins,
                                            normed=1, facecolor='green', alpha=0.75)

                # Need tuple for indexing. Make sure is tuple
                if not isinstance(pixel, tuple):
                    pixel = tuple(pixel)

                # add a 'best fit' line
                norm_fit_line = mlab.normpdf(bins, mu[pir_mxn][pixel],
                                                   sigma[pir_mxn][pixel])
                l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1)

                plt.xlabel('Temperature ($^{\circ}C$)')
                plt.ylabel('Probability')
                plt.title(r'Histogram of pixel {0} for set {1}: $\mu$= {2}, $\sigma$={3}'.format(pixel,
                                                                                                 pir_mxn,
                                                                                                 mu[pir_mxn][pixel],
                                                                                                 sigma[pir_mxn][pixel]))
            # plt.axis([40, 160, 0, 0.03])
            plt.grid(True)

        plt.draw()

    def plot_time_series_for_pixel(self, t_start_str=None, t_end_str=None,
                                   pixel_list=list(), data_option=None):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param t_start_str: str str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param pixel_list: list of tuples, [('pir_1x16', [(1,1),(1,5)]), ('pir_2x16', [(1,1),(2,1)]) ]
        :param data_option: string, 'raw', 'cleaned'
        :return: a figure with all the pixels time series
        """

        time_series_to_plot = []

        for pir_mxn_list in pixel_list:

            pir_mxn = pir_mxn_list[0]
            pixels = pir_mxn_list[1]
            if pir_mxn not in self.PIR.keys():
                print 'Error: incorrect pixel definition.'
                return -1

            # get the size of this pir data set
            row, col = pir_mxn.split('_')[1].split('x')
            row = float(row)
            col = float(col)

            for pixel in pixels:

                # This pixel is used to get the raw and cleaned data
                pixel_index = (pixel[1]-col)*row + (pixel[0]-1)

                timestamps = self.PIR[pir_mxn]['time']

                # get the data for the corresponding pixel
                if data_option == 'raw':
                    data = self.PIR[pir_mxn]['raw_data'][pixel_index, :]
                elif data_option == 'cleaned':
                    data = self.PIR[pir_mxn]['cleaned_data'][pixel_index, :]
                else:
                    data = []
                    print 'Error: pixel {0} is not recognized'.format(pixel)

                pixel_series = {}
                pixel_series['info'] = pixel
                pixel_series['time'] = timestamps
                pixel_series['data'] = data

                time_series_to_plot.append(pixel_series)

            # call the generic function to plot
            self.plot_time_series(None, time_series_to_plot, t_start_str, t_end_str)

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

    def plot_heat_map_in_period(self, t_start_str=None, t_end_str=None,
                                T_min=None, T_max=None, data_option=None):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each frame 1x16 pixels
                    stacked column by column to 16 x n samples
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :param data_option: string 'raw' or 'cleaned'; 'cleaned' removed the background
        :return: a figure with 16 x n color map for n frame
        """

        for pir_mxn in self.PIR.keys():

            if t_start_str is None or t_end_str is None:

                if data_option == 'raw':
                    temperature_mxn = self.PIR[pir_mxn]['raw_data']
                elif data_option == 'cleaned':
                    temperature_mxn = self.PIR[pir_mxn]['cleaned_data']
                else:
                    print 'Error: data_option not recognized'
                    return

                self.plot_2d_colormap(temperature_mxn, T_min, T_max, 'All data')

            else:
                t_start = self.parse_time(t_start_str)
                t_end = self.parse_time(t_end_str)

                index_start = self.PIR[pir_mxn]['time'] >= t_start
                index_end = self.PIR[pir_mxn]['time'] <= t_end
                index = index_start & index_end

                if data_option == 'raw':
                    temperature_mxn = self.PIR[pir_mxn]['raw_data'][:,index]
                elif data_option == 'cleaned':
                    temperature_mxn = self.PIR[pir_mxn]['cleaned_data'][:,index]
                else:
                    print 'Error: data_option not recognized'
                    return

                temperature_mxn = np.array(temperature_mxn)

                print 'Temperature range: {0} ~ {1}\n'.format(np.min(temperature_mxn), np.max(temperature_mxn))
                self.plot_2d_colormap(temperature_mxn, T_min, T_max, '{0}: {1} to {2}'.format(
                                      t_start.strftime('%Y/%m/%d'),
                                      t_start.strftime('%H:%M:%S.%f'),
                                      t_end.strftime('%H:%M:%S.%f')))

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
            v_max = np.max(data_to_plot)

        # adjust the figure width and height to best represent the data matrix
        # Best figure size (,12):(480,192)
        row,col = data_to_plot.shape
        fig_height = 12
        fig_width = int(0.8*col/row)*12
        if fig_width >= 2000:
            # two wide
            fig_width = 2000

        fig = plt.figure(figsize=( fig_width, fig_height), dpi=100)
        im = plt.imshow(data_to_plot,
                        cmap=plt.get_cmap('jet'),
                        interpolation='nearest',
                        vmin=v_min, vmax=v_max)

        plt.title('{0}'.format(title))
        cax = fig.add_axes([0.02, 0.25, 0.01, 0.4])
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.draw()


    def play_video(self, t_start_str=None, t_end_str=None, T_min=-20, T_max=50, speed=1, data_option='raw_data'):
        """
        This function plays the heat map video
        :param t_start_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param t_end_str: str %Y%m%d_%H:%M:%S_%f e.g. 20151131_15:23:33_123456
        :param T_min: The min temperature for heatmap colorbar
        :param T_max: The max temperature for heatmap colorbar
        :param speed: Speed of video. 1: realtime; 2: 2x faster; 0.5: 2x slower
        :param data_option: 'raw_data', 'cleaned data'
        :return: A video plotting
        """
        # initialize figure
        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 3.5)
        ax.hold(True)

        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

        # The image to be updated
        # Initialize with 4x48 full pixel frame. Will change over the time
        T_init = np.zeros((4, 48)) + T_min
        image = ax.imshow(T_init, cmap=plt.get_cmap('jet'),
                            interpolation='nearest', vmin=T_min, vmax=T_max)
        # add some initial figure properties
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(image, cax=cax)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.show(False)
        plt.draw()

        #  extract data to play by checking the time interval
        datasets_to_play = {}
        # get the order of the dataset in time, assuming datasets are unoverlapping in time
        dataset_order = []
        t_start = self.parse_time(t_start_str)
        t_end = self.parse_time(t_end_str)
        for pir_mxn in self.PIR.keys():

            index_start = self.PIR[pir_mxn]['time'] >= t_start
            index_end = self.PIR[pir_mxn]['time'] <= t_end
            index = index_start & index_end

            # if there is any element that is True
            if any(index):
                if data_option == 'raw_data':
                    datasets_to_play[pir_mxn] = [ self.PIR[pir_mxn]['time'][index],
                                                  self.PIR[pir_mxn]['raw_data'][:, index] ]
                elif data_option == 'cleaned_data':
                    datasets_to_play[pir_mxn] = [ self.PIR[pir_mxn]['time'][index],
                                                  self.PIR[pir_mxn]['cleaned_data'][:, index] ]

                # append the start time, end time, and key as a tuple
                dataset_order.append( (datasets_to_play[pir_mxn][0][0],
                                       datasets_to_play[pir_mxn][0][-1],
                                       pir_mxn) )

        # sort based on the start time or end time. If same then overlapping, otherwise error
        if sorted(dataset_order, key=lambda tup: tup[0]) != sorted(dataset_order, key=lambda tup: tup[1]):
            print 'Error: PIR datasets are overlapping in time. Check data format.'
            return -1
        else:
            dataset_order.sort(key=lambda tup: tup[0])

        # For each PIR data set, plot
        for pir_set in dataset_order:

            pir_mxn = pir_set[2]

            row, col = pir_mxn.split('_')[1].split('x')
            row = float(row)
            col = float(col)

            for frame_index in range(0, datasets_to_play[pir_mxn][1].shape[1]-1):

                # wait using the realtime and speed
                time.sleep( (datasets_to_play[pir_mxn][0][frame_index+1]-
                             datasets_to_play[pir_mxn][0][frame_index]).total_seconds()*1000/speed )

                # update figure
                # print 'T[{0}]: {1}'.format(i, T[i])
                image.set_data(datasets_to_play[pir_mxn][1][:, frame_index].reshape(col,row).T)

                fig.canvas.restore_region(background)
                ax.draw_artist(image)
                fig.canvas.blit(ax.bbox)
















