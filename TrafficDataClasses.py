
"""
The classes interface with the data files. It reads, pre-processes, and visualizes the data.
Since we are using different configurations of the sensor, multiple classes are created here with each class for
each data configuration. Add a new class based on old ones if using a new data format. It should require only minimum
amount of modification.

The structure of each class:
# - Visualization:
# - Statistic analysis:

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
from datetime import datetime
from os.path import exists
import sys
import time


class TrafficData_4x48:
    """
    This class deals with the ultrasonic, 3 PIR in total 4x48 pixels, and IMU data
    """

    def __init__(self):

        # raw data for ultrasonic, three PIRs, and IMU
        # 3 x n samples since the time stamps for each PIR sensor is slightly different, in seconds
        self.pir_timestamps = []
        for i in range(0,3):
            self.pir_timestamps.append([])
        # 192 x n samples. stacked column by column from sensor 1 to sensor 3
        self.pir_raw_data = []
        for i in range(0,192):
            self.pir_raw_data.append([])

        # 1 x n samples
        self.ultra_timestamps = None
        # 1 x n samples
        self.ultra_raw_data = []

        # 1 x n samples:
        self.imu_timestamps = None
        # 9 x n samples; m is the dimension of data (mag, accel, gyro)
        self.imu_data = []
        for i in range(0,9):
            self.imu_data.append([])

        # data with background subtracted ultrasonic, three PIRs, and IMU
        self.pir_data_background_removed = None

    def read_data_file(self, file_name_str):
        """
        This function reads and parses the file into raw data; it should be updated for new data format
        :param file_name_str: string, the file name string
                            Two types of files can be parsed; format specified by file names
                            PIR_~_4x48.csv: only PIR data with timestamps being the milliseconds; each row:
                                ms_1, ms_2, ms_3, ptat_1, ptat_2, ptat_3, pir1_pixel00, pir1_pixel10, ... pir2_pixel00..
                            ALL_~.csv: all IMU and PIR data; time format: HH_MM_SS_MS;
                                timestamp,'IMU',magx,magy,magz,accelx,accely,accelz,ultra
                                timestamp,'PIR',ms_1,ms_2,ms_3,ptat_1,ptat_2,ptat_3, pir1_pixel00, pir1_pixel10, ...
        :return: saved data in class
        """

        for file_str in file_name_str:

            if not exists(file_str):
                print 'Error: file {0} does not exists'.format(file_str)
                return 1

            f = open(file_str, 'r')

            # identify the file format
            file_str = file_str.strip()
            file_str_items = file_str.split('/')
            file_name_items = file_str_items[-1].split('_')
            if file_name_items[0] == 'PIR':
                file_format = 'PIR'
            elif file_name_items[0] == 'ALL':
                file_format = 'ALL'
            else:
                print 'Error: Could not recognize data format for file {0}\n Name should start with PIR or ALL'.format(file_str)
                return 1

            for line in f:
                self.line_parser(line, file_format)

            # change to numpy for easier access
            self.pir_timestamps = np.array(self.pir_timestamps)

            # make sure the timestamp is monotonically increasing
            # millis counter resets when reaches 2^16-1 = 65535, keep track of how many times it reset
            num_reset = 0
            # update for each pir timestamp
            for pir in range(0, 3):

                for i in range(1, len(self.pir_timestamps[pir])):

                    if self.pir_timestamps[pir][i] < self.pir_timestamps[pir][i-1]:
                        # update number of reset
                        if self.pir_timestamps[pir][i] + num_reset*65535 < self.pir_timestamps[pir][i-1]:
                            num_reset += 1
                        # update the current time stamp
                        self.pir_timestamps[pir][i] += num_reset*65535

            # subtract the first time stamp offset and convert to seconds
            for i in range(0,3):
                self.pir_timestamps[i, :] = self.pir_timestamps[i, :] - self.pir_timestamps[i, 0]
                # self.pir_timestamps[i, :] = self.pir_timestamps[i, :]/1000.0

            self.pir_raw_data = np.array(self.pir_raw_data)

            print '\n 8 Hz, size of time {0}; size of data {1}\n'.format(self.pir_timestamps.shape,
                                                                   self.pir_raw_data.shape)

            f.close()

    def line_parser(self, line=None, parser_option='ALL'):
        """
        This function parses each line of data and save in corresponding class property. Need to be updated if new data
        is in a different format.
        :param line: The line in the data file;
        :param parser_option: 'PIR', 'ALL'; different data format
        :return: save data self.pir_raw_data, self.ultra_raw_data, self.imu_data
        """
        line = line.strip()
        items = line.split(',')

        # file only contains PIR data
        if parser_option == 'PIR':
            if len(items) != 198:
                print 'Error: Data format is incorrect. Update parser!'
                return 1

            self.pir_timestamps[0].append(int(items[0]))
            self.pir_timestamps[1].append(int(items[1]))
            self.pir_timestamps[2].append(int(items[2]))

            for i in range(0, 192):
                    self.pir_raw_data[i].append(float(items[i+6]))

        # file contains both PIR and ultrasonics data
        if parser_option == 'ALL':
            # TODO: parse the timestamp format
            timestamp_str = items[0]

            if items[1] == 'IMU':
                # parse IMU data
                self.imu_timestamps = timestamp_str
                if len(items) == 9:
                    print 'Warning: missing Gyro data'
                    for i in range(0, 6):
                        self.imu_data[i].append(float(items[i+2]))
                elif len(items) == 12:
                    for i in range(0, 9):
                        self.imu_data[i].append(float(items[i+2]))

                # get the ultrasonic sensor data
                self.ultra_timestamps = timestamp_str
                self.ultra_raw_data.append(float(items[-1]))

            elif items[1] == 'PIR':
                # parse PIR data
                if len(items) < 10:
                    # THe full line should be 2+6+64*3 = 200; shorter length means corrupted data, skip
                    return
                else:
                    # use the same timestamp for 3 pir sensors
                    for i in range(0,3):
                        self.pir_timestamps[i].append(timestamp_str)

                    for i in range(0,192):
                        self.pir_raw_data[i].append(float(items[i+8]))

            else:
                print 'Error: incorrect data format.'
                return 1



    def subtract_background(self, background_duration):
        """
        This function subtracts the background of the PIR sensor data; the background is defined as the median of the
        past background_duration samples
        :param background_duration: int, the number of samples regarded as the background
        :return: data saved in self.pir_data_background_removed
        """
        pass

    def calculate_std(self):
        """
        Statistic Analysis:
        This function calculates the mean and standard deviation of each pixel, to better understand the noise.
        :return: print two 4 x 48 matrix: Mean, and STD
        """
        mean = []
        std = []
        for i in range(0, 192):

            if len(self.pir_raw_data[i]) != 0:    # if not empty
                mean.append( np.mean(self.pir_raw_data[i] ) )
                std.append( np.std(self.pir_raw_data[i]) )

                print '8Hz: Pixel {0} with mean {1} and std {2}'.format(i, mean[i], std[i])

        print '8 Hz: Average Std: {0}'.format(np.mean(std))

        # sort the std; remove the large 20% (pixels not well calibrated); then compute the mean std
        std_sorted = sorted(std)
        std_sorted_partial = std_sorted[0:int(len(std_sorted)*0.8)]
        print '8 Hz: Average Std for 80% well calibrated pixels: {0}'.format(np.mean(std_sorted_partial))

        return mean, std

    def plot_histogram_for_pixel(self, pixel_list):
        """
        Statistic Analysis:
        This function plots the histogram of the data for a selected pixel, to better understand the noise
        :param pixel_list: list, [pixel_1, pixel_2...]; pixel_n := (row, col) in ([0,3] [0,47])
        :return: one figure for each pixel
        """

        mu, sigma = self.calculate_std()

        for pixel in pixel_list:

            pixel_index = pixel[1]*4 + pixel[0]

            # grab the data
            time_series = self.pir_raw_data[pixel_index, :]

            # the histogram of the data
            num_bins = 25
            fig = plt.figure(figsize=(16,8), dpi=100)
            n, bins, patches = plt.hist(time_series, num_bins, normed=1, facecolor='green', alpha=0.75)

            # add a 'best fit' line
            norm_fit_line = mlab.normpdf(bins, mu[pixel_index], sigma[pixel_index])
            l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1)

            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Probability')
            plt.title(r'Histogram of pixel {0} at 8 Hz: $\mu$= {1}, $\sigma$={2}'.format(pixel,
                                                                                            mu[pixel_index],
                                                                                            sigma[pixel_index]))
            # plt.axis([40, 160, 0, 0.03])
            plt.grid(True)

        plt.draw()

    def plot_time_series_for_pixel(self, t_start=None, t_end=None, pixel_list=None, data_option=None):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param t_start: float, in seconds; plot all data if None
        :param t_end: float, in seconds, millisecond; plot all data if None
        :param pixel_list: list, [pixel_1, pixel_2...]; pixel_n := (row, col) in ([0,3] [0,47])
        :param data_option: string, 'raw', 'background_removed'
        :return: a figure with all the pixels time series
        """

        time_series_to_plot = []

        for pixel in pixel_list:

            pixel_index = pixel[1]*4 + pixel[0]

            # find out which timestamps we should use
            if pixel[1] <= 15:
                timestramps = self.pir_timestamps[0, :]
            elif 16 <= pixel[1] <= 31:
                timestramps = self.pir_timestamps[1, :]
            elif 32 <= pixel[1] <= 47:
                timestramps = self.pir_timestamps[2, :]
            else:
                print 'Error: pixel {0} is not recognized'.format(pixel)
                timestramps = []

            # get the data for the corresponding pixel
            if data_option == 'raw':
                data = self.pir_raw_data[pixel_index, :]
            elif data_option == 'background_removed':
                data = self.pir_data_background_removed[pixel_index, :]
            else:
                data = []
                print 'Error: pixel {0} is not recognized'.format(pixel)

            pixel_series = {}
            pixel_series['info'] = pixel
            pixel_series['time'] = timestramps
            pixel_series['data'] = data

            time_series_to_plot.append(pixel_series)

            # call the generic function to plot
            self.plot_time_series(None, time_series_to_plot, t_start, t_end)

    def plot_time_series_for_ultra(self, t_start, t_end):
        """
        Visualization:
        This function plots the time series for the ultrasonics sensor data from t_start to t_end
        :param t_start: format to be defined; plot all data if None
        :param t_end: format to be defined; plot all data if None
        :return: a figure with the ultrasonic data plotted
        """
        pass

    def plot_time_series(self, fig_handle, time_series_list, t_start, t_end):
        """
        Visualization:
        This function plots all the time series in the time_series_list, which offers more flexibility.
        :param fig_handle: fig or ax handle to plot on; create new figure if None
        :param time_series_list: list, [time_series_1, time_series_2,...];
                                 time_series_n: dict; time_series_n['info'] = string or label in figure
                                                      time_series_n['time'] = list of float
                                                      time_series_n['data'] = list of float
        :param t_start: format to be defined; plot all data if None
        :param t_end: format to be defined; plot all data if None
        :return: a (new) figure with all time series
        """
        # plot in new figure window if not specified
        if fig_handle is None:
            fig = plt.figure(figsize=(16,8), dpi=100)

        for time_series in time_series_list:
            if t_start is None or t_end is None:
                # if not specified, then plot all data
                time_to_plot = time_series['time']
                data_to_plot = time_series['data']
            else:
                index = np.nonzero(t_start <= time_series['time'] <= t_end)[0]
                time_to_plot = time_series['time'][index]
                data_to_plot = time_series['data'][index]

            plt.plot(time_to_plot, data_to_plot, label=time_series['info'])

        plt.title('Time series of pixels'.format())
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamps in seconds')
        plt.legend()
        plt.grid(True)
        plt.draw()

    def plot_heat_map_single_frame(self, timestamp, T_min, T_max):
        """
        Visualization:
        This function plots the heat map for single frame 4x48 at timestamp
        :param timestamp: format to be defined;
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :return: a figure with 4 x 48 color pixel heat map
        """
        pass

    def plot_heat_map_in_period(self, t_start, t_end, T_min, T_max):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each frame 4x48 pixels stacked column by column to 1 x 192
        :param t_start: format to be defined; plot all if None
        :param t_end: format to be defined;; plot all if None
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :return: a figure with 192 x n color map for n frame
        """
        pass

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

        fig = plt.figure(figsize=(16,8), dpi=100)
        im = plt.imshow(data_to_plot,
                        cmap=plt.get_cmap('jet'),
                        interpolation='nearest',
                        vmin=v_min, vmax=v_max)

        plt.title('{0}'.format(title))
        cax = fig.add_axes([0.12, 0.3, 0.78, 0.03])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        plt.draw()







"""
########################################################################################################################
A new class for 2x16 data format
"""

class TrafficData_2x16:
    """
    This class deals with the ultrasonic, 3 PIR in total 2x16 pixels, and IMU data
    """

    def __init__(self):

        # raw data for ultrasonic, the center PIR, and IMU
        # 1 x n samples since the time stamps for the center PIR, in seconds
        self.pir_timestamps = []
        # 32 x n samples. stacked column by column from the second sensor
        self.pir_raw_data = []
        for i in range(0,32):
            self.pir_raw_data.append([])

        # 1 x n samples
        self.ultra_timestamps = None
        # 1 x n samples
        self.ultra_raw_data = None

        # 1 x n samples:
        self.imu_timestamps = None
        # m x n samples; m is the dimension of data (accel, gyro, mag)
        self.imu_data = None

        # data with background subtracted ultrasonic, three PIRs, and IMU
        self.pir_data_background_removed = None

    def read_data_file(self, file_name_str):
        """
        This function reads and parses the file into raw data; it should be updated for new data format
        :param file_name_str: string, the file name string
                            Two types of files can be parsed; format specified by file names
                            PIR_~_2x16.csv: only PIR data with timestamps being the milliseconds; each row:
                                ms, ptat, pir2_pixel00, pir2_pixel10, ...
                            ALL_~.csv: all IMU and PIR data; time format: HH_MM_SS_MS;
                                timestamp,'IMU',magx,magy,magz,accelx,accely,accelz,ultra
                                timestamp,'PIR',ms,ptat, pir2_pixel00, pir2_pixel10, ...
        :return: saved data in class
        """

        for file_str in file_name_str:

            if not exists(file_str):
                print 'Error: file {0} does not exists'.format(file_str)
                return 1

            f = open(file_str, 'r')

            # identify the file format
            file_str = file_str.strip()
            file_str_items = file_str.split('/')
            file_name_items = file_str_items[-1].split('_')
            if file_name_items[0] == 'PIR':
                file_format = 'PIR'
            elif file_name_items[0] == 'ALL':
                file_format = 'ALL'
            else:
                print 'Error: Could not recognize data format for file {0}\n Name should start with PIR or ALL'.format(file_str)
                return 1

            for line in f:
                self.line_parser(line, file_format)

            # change to numpy for easier access
            self.pir_timestamps = np.array(self.pir_timestamps)

            # make sure the timestamp is monotonically increasing
            # millis counter resets when reaches 2^16-1 = 65535, keep track of how many times it reset
            num_reset = 0
            for i in range(1, len(self.pir_timestamps)):

                if self.pir_timestamps[i] < self.pir_timestamps[i-1]:
                    # update number of reset
                    if self.pir_timestamps[i] + num_reset*65535 < self.pir_timestamps[i-1]:
                        num_reset += 1
                    # update the current time stamp
                    self.pir_timestamps[i] += num_reset*65535

            # subtract the first time stamp offset and convert to seconds
            self.pir_timestamps = self.pir_timestamps - self.pir_timestamps[0]
            # self.pir_timestamps[:] = self.pir_timestamps[:]/1000.0

            self.pir_raw_data = np.array(self.pir_raw_data)

            print '\n 32 Hz: size of time {0}; size of data {1}\n'.format(self.pir_timestamps.shape,
                                                                         self.pir_raw_data.shape)

            f.close()

    def line_parser(self, line, parser_option='ALL'):
        """
        This function parses each line of data and save in corresponding class property. Need to be updated if new data
        is in a different format.
        :param line: The line in the data file;
        :param parser_option: 'PIR', 'ALL'; different data format
        :return: save data self.pir_raw_data, self.ultra_raw_data, self.imu_data
        """
        line = line.strip()
        items = line.split(',')

        if parser_option == 'PIR':
            if len(items) != 34:
                print 'Error: Data format is incorrect. Update parser!'
                return 1

            self.pir_timestamps.append(int(items[0]))

            for i in range(2, 34):
                    self.pir_raw_data[i-2].append(float(items[i]))

        # file contains both PIR and ultrasonics data
        if parser_option == 'ALL':
            # TODO: parse the timestamp format
            timestamp_str = items[0]

            if items[1] == 'IMU':
                # parse IMU data
                self.imu_timestamps = timestamp_str
                if len(items) == 9:
                    print 'Warning: missing Gyro data'
                    for i in range(0, 6):
                        self.imu_data[i].append(float(items[i+2]))
                elif len(items) == 12:
                    for i in range(0, 9):
                        self.imu_data[i].append(float(items[i+2]))

                # get the ultrasonic sensor data
                self.ultra_timestamps = timestamp_str
                self.ultra_raw_data.append(float(items[-1]))

            elif items[1] == 'PIR':
                # parse PIR data
                if len(items) < 10:
                    # THe full line should be 2+6+2x16 = 40; shorter length means corrupted data, skip
                    return
                else:
                    self.pir_timestamps.append(timestamp_str)

                    for i in range(0,32):
                        self.pir_raw_data[i].append(float(items[i+4]))

            else:
                print 'Error: incorrect data format.'
                return 1


    def subtract_background(self, background_duration):
        """
        This function subtracts the background of the PIR sensor data; the background is defined as the median of the
        past background_duration samples
        :param background_duration: int, the number of samples regarded as the background
        :return: data saved in self.pir_data_background_removed
        """
        pass

    def calculate_std(self):
        """
        Statistic Analysis:
        This function calculates the mean and standard deviation of each pixel, to better understand the noise.
        :return: print two 4 x 48 matrix: Mean, and STD
        """
        mean = []
        std = []
        for i in range(0, 32):

            if len(self.pir_raw_data[i]) != 0:    # if not empty
                mean.append( np.mean(self.pir_raw_data[i] ) )
                std.append( np.std(self.pir_raw_data[i]) )

                print '32Hz: Pixel {0} with mean {1} and std {2}'.format(i, mean[i], std[i])

        print '32 Hz: Average Std: {0}'.format(np.mean(std))

        # sort the std; remove the large 20% (pixels not well calibrated); then compute the mean std
        std_sorted = sorted(std)
        std_sorted_partial = std_sorted[0:int(len(std_sorted)*0.8)]
        print '32 Hz: Average Std for 80% well calibrated pixels: {0}'.format(np.mean(std_sorted_partial))

        return mean, std

    def plot_histogram_for_pixel(self, pixel_list):
        """
        Statistic Analysis:
        This function plots the histogram of the data for a selected pixel, to better understand the noise
        :param pixel_list: list, [pixel_1, pixel_2...]; pixel_n := (row, col) in ([1,2] [16,31])
        :return: one figure for each pixel
        """

        mu, sigma = self.calculate_std()

        for pixel in pixel_list:

            pixel_index = (pixel[1]-16)*2 + (pixel[0]-1)

            # grab the data
            time_series = self.pir_raw_data[pixel_index, :]

            # the histogram of the data
            num_bins = 50
            fig = plt.figure(figsize=(16,8), dpi=100)
            n, bins, patches = plt.hist(time_series, num_bins, normed=1, facecolor='green', alpha=0.75)

            # add a 'best fit' line
            norm_fit_line = mlab.normpdf(bins, mu[pixel_index], sigma[pixel_index])
            l = plt.plot(bins, norm_fit_line, 'r--', linewidth=1)

            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Probability')
            plt.title(r'Histogram of pixel {0} at 32 Hz: $\mu$= {1}, $\sigma$={2}'.format(pixel,
                                                                                            mu[pixel_index],
                                                                                            sigma[pixel_index]))
            # plt.axis([40, 160, 0, 0.03])
            plt.grid(True)

        plt.draw()

    def plot_time_series_for_pixel(self, t_start=None, t_end=None, pixel_list=None, data_option=None):
        """
        Visualization:
        This function plots the time series from t_start to t_end for pixels in pixel_list; data_option specifies whether
        it should plot the raw data or the data with background removed.
        :param t_start: float, in seconds; plot all data if None
        :param t_end: float, in seconds, millisecond; plot all data if None
        :param pixel_list: list, [pixel_1, pixel_2...]; pixel_n := (row, col) in ([0,3] [0,47])
        :param data_option: string, 'raw', 'background_removed'
        :return: a figure with all the pixels time series
        """

        time_series_to_plot = []

        for pixel in pixel_list:

            pixel_index = (pixel[1]-16)*2 + (pixel[0]-1)

            if 1<= pixel[0] <=2 and 16 <= pixel[1] <= 31:
                timestramps = self.pir_timestamps
            else:
                print 'Error: pixel {0} is not sampled'.format(pixel)
                return 1

            # get the data for the corresponding pixel
            if data_option == 'raw':
                data = self.pir_raw_data[pixel_index,:]
            elif data_option == 'background_removed':
                data = self.pir_data_background_removed[pixel_index, :]
            else:
                data = []
                print 'Error: pixel {0} is not recognized'.format(pixel)

            pixel_series = {}
            pixel_series['info'] = pixel
            pixel_series['time'] = timestramps
            pixel_series['data'] = data

            time_series_to_plot.append(pixel_series)

            # call the generic function to plot
            self.plot_time_series(None, time_series_to_plot, t_start, t_end)

    def plot_time_series_for_ultra(self, t_start, t_end):
        """
        Visualization:
        This function plots the time series for the ultrasonics sensor data from t_start to t_end
        :param t_start: format to be defined; plot all data if None
        :param t_end: format to be defined; plot all data if None
        :return: a figure with the ultrasonic data plotted
        """
        pass

    def plot_time_series(self, fig_handle, time_series_list, t_start, t_end):
        """
        Visualization:
        This function plots all the time series in the time_series_list, which offers more flexibility.
        :param fig_handle: fig or ax handle to plot on; create new figure if None
        :param time_series_list: list, [time_series_1, time_series_2,...];
                                 time_series_n: dict; time_series_n['info'] = string or label in figure
                                                      time_series_n['time'] = list of float
                                                      time_series_n['data'] = list of float
        :param t_start: format to be defined; plot all data if None
        :param t_end: format to be defined; plot all data if None
        :return: a (new) figure with all time series
        """
        # plot in new figure window if not specified
        if fig_handle is None:
            fig = plt.figure(figsize=(16,8), dpi=100)

        for time_series in time_series_list:
            if t_start is None or t_end is None:
                # if not specified, then plot all data
                time_to_plot = time_series['time']
                data_to_plot = time_series['data']
            else:
                index = np.nonzero(t_start <= time_series['time'] <= t_end)[0]
                time_to_plot = time_series['time'][index]
                data_to_plot = time_series['data'][index]

            plt.plot(time_to_plot, data_to_plot, label=time_series['info'])

        plt.title('Time series of pixels'.format())
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamps in seconds')
        plt.legend()
        plt.grid(True)
        plt.draw()

    def plot_heat_map_single_frame(self, timestamp, T_min, T_max):
        """
        Visualization:
        This function plots the heat map for single frame 4x48 at timestamp
        :param timestamp: format to be defined;
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :return: a figure with 4 x 48 color pixel heat map
        """
        pass

    def plot_heat_map_in_period(self, t_start, t_end, T_min, T_max):
        """
        Visualization:
        This function plots the heat map from t_start to t_end with each frame 4x48 pixels stacked column by column to 1 x 192
        :param t_start: format to be defined; plot all if None
        :param t_end: format to be defined;; plot all if None
        :param T_min: the min temperature for the color bar
        :param T_max: the max temperature for the color bar
        :return: a figure with 192 x n color map for n frame
        """
        pass

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

        fig = plt.figure(figsize=(16,8), dpi=100)
        im = plt.imshow(data_to_plot,
                        cmap=plt.get_cmap('jet'),
                        interpolation='nearest',
                        vmin=v_min, vmax=v_max)

        plt.title('{0}'.format(title))
        cax = fig.add_axes([0.12, 0.3, 0.78, 0.03])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        plt.draw()


'''

# This is the class for a single MLX90620
# including the library for computing the temperature from IRraw
# also a variety of visualization methods
class PIR_MLX90620:

    def __init__(self, pir_id):

        self.pir_id = pir_id

        # The following properties declared here are used for computing from IRraw to temperature
        # alpha matrix for each PIR sensor, should be set using import_eeprom function
        self.alpha_ij = None    # will be set later
        self.eepromData = None

        # Following constant are used to compute the temperature from IRraw
        # constants needs to be first computed and then used
        self.v_th = 0
        self.a_cp = 0
        self.b_cp = 0
        self.tgc = 0
        self.b_i_scale = 0
        self.k_t1 = 0
        self.k_t2 = 0
        self.emissivity = 0

        self.a_ij = np.zeros(64)  # 64 array
        self.b_ij = np.zeros(64)

        # all temperature data
        # timestamps [epoch time] np array; t_millis np array
        self.time_stamps = None
        self.t_millis = None
        # all_temperatures, a np matrix: 64 row x n_sample; each column is a frame
        self.all_temperatures = None
        # all ambient temperature, np array
        self.all_Ta = None

        # the following are the latest frame of data
        self.temperatures = np.zeros((4, 16))
        self.Tambient = 0

        # figure handles for plotting the time series of one pixel
        # a list of used figure handles
        self.fig_handles = []
        self.pixel_id = 0   # by default 0




    # statistic analysis
    # calculate the mean and std of the measurement of each pixel
    def calculate_std(self):

        # skip the first a few that have non sense values due to transmission corruption
        std = []
        for i in range(0, 64):
            # print 'all_temperatures:{0}'.format(self.all_temperatures[i])

            if self.all_temperatures[i]:    # if not empty
                # print 'before_std'
                std.append( np.std(self.all_temperatures[i]) )
                # print 'after_std'
        if len(std) == 64:
            print 'std of PIR {0} with mean std {1}: \n'.format(self.pir_id, np.mean(std))
            # print 'std {0}'.format(std)
            print 'std of each pix:{0}'.format(np.reshape(std, (16, 4) ).T)


    # visualization
    # init_fig and update_fig are used for real-time plotting from serial stream data.
    # initialize the plotting for time series of chosen pixels
    def init_fig(self, pixel_id, T_min, T_max):

        # update pixel id
        self.pixel_id = pixel_id

        fig = plt.figure()
        ax = fig.add_subplot(111)    # fig_handles[0]
        self.fig_handles.append( (fig, ax) )

        # initial data
        # ultrasonic sensor sample at 30 Hz
        # Hence 5s = 150 pts
        # t_init = np.arange(-150*0.03, 0, 0.03)
        t_init = np.arange(0, 150*0.03, 0.03)
        y_init = np.zeros(150)
        self.fig_handles.append(t_init)     # fig_handles[1]
        self.fig_handles.append(y_init)     # fig_handles[2]

        ul, = self.fig_handles[0][1].plot(self.fig_handles[1],
                                    self.fig_handles[2])

        self.fig_handles.append(ul) # fig_handles[3]

        # draw and show it
        self.fig_handles[0][0].canvas.draw()
        plt.show(block=False)
        plt.ylim((T_min,T_max))

    # visualization
    def update_fig(self):

        if len(self.all_temperatures) >= self.pixel_id:
            # pad to 5 s
            if len(self.all_temperatures[self.pixel_id]) >= 150:
                # t_toplot = self.all_ultra[0][-150:]
                y_toplot = self.all_temperatures[self.pixel_id][-150:]
            else:
                # t_tmp = np.concatenate([self.fig_handles[1], self.all_ultra[0]])
                y_tmp = np.concatenate([self.fig_handles[2], self.all_temperatures[self.pixel_id]])
                # t_toplot = t_tmp[-150:]
                y_toplot = y_tmp[-150:]

            # self.fig_handles[3].set_xdata(t_toplot)
            self.fig_handles[3].set_ydata(y_toplot)

            self.fig_handles[0][0].canvas.draw()


    # visualization
    # The following function is used for plot the time series of one or multiple pixel in a given time range
    # input: t_start, and t_end are the epoch time of the interested interval
    #        if t_start or t_end not specified, then plot all data.
    #       pixel_id is a list of pixel index tuples [(0,0),(3,4)] in the 4x16 matrix
    def plot_time_series_of_pixel(self, t_start, t_end, pixel_id):

        # extract data to plot from the properties
        if t_start is None or t_end is None:
            # if not specified, then plot all data
            index = np.nonzero(self.time_stamps)[0]
        else:
            index = np.nonzero(t_start <= self.time_stamps <= t_end)[0]

        data_to_plot = []
        # only extract the pixels data to be plotted
        for i in range(0, len(pixel_id)):
            pixel_index = pixel_id[i][1]*4 + pixel_id[i][0]
            data_to_plot.append(self.all_temperatures[pixel_index, index])

        fig = plt.figure(figsize=(16,8), dpi=100)
        for i in range(0, len(pixel_id)):
            plt.plot(self.time_stamps, data_to_plot[i])

        plt.title('Time series of PIR {0}, pixel {1}'.format(self.pir_id, pixel_id))
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamps in EPOCH')
        plt.show()


    # visualization
    # The following function plot a static single 4x16 pixel frame given the time interval
    # input: each frame in the [t_start, t_end] interval will be plotted in separate figures
    #        T_min and T_max are the limit for the color bar
    # TODO: There is a minor bug on im[2]
    def plot_heat_map(self, t_start, t_end, T_min, T_max):
        
        # extract data to plot from the properties
        index = np.nonzero(t_start <= self.time_stamps <= t_end)[0]
        data_to_plot = self.all_temperatures[:, index]

        for i in range(0, data_to_plot.shape[1]):
            fig = plt.figure(figsize=(16,8), dpi=100)
            im = plt.imshow(data_to_plot[i][:,i].reshape(16,4).T,
                            cmap=plt.get_cmap('jet'),
                            interpolation='nearest',
                            vmin=T_min, vmax=T_max)

            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            fig.colorbar(im[2], cax=cax)
            plt.title('heat map of PIR {0}'.format(self.pir_id))
            plt.show()


    # visualization
    # The following function plot the heat map video given a time interval for just one PIR sensor
    # input: t_start and t_end are in epoch time,
    #        T_min and T_max are the limits for color bar
    #        fps is the number of frames per second
    def plot_heat_map_video(self, t_start, t_end, T_min, T_max, fps):
        # initialize figure
        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 3.5)
        ax.hold(True)

        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

        im = []
        T_init = np.zeros((4, 16)) + T_min
        im.append(ax.imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=T_min, vmax=T_max))

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(im[2], cax=cax)

        ax.set_ylabel('PIR {0}'.format(self.pir_id))
        # self.ax[i].get_xaxis().set_visible(False)
        # self.ax[i].get_yaxis().set_visible(False)

        plt.show(False)
        plt.draw()

        # extract data to play from the properties
        index = np.nonzero(t_start <= self.time_stamps <= t_end)[0]
        # each columnn is one frame
        data_to_play = self.all_temperatures[:, index]

        timer_start = time.time()
        for frame_index in range(0, data_to_play.shape[1]):

            # wait using the fps
            while time.time() - timer_start <= 1/fps:
                time.sleep(0.005)   # sleep 5 ms
                continue

            # reset timer
            timer_start = time.time()

            # update figure
            # print 'T[{0}]: {1}'.format(i, T[i])
            im.set_data(data_to_play[:,frame_index].reshape(16,4).T)

            fig.canvas.restore_region(background)
            ax.draw_artist(im)
            fig.canvas.blit(ax.bbox)



# This is the class for three MLX90620 PIR sensors
# Sometimes we need to process three PIR sensors at the same time, such as plotting a heat map video in 3 subfigures,
# or read 3 PIR data from a file
class PIR_3_MLX90620:

    def __init__(self):

        # create three PIR objects
        self.pir1 = PIR_MLX90620(1)
        self.pir2 = PIR_MLX90620(2)
        self.pir3 = PIR_MLX90620(3)

    # read all three PIR data from a file
    def read_data_from_file(self, file_name_str):

        f_handle = open(file_name_str, 'r')
        data_set = csv.reader(f_handle)

        # save in a list, then save to pir np matrix
        time_stamps = []
        all_temperatures_1 = []
        all_temperatures_2 = []
        all_temperatures_3 = []
        for i in range(0,64):
            all_temperatures_1.append([])
            all_temperatures_2.append([])
            all_temperatures_3.append([])

        t_millis_1 = []
        t_millis_2 = []
        t_millis_3 = []

        all_Ta_1 = []
        all_Ta_2 = []
        all_Ta_3 = []

        # parse line into the list
        for line in data_set:
            #print line
            #print len(line)
            # the first line may be \n
            if len(line) < 100:
                continue

            time_stamps.append(float(line[0]))

            # pir sensor 1
            index = 1
            t_millis_1.append(float(line[index]))
            index +=1

            for i in range(0,64):
                all_temperatures_1[i].append(float(line[index]))
                index +=1

            all_Ta_1.append(float(line[index]))
            index +=1
            # skip ultrasonic sensor
            index +=1

            # pir sensor 2
            t_millis_2.append(float(line[index]))
            index +=1

            for i in range(0,64):
                all_temperatures_2[i].append(float(line[index]))
                index +=1

            all_Ta_2.append(float(line[index]))
            index +=1
            # skip ultrasonic sensor
            index +=1

            # pir sensor 3
            t_millis_3.append(float(line[index]))
            index +=1

            for i in range(0,64):
                all_temperatures_3[i].append(float(line[index]))
                index +=1

            all_Ta_3.append(float(line[index]))
            index +=1
            # skip ultrasonic sensor
            index +=1

        f_handle.close()

        # save and convert those into np matrix for each PIR object
        self.pir1.time_stamps = np.array(time_stamps)
        self.pir1.t_millis = t_millis_1
        self.pir1.all_temperatures = np.array(all_temperatures_1)
        self.pir1.all_Ta = np.array(all_Ta_1)

        self.pir2.time_stamps = np.array(time_stamps)
        self.pir2.t_millis = t_millis_2
        self.pir2.all_temperatures = np.array(all_temperatures_2)
        self.pir2.all_Ta = np.array(all_Ta_2)

        self.pir3.time_stamps = np.array(time_stamps)
        self.pir3.t_millis = t_millis_3
        self.pir3.all_temperatures = np.array(all_temperatures_3)
        self.pir3.all_Ta = np.array(all_Ta_3)


    # plot PIR heat map video from saved data (For real-time play from serial ports, refer to PlotPIR class)
    # t_start and t_end are the starting and end time of the video
    # T_min, T_max are the colorbar limits
    # fps is frames per second (theoretically up to 300 fps
    def play_video(self, t_start, t_end, T_min, T_max, fps):

        # initialize figure
        fig, ax = plt.subplots(3,1, figsize=(15,15))

        for i in range(0,3):
            ax[i].set_aspect('equal')
            ax[i].set_xlim(-0.5, 15.5)
            ax[i].set_ylim(-0.5, 3.5)
            ax[i].hold(True)

        # cache the background
        background = fig.canvas.copy_from_bbox(ax[0].bbox)

        im = []
        T_init = np.zeros((4, 16)) + T_min
        for i in range(0, 3):
            im.append(ax[i].imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=T_min, vmax=T_max))

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(im[2], cax=cax)

        # set labels
        for i in range(0, 3):
            ax[i].set_ylabel('PIR {0}'.format(i+1))
            # self.ax[i].get_xaxis().set_visible(False)
            # self.ax[i].get_yaxis().set_visible(False)

        plt.show(False)
        plt.draw()

        # extract data to play from the properties
        if t_start is None or t_end is None:
            # if none, then play the entire date set
            index = np.nonzero(self.pir1.time_stamps)[0]
        else:
            index = np.nonzero(t_start <= self.pir1.time_stamps <= t_end)[0]
        # each columnn is one frame
        data_to_play = []
        data_to_play.append(self.pir1.all_temperatures[:, index])
        data_to_play.append(self.pir2.all_temperatures[:, index])
        data_to_play.append(self.pir3.all_temperatures[:, index])

        timer_start = time.time()
        for frame_index in range(0, data_to_play[0].shape[1]):

            # wait using the fps
            while time.time() - timer_start <= 1/fps:
                time.sleep(0.005)   # sleep 5 ms
                continue

            # reset timer
            timer_start = time.time()

            # update figure
            for i in range(0, 3):
                # print 'refreshed figure'
                # print 'T[{0}]: {1}'.format(i, T[i])

                print 'pir-{0}:{1}'.format(i+1,data_to_play[i][:,frame_index].reshape(16,4).T)

                im[i].set_data(data_to_play[i][:,frame_index].reshape(16,4).T)

                fig.canvas.restore_region(background)
                ax[i].draw_artist(im[i])
                fig.canvas.blit(ax[i].bbox)





# This class plots the heat map for three PIR sensors
# it plots three subfigures in three rows and one column, each figure use imshow to plot a 4x16 matrix
# theoretically this plot can refresh as fast as 300 frames/s
class PlotPIR:
    def __init__(self, num_plot, T_min, T_max):
        self.fig, self.ax = plt.subplots(num_plot, 1)
        for i in range(0, num_plot):
            self.ax[i].set_aspect('equal')
            self.ax[i].set_xlim(-0.5, 15.5)
            self.ax[i].set_ylim(-0.5, 3.5)
            self.ax[i].hold(True)

        # cache the background
        # in fact, they can share the background
        self.background_1 = self.fig.canvas.copy_from_bbox(self.ax[0].bbox)
        self.background_2 = self.fig.canvas.copy_from_bbox(self.ax[1].bbox)
        self.background_3 = self.fig.canvas.copy_from_bbox(self.ax[2].bbox)

        self.im = []
        T_init = np.zeros((4, 16)) + T_min
        for i in range(0, 3):
            self.im.append(self.ax[i].imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=T_min, vmax=T_max))

        cax = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
        self.fig.colorbar(self.im[2], cax=cax)

        # set axis
        for i in range(0, 3):
            self.ax[i].set_ylabel('PIR {0}'.format(i+1))
            # self.ax[i].get_xaxis().set_visible(False)
            # self.ax[i].get_yaxis().set_visible(False)

        plt.show(False)
        plt.draw()

    # T is a list of three element, each element is a 4x16 matrix corresponding to three PIR
    def update(self, T):
        # tic = time.time()

        for i in range(0, 3):
            # print 'T[{0}]: {1}'.format(i, T[i])
            self.im[i].set_data(T[i])

            self.fig.canvas.restore_region(self.background_1)

            self.ax[i].draw_artist(self.im[i])

            self.fig.canvas.blit(self.ax[i].bbox)
'''















