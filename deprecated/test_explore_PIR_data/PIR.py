# This is the Class for PIR90620 sensor.
# This class can:
# 1. import the EEPROM register values and compute the temperature from IRraw data
# 2. Save all data in clean data structure, either from serial port reading or data files
# 3. visualize the data, heat_map or time series of a single pixel

# if a comment for a function starts with:
# LIBRARY: for computing IRraw data to temperature. Not needed to process data which is already in temperature form
# Visualization: methods for visualizing data
# statistic analysis: some handy functions for checking the statistic property

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import time
from MLX90620_register import *


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

    # PIR LIBRARY
    # TODO: changed the property of the PIR class to np array. Need to update.
    # import EEPROM, must be called if want to compute temperature from IRraw data.
    def import_eeprom(self, alpha_ij, eepromData):
        self.alpha_ij = np.copy(alpha_ij)
        self.eepromData = np.copy(eepromData)

    # PIR LIBRARY
    # initialize the constant parameters
    def const_init(self):
        self.v_th = 256*self.eepromData[VTH_H] + self.eepromData[VTH_L]
        self.k_t1 = (256*self.eepromData[KT1_H] + self.eepromData[KT1_L])/1024.0
        self.k_t2 = (256 * self.eepromData[KT2_H] + self.eepromData[KT2_L]) / 1048576.0 #2^20 = 1,048,576
        self.emissivity = (256 * self.eepromData[CAL_EMIS_H] + self.eepromData[CAL_EMIS_L]) / 32768.0

        self.a_cp = self.eepromData[CAL_ACP]
        if(self.a_cp > 127):
            self.a_cp -= 256 #These values are stored as 2's compliment. This coverts it if necessary.

        self.b_cp = self.eepromData[CAL_BCP]
        if(self.b_cp > 127):
            self.b_cp -= 256

        self.tgc = self.eepromData[CAL_TGC]
        if(self.tgc > 127):
            self.tgc -= 256

        self.b_i_scale = self.eepromData[CAL_BI_SCALE]

        for i in range(0, 64):
            # Read the individual pixel offsets
            self.a_ij[i] = self.eepromData[i]
            if(self.a_ij[i] > 127):
                self.a_ij[i] -= 256 #These values are stored as 2's compliment. This coverts it if necessary.

            #Read the individual pixel offset slope coefficients
            self.b_ij[i] = self.eepromData[0x40 + i] #Bi(i,j) begins 64 bytes into EEPROM at 0x40
            if(self.b_ij[i] > 127):
                self.b_ij[i] -= 256

        # print 'finished initializing values\n'
        # print 'v_th: {0}\n'.format(self.v_th)
        # print 'k_t1: {0}\n'.format(self.k_t1)
        # print 'k_t2: {0}\n'.format(self.k_t2)
        # print 'emissivity: {0}\n'.format(self.emissivity)
        # print 'a_cp: {0}\n'.format(self.a_cp)
        # print 'b_cp: {0}\n'.format(self.b_cp)
        # print 'tgc: {0}\n'.format(self.tgc)
        # print 'b_i_scale: {0}\n'.format(self.b_i_scale)
        # print 'a_ij:'
        # for i in range(0, 64):
        #     print self.a_ij[i]
        # print '\n b_ij:'
        # for i in range(0, 64):
        #     print self.b_ij[i]

    # PIR LIBRARY
    # compute the Ta and To
    # Due to data corruption, if Ta is negative, then stop and do not compute To
    def calculate_temperature(self, ptat, irData, cpix):

        self.calculate_TA(ptat)

        if self.Tambient <= -100:
            self.Tambient = 0   # reset, do not proceed to compute the To
        else:
            self.calculate_TO(irData, cpix)

    # PIR LIBRARY
    # calculate TA
    def calculate_TA(self, ptat):
        self.Tambient = (-self.k_t1 + np.sqrt(np.power(self.k_t1, 2) -
                                              (4 * self.k_t2 * (self.v_th - ptat)))) / (2*self.k_t2) + 25
        # print 'ambient temperature: '
        # print self.Tambient

    # PIR LIBRARY
    # calculate object temperature
    # irData: a np(4,16) matrix with raw data
    # output saved in temperatures(4,16)
    def calculate_TO(self, irData, cpix):

        #Calculate the offset compensation for the one compensation pixel
        #This is a constant in the TO calculation, so calculate it here.
        v_cp_off_comp = cpix - (self.a_cp + (self.b_cp/np.power(2, self.b_i_scale)) * (self.Tambient - 25))

        for col in range(0, 16):
            for row in range(0, 4):
                i = col*4 + row
                v_ir_off_comp = irData[row, col] - (self.a_ij[i] + (self.b_ij[i]/np.power(2, self.b_i_scale)) * (self.Tambient - 25)) #1: Calculate Offset Compensation

                v_ir_tgc_comp = v_ir_off_comp - ((self.tgc/32) * v_cp_off_comp) #2: Calculate Thermal Gradien Compensation (TGC)

                v_ir_comp = v_ir_tgc_comp/self.emissivity  #3: Calculate Emissivity Compensation

                self.temperatures[row, col] = np.sqrt(np.sqrt((v_ir_comp/self.alpha_ij[i]) + np.power(self.Tambient + 273.15, 4) )) - 273.15

                # save into all_temperatures
                self.all_temperatures[i].append(self.temperatures[row, col])


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


    # statistic analysis
    # aggregate the values across pixels in the second row
    def plot_agg_temp(self, pixel_id, t_start, t_end):
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

        data_to_plot_sum = np.zeros(len(data_to_plot[0]))
        # sum the values across pixels
        for i in range(0, len(data_to_plot[0])):
            for pix in range(0,len(pixel_id)):
                data_to_plot_sum[i] += data_to_plot[pix][i]

        fig = plt.figure(figsize=(16,8), dpi=100)
        plt.plot(self.time_stamps, data_to_plot_sum)

        plt.title('Aggregated time series of PIR {0}, pixel {1}'.format(self.pir_id, pixel_id))
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamps in EPOCH')
        plt.show()


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
        data_to_plot = []
        time_stamps_to_plot = []
        if t_start is not None and t_end is not None:
            print 'extracting data between {0} and {1}'.format(t_start, t_end)
            for index in range(0, len(self.time_stamps)):
                if self.time_stamps[index] >= t_start and self.time_stamps[index] <= t_end:
                    time_stamps_to_plot.append(self.time_stamps[index])
                    data_to_plot.append(self.all_temperatures[:, index])
                    # print data_to_plot

        else:
            time_stamps_to_plot = self.time_stamps
            data_to_plot = self.all_temperatures

        # print 'data_to_plot size: {0}'.format(len(data_to_plot))

        if len(data_to_plot) == 0:
            print 'Warning: no data between the specified start and end time'
            return 1

        # make sure the dimension is correct: 64 x n frames
        data_to_plot = np.array(data_to_plot)
        if data_to_plot.shape[0] != 64:
            data_to_plot = data_to_plot.T

            if data_to_plot.shape[0] != 64:
                print 'Error: check the dimension of the data. It should be 64 x n frames'
                return 1

        print 'data_to_plot dimension: {0}'.format(data_to_plot.shape)


        pixel_data_to_plot = []
        # only extract the pixels data to be plotted
        for i in range(0, len(pixel_id)):
            pixel_index = pixel_id[i][1]*4 + pixel_id[i][0]
            pixel_data_to_plot.append(data_to_plot[pixel_index, :])


        print 'time_stamps_to_plot dim:{0} and pixel_data_to_plot dim {1}'.format(len(time_stamps_to_plot), len(pixel_data_to_plot[0]))

        fig = plt.figure(figsize=(8,4), dpi=100)
        for i in range(0, len(pixel_id)):
            plt.plot(time_stamps_to_plot, pixel_data_to_plot[i])

        plt.title('Time series of PIR {0}, pixel {1}'.format(self.pir_id, pixel_id))
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamps in EPOCH')
        plt.draw()


    # visualization
    # The following function plot a static single 4x16 pixel frame given the time interval
    # input: each frame in the [t_start, t_end] interval will be plotted in separate figures
    #        T_min and T_max are the limit for the color bar
    def plot_heat_map(self, t_start, t_end, T_min, T_max):

        # extract data to plot from the properties
        time_stamps_to_plot = []
        data_to_plot = []
        if t_start is not None and t_end is not None:
            for index in range(0, len(self.time_stamps)):
                if self.time_stamps[index] >= t_start and self.time_stamps[index] <= t_end:
                    time_stamps_to_plot.append(self.time_stamps[index])
                    data_to_plot.append(self.all_temperatures[:, index])

        else:
            if len(self.time_stamps) >= 10:
                print 'Warning: there are too many frames ({0}) to plot, please use plot_heat_map_video'.format(len(self.time_stamps))
                return 1
            else:
                time_stamps_to_plot = self.time_stamps
                data_to_plot = self.all_temperatures

        data_to_plot = np.array(data_to_plot)

        # make sure the data to plot has the righ dimension: 64 x n frames
        if data_to_plot.shape[0] != 64:
            data_to_plot = data_to_plot.T

            if data_to_plot.shape[0] != 64:
                print 'Error: check the dimension of the data. It should be 64 x n frames'
                return 1

        for i in range(0, data_to_plot.shape[1]):
            fig = plt.figure(figsize=(10,5), dpi=100)
            im = plt.imshow(data_to_plot[i][:,i].reshape(16,4).T,
                            cmap=plt.get_cmap('jet'),
                            interpolation='nearest',
                            vmin=T_min, vmax=T_max)

            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            fig.colorbar(im[2], cax=cax)
            plt.title('heat map of PIR {0}'.format(self.pir_id))
            plt.draw()


    # visualization
    # The following function plot the heat map video given a time interval for just one PIR sensor
    # input: t_start and t_end are in epoch time,
    #        T_min and T_max are the limits for color bar
    #        fps is the number of frames per second
    def plot_heat_map_video(self, t_start, t_end, T_min, T_max, fps, num_background_frame):
        # initialize figure
        fig, ax = plt.subplots(figsize=(10,5))

        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 3.5)
        ax.hold(True)

        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

        T_init = np.zeros((4, 16)) + T_min
        im = ax.imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=T_min, vmax=T_max)
        # im = ax.imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest')

        cax = fig.add_axes([0.92, 0.3, 0.01, 0.5])
        fig.colorbar(im, cax=cax)

        ax.set_ylabel('PIR {0}'.format(self.pir_id))
        # self.ax[i].get_xaxis().set_visible(False)
        # self.ax[i].get_yaxis().set_visible(False)

        plt.show(False)
        plt.draw()

        # extract data to play in a time period
        data_to_play = []
        if t_start is not None and t_end is not None:
            for index in range(0, len(self.time_stamps)):
                if self.time_stamps[index] >= t_start and self.time_stamps[index] <= t_end:
                    data_to_play.append(self.all_temperatures[:, index])
        else:
            # otherwise plot all data
            data_to_play = self.all_temperatures

        data_to_play = np.array(data_to_play)

        if data_to_play.shape[0] != 64:
            data_to_play = data_to_play.T

            if data_to_play.shape[0] != 64:
                print 'Error: check the dimension of the data. It should be 64 x n frames'
                return 1

        timer_start = time.time()
        for frame_index in range(0, data_to_play.shape[1]):

            # wait using the fps
            while time.time() - timer_start <= 1.0/fps:
                time.sleep(0.005)   # sleep 5 ms
                continue

            # reset timer
            timer_start = time.time()
            # print('%.5f' % timer_start)

            # update figure
            # print 'T[{0}]: {1}'.format(i, T[i])
            if num_background_frame is None or frame_index < num_background_frame:
                frame_to_plot = data_to_play[:, frame_index].reshape(16,4).T
            else:
                # subtract the background which is the average of the past num_background_frame
                start_background_index = frame_index - num_background_frame
                end_background_index = frame_index # in fact should be frame_index - 1; not included in numpy indexing
                background_data = data_to_play[:, start_background_index: end_background_index]
                background = np.mean(background_data, 1)

                # subtract the background
                frame_background_subtracted = data_to_play[:, frame_index] - background
                frame_to_plot = frame_background_subtracted.reshape(16,4).T

                print 'before: {0}\nafter:  {1}'.format(data_to_play[:, frame_index], frame_background_subtracted)


            im.set_data(frame_to_plot)

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
            index = np.nonzero([i&j for i in t_start <= self.pir1.time_stamps for j in self.pir1.time_stamps <= t_end])[0]
        # each columnn is one frame

        temp_pir1_data = self.pir1.all_temperatures[:, index]
        temp_pir2_data = self.pir2.all_temperatures[:, index]
        temp_pir3_data = self.pir3.all_temperatures[:, index]

        # for pix_id in range(0,64):
        #     temp_pir1_data[pix_id,:] -= np.mean(temp_pir1_data[i,:])
        #     temp_pir2_data[pix_id,:] -= np.mean(temp_pir2_data[i,:])
        #     temp_pir3_data[pix_id,:] -= np.mean(temp_pir3_data[i,:])


        data_to_play = []
        # data_to_play.append(self.pir1.all_temperatures[:, index] - np.mean(self.pir1.all_temperatures[:, index],1))
        # data_to_play.append(self.pir2.all_temperatures[:, index] - np.mean(self.pir2.all_temperatures[:, index],1))
        # data_to_play.append(self.pir3.all_temperatures[:, index] - np.mean(self.pir3.all_temperatures[:, index],1))
        data_to_play.append(temp_pir1_data)
        data_to_play.append(temp_pir2_data)
        data_to_play.append(temp_pir3_data)



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
















