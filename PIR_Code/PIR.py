# This is the Class for PIR90620 sensor.
# The goal of this class is to
# 1. easily integrate PIR data in data collection code
# 2. visualize data
# 3. post process data from files


import numpy as np
import matplotlib.pyplot as plt
from MLX90620_register import *



class PIR_MLX90620:

    def __init__(self, pir_id, alpha_ij, eepromData):
        # alpha matrix for each PIR sensor
        self.pir_id = pir_id
        self.alpha_ij = np.copy(alpha_ij)
        self.eepromData = np.copy(eepromData)

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

        # raw data to be received
        # a list of 64 arrays. each array is a time series temperature data for one pixel
        self.all_temperatures = []
        for i in range(0,64):
            self.all_temperatures.append([])

        self.temperatures = np.zeros((4, 16))
        self.Tambient = 0

        # figure handles for plotting the time series of one pixel
        # a list of used figure handles
        self.fig_handles = []
        self.pixel_id = 0   # by default 0


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


    # compute the Ta and To
    # Due to data corruption, if Ta is negative, then stop and do not compute To
    def calculate_temperature(self, ptat, irData, cpix):

        self.calculate_TA(ptat)

        if self.Tambient <= -100:
            self.Tambient = 0   # reset, do not proceed to compute the To
        else:
            self.calculate_TO(irData, cpix)


    # calculate TA
    def calculate_TA(self, ptat):
        self.Tambient = (-self.k_t1 + np.sqrt(np.power(self.k_t1, 2) -
                                              (4 * self.k_t2 * (self.v_th - ptat)))) / (2*self.k_t2) + 25
        # print 'ambient temperature: '
        # print self.Tambient

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





# This class plots the temperature heat map
# it plots three subfigures in three rows and one colume, each figure use imshow to plot a 4x16 matrix
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
        self.background_1 = self.fig.canvas.copy_from_bbox(self.ax[0].bbox)
        self.background_2 = self.fig.canvas.copy_from_bbox(self.ax[1].bbox)
        self.background_3 = self.fig.canvas.copy_from_bbox(self.ax[2].bbox)

        self.im = []
        T_init = np.zeros((4, 16)) + T_min
        for i in range(0, 3):
            self.im.append(self.ax[i].imshow(T_init, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=T_min, vmax=T_max))

        cax = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
        self.fig.colorbar(self.im[2], cax=cax)

        # set
        for i in range(0, 3):
            self.ax[i].set_ylabel('PIR {0}'.format(i+1))
            # self.ax[i].get_xaxis().set_visible(False)
            # self.ax[i].get_yaxis().set_visible(False)

        plt.show(False)
        plt.draw()

    def update(self, T):
        # tic = time.time()

        for i in range(0, 3):
            # print 'T[{0}]: {1}'.format(i, T[i])
            self.im[i].set_data(T[i])

            self.fig.canvas.restore_region(self.background_1)

            self.ax[i].draw_artist(self.im[i])

            self.fig.canvas.blit(self.ax[i].bbox)
