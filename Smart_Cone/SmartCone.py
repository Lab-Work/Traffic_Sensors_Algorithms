'''
Authors: Yanning Li, Fangyu Wu
Date: Summer 2015

THIS MODULE CONSISTS OF 5 sections:
     LIABRARY: The raw data feedback from the sensor, parsed in a useful form
     VISUALIZATION: Visualization tools
     STATISTICS: Simple statistics properties of the library
     ESTIMATION: Model for vehicle detection, classification, and speed estimation
     VALIDATION: Validation tools
'''

import os
import csv
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn import cross_validation
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True) # This will create LaTex style plotting.

#Primary class
class SmartCone: 

    #LIABRARY
    def __init__(self,mode='read',buffer='inf'):

        #Data parsed into useful form
        self.time = np.array([])   #Time stamp
        self.elapse = np.array([]) #Time elapsed from the starting time stamp

        self.millis1 = np.array([])#The millisecs from the start of Arduino to time of measure
        self.pir1 = np.array([])   #The leftmost PIR reading
        self.amb1 = np.array([])   #The leftmost PIR ambient temperature reading 
        self.ultra1 = np.array([]) #The corresponding ultrasonic sensor reading

        self.millis2 = np.array([])#The millisecs from the start of Arduino to time of measure
        self.pir2 = np.array([])   #The middle PIR reading
        self.amb2 = np.array([])   #The middle PIR ambient temperature reading
        self.ultra2 = np.array([]) #The corresponding ultrasonic sensor reading

        self.millis3 = np.array([])#The millisecs from the start of Arduino to time of measure
        self.pir3 = np.array([])   #The rightmost PIR reading
        self.amb3 = np.array([])   #The rightmost PIR ambient temperature reading
        self.ultra3 = np.array([]) #The corresponding ultrasonic sensor reading

        if mode == 'read':
            #Get the relative paths of data in data/.
            self.FILES = []
            for root, dirs, files in os.walk('data/'):
                for file in sorted(files):
                    if file.endswith(".csv"):
                        self.FILES.append(os.path.join(root, file))
                        #print self.FILES

            #Store data in buffer, which essentially acts like a queue.
            if buffer == 'inf':
                print 'Buffer size is infinitive.'
            elif isinstance(buffer,int) and buffer > 0:
                print 'Buffer size is a positive integer.'
            else:
                sys.exit('ERROR: Buffer needs to be a positive integer. Abort!')
        
        #TODO
        elif mode == 'listen':
            sys.exit('WARNING: Mode to be defined. Abort!')
            pass
        else:
            sys.exit('ERROR: Mode not defined. Abort!')

    #Buffer new data into the data queue
    def buffering(self,data):
        self.time = np.append(self.time,timeFormat(data[0]))
        self.elapse = np.append(self.elapse,self.time[0]-timeFormat(data[0]))

        self.millis1 = np.append(self.millis1,float(data[1]))
        self.pir1 = np.append(self.pir1,float(data[2:66]))
        self.amb1 = np.append(self.amb1,float(data[67]))
        self.ultra1 = np.append(self.ultra1,float(data[68]))

        self.millis2 = np.append(self.millis2,float(data[69]))
        self.pir2 = np.append(self.pir2,float(data[70:134]))
        self.amb2 = np.append(self.amb2,float(data[135]))
        self.ultra2 = np.append(self.ultra2,float(data[136]))

        self.millis3 = np.append(self.millis3,float(data[137]))
        self.pir3 = np.append(self.pir3,float(data[138:202]))
        self.amb3 = np.append(self.amb3,float(data[203]))
        self.ultra3 = np.append(self.ultra3,float(data[204]))

    #Debuffer old data from the data queue
    def debuffering(self):
        np.delete(self.time,0)
        np.delete(self.elapse,0)

        np.delete(self.millis1,0)
        np.delete(self.pir1,0)
        np.delete(self.amb1,0)
        np.delete(self.ultra1,0)

        np.delete(self.millis2,0)
        np.delete(self.pir2,0)
        np.delete(self.amb2,0)
        np.delete(self.ultra2,0)

        np.delete(self.millis3,0)
        np.delete(self.pir3,0)
        np.delete(self.amb3,0)
        np.delete(self.ultra3,0)

    #VISUALIZATION
    def timeSeries(self):
        pass

    def heatMap(self):
        pass

    #STATISTICS
    def stdev(self):
        pass

    def avg(self):
        pass

    #ESTIMATION
    def estimate(self,estimator=None):
        pass

    #VALIDATION
    def scores(self):
        pass

    def inspect(self):
        pass


#Helper functions
def timeFormat(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

