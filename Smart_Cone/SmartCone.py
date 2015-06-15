'''
Authors: Yanning Li, Fangyu Wu
Date: Summer 2015

THIS MODULE CONSISTS OF 5 sections:
     LIABRARY: The raw data feedback from the sensor, parsed in a useful form
     VISUALIZATION: Visualization tools
     STATISTICS: Simple statistics properties of the library
     ESTIMATION: Model for vehicle detection, classification, and speed estimation
     VALIDATION: Validation tools

Essentially, this is an online emulator of the real time smart cone sensor packs.
To use the SmartCone class, do
    emulator = SmartCone(mode,buffer_size,estimator)
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
from Estimators import Estimators

#Primary class
class SmartCone: 

    #LIABRARY
    def __init__(self,mode='read',buffer='inf',estor=None):

        if mode == 'read':

            #Get the relative paths of data in data/.
            self.FILES = []
            for root, dirs, files in os.walk('data/'):
                for file in sorted(files):
                    if file.endswith(".csv"):
                        self.FILES.append(os.path.join(root, file))
                        #print self.FILES
            
            #Pour data from files to a list.
            self.DATASETS = []
            for file in self.FILES:
                with open(file,'r') as dataset:
                    file = csv.reader(file)
                    for line in file:
                        self.DATASETS.append(line)

            #Store data in buffer, which essentially acts like a queue.
            self.BUFFER = []
            if buffer == 'inf':
                print 'Buffer size is infinitive.'
                self.BUFFER = self.DATASETS

            elif isinstance(buffer,int) and buffer > 0:
                print 'Buffer size is a positive integer.'
                self.BUFFER = self.DATASETS[0:buffer]

            else:
                sys.exit('ERROR: Buffer needs to be a positive integer. Abort!')
        
        #TODO: Read data from serial port.
        elif mode == 'listen':
            sys.exit('WARNING: Mode to be defined. Abort!')
            pass

        else:
            sys.exit('ERROR: Mode not defined. Abort!')
            
        #Instantiate an estimator class.
        Estor = Estimators(BUFFER=self.BUFFER,estimator=estor)


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
    def estimate(self):
        
        

    #VALIDATION
    def scores(self):
        pass

    def inspect(self):
        pass


    #HLPER FUNCTIONS

    #Change MATLAB time format to Python datetime format
    def timeFormat(matlab_datenum):
        return datetime.fromordinal(int(matlab_datenum)) + \
               timedelta(days=matlab_datenum%1) - timedelta(days = 366)

    #Update buffer
    def updateBuffer(self,data):

