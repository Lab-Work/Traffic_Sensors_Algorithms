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
import datetime
from sklearn import linear_model
from sklearn import cross_validation
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True) # This will create LaTex style plotting.
from Estimators import Estimators

#Primary class
class SmartCone: 

    #LIABRARY
    def __init__(self,mode='read',bufferSize='inf',estimatorType=None):

        self.timeDiff = datetime.timedelta(hours=21,minutes=39,seconds=46,milliseconds=472.497)
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
                    fileReader = csv.reader(dataset)
                    for line in fileReader:
                        #print line
                        self.DATASETS.append(line)

            #Store data in buffer, which essentially acts like a queue.
            self.BUFFER = []
            if bufferSize == 'inf':
                print 'Buffer size is infinitive.'
                self.BUFFER = self.DATASETS

            elif isinstance(bufferSize,int) and bufferSize > 0:
                print 'Buffer size is a positive integer.'
                self.BUFFER = self.DATASETS[0:buffer]
                self.currLocation = bufferSize

            else:
                sys.exit('ERROR: Buffer needs to be a positive integer.')
        
        #TODO: Read data from serial port.
        elif mode == 'listen':
            sys.exit('WARNING: Mode to be defined.')
            pass

        else:
            sys.exit('ERROR: Mode not defined.')
            
        #Instantiate an estimator class.
        self.Estimator = Estimators(BUFFER=self.BUFFER,estimator=estimatorType)


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
        self.Estimator.run()
        

    #VALIDATION
    def scores(self):
        pass

    def inspect(self):
        pass


    #HLPER FUNCTIONS

    #Update buffer
    def update(self,step=1):

        if mode == 'read': 
            for i in range(step):
                self.BUFFER.pop(0)
                self.BUFFER.append(self.DATASETS[self.curr])
            Estimator.update(self.BUFFER)

        else:  
            sys.exit('WARNING: Mode to be defined.') #Listen to serial
