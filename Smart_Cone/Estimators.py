'''
Authors: Yanning Li, Fangyu Wu
Date: Summer 2015

This module contains Estimators class, which further includes all the experimental algorithms:
    1) Adapative threshold (baseline algorithm)
    2) ...
'''


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

class Estimators:

    def __init__(self,BUFFER,estimator=None):

        self.timeDiff = timedelta(hours=21,minutes=39,seconds=46,milliseconds=472.497)
        self.estimator = estimator
        #Data parsed into useful form
        self.time = []   #Time stamp
        self.elapse = [] #Time elapsed from the starting time stamp

        self.millis1 = []#The millisecs from the start of Arduino to time of measure
        self.pir1 = []   #The leftmost PIR reading
        self.amb1 = []   #The leftmost PIR ambient temperature reading 
        self.ultra1 = [] #The corresponding ultrasonic sensor reading

        self.millis2 = []#The millisecs from the start of Arduino to time of measure
        self.pir2 = []   #The middle PIR reading
        self.amb2 = []   #The middle PIR ambient temperature reading
        self.ultra2 = [] #The corresponding ultrasonic sensor reading

        self.millis3 = []#The millisecs from the start of Arduino to time of measure
        self.pir3 = []   #The rightmost PIR reading
        self.amb3 = []   #The rightmost PIR ambient temperature reading
        self.ultra3 = [] #The corresponding ultrasonic sensor reading
        
        #Initialize the properties defined above.
        self.t0 = self.timeFormat(float(BUFFER[0][0]))
        print 'Starting measurement at:'
        print self.t0 + self.timeDiff
        for data in BUFFER:
            self.time.append(self.timeFormat(float(data[0])))
            self.elapse.append(self.t0-self.timeFormat(float(data[0])))

            self.millis1.append(float(data[1]))
            self.pir1.append([float(i) for i in data[2:66]])
            self.amb1.append(float(data[66]))
            self.ultra1.append(float(data[67]))
            
            self.millis2.append(float(data[68]))
            self.pir2.append([float(i) for i in data[69:133]])
            self.amb2.append(float(data[133]))
            self.ultra2.append(float(data[134]))

            self.millis3.append(float(data[135]))
            self.pir3.append([float(i) for i in data[136:200]])
            self.amb3.append(float(data[200]))
            self.ultra3.append(float(data[201]))

    def run(self):
        if self.estimator == None:
            sys.exit('Please input estimator type.')
        elif self.estimator == 'adaptiveThreshold':
            self.adaptiveThreshold()
        else:
            sys.exit('ERROR: Estimator not defined.')

    def update(self,BUFFER):
        for data in BUFFER:
            self.time.append(self.timeFormat(float(data[0])))
            self.elapse.append(self.t0-self.timeFormat(float(data[0])))

            self.millis1.append(float(data[1]))
            self.pir1.append([float(i) for i in data[2:66]])
            self.amb1.append(float(data[66]))
            self.ultra1.append(float(data[67]))
            
            self.millis2.append(float(data[68]))
            self.pir2.append([float(i) for i in data[69:133]])
            self.amb2.append(float(data[133]))
            self.ultra2.append(float(data[134]))

            self.millis3.append(float(data[135]))
            self.pir3.append([float(i) for i in data[136:200]])
            self.amb3.append(float(data[200]))
            self.ultra3.append(float(data[201]))

    def adaptiveThreshold(self):
        print 'Hi there, it is connected!'

    #HLPER FUNCTIONS

    #Change time stamp format to Python datetime format
    def timeFormat(self,datenum):
        return datetime.fromtimestamp(int(datenum))
