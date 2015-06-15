'''
Authors: Yanning Li, Fangyu Wu
Date: Summer 2015

This module contains Estimators class, which further includes all the experimental algorithms:
    1) Adapative threshold (baseline algorithm)
    2) ...
'''


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

class Estimators:

    def __init__(self,BUFFER,estimator=None):

        self.estimator = estimator
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
        
        #Initialize the properties defined above.
        self.t0 = self.timeFormat(float(BUFFER[0][0]))
        print self.t0
        for data in BUFFER:
            self.time = np.append(self.time,self.timeFormat(float(data[0])))
            self.elapse = np.append(self.elapse,self.t0-self.timeFormat(float(data[0])))

            self.millis1 = np.append(self.millis1,float(data[1]))
            self.pir1 = np.append(self.pir1,[float(i) for i in data[2:66]])
            self.amb1 = np.append(self.amb1,float(data[66]))
            self.ultra1 = np.append(self.ultra1,float(data[67]))
            
            self.millis2 = np.append(self.millis2,float(data[68]))
            self.pir2 = np.append(self.pir2,[float(i) for i in data[69:133]])
            self.amb2 = np.append(self.amb2,float(data[133]))
            self.ultra2 = np.append(self.ultra2,float(data[134]))

            self.millis3 = np.append(self.millis3,float(data[135]))
            self.pir3 = np.append(self.pir3,[float(i) for i in data[136:200]])
            self.amb3 = np.append(self.amb3,float(data[200]))
            self.ultra3 = np.append(self.ultra3,float(data[201]))

    def run(self):
        if self.estimator == None:
            sys.exit('Please input estimator type.')
        elif self.estimator == 'adaptiveThreshold':
            self.adaptiveThreshold()
        else:
            sys.exit('ERROR: Estimator not defined.')

    def update(self,BUFFER):
        for data in BUFFER:
            self.time = np.append(self.time,self.timeFormat(float(data[0])))
            self.elapse = np.append(self.elapse,self.t0-self.timeFormat(float(data[0])))

            self.millis1 = np.append(self.millis1,float(data[1]))
            self.pir1 = np.append(self.pir1,[float(i) for i in data[2:66]])
            self.amb1 = np.append(self.amb1,float(data[66]))
            self.ultra1 = np.append(self.ultra1,float(data[67]))
            
            self.millis2 = np.append(self.millis2,float(data[68]))
            self.pir2 = np.append(self.pir2,[float(i) for i in data[69:133]])
            self.amb2 = np.append(self.amb2,float(data[133]))
            self.ultra2 = np.append(self.ultra2,float(data[134]))

            self.millis3 = np.append(self.millis3,float(data[135]))
            self.pir3 = np.append(self.pir3,[float(i) for i in data[136:200]])
            self.amb3 = np.append(self.amb3,float(data[200]))
            self.ultra3 = np.append(self.ultra3,float(data[201]))

    def adaptiveThreshold(self):
        print 'Hi there, it is connected!'

    #HLPER FUNCTIONS

    #Change time stamp format to Python datetime format
    def timeFormat(self,datenum):
        return datetime.fromtimestamp(int(datenum))
