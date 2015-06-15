import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Estimators:

    def __init__(self,BUFFER,estimator):

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
        self.t0 = timeFormat(BUFFER[0][0])
        for data in BUFFER:
            self.time = np.append(self.time,timeFormat(data[0]))
            self.elapse = np.append(self.elapse,self.t0-timeFormat(data[0]))

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

    def run(self):
        pass

    def updateBuffer(self,BUFFER):
        pass

    def adaptiveThreshold():
        pass
