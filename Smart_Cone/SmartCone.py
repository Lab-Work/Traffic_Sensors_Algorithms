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
from scipy.signal import argrelextrema
import datetime
import time
from sklearn import linear_model
from sklearn import cross_validation
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True) # This will create LaTex style plotting.
from Estimators import Estimators

#Primary class
class SmartCone: 

    #LIABRARY
    def __init__(self,mode='read',bufferSize='inf',estimatorType=None):

        #Use timeDiff to adjust time discrepancy
        self.timeDiff = datetime.timedelta(hours=21,minutes=39,seconds=15)
        if mode == 'read':

            #Get the relative paths of data in data/.
            self.FILES = []
            for root, dirs, files in os.walk('data/'):
                for file in sorted(files):
                    if file.endswith(".csv"):
                        self.FILES.append(os.path.join(root, file))
            print self.FILES
            
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
                print 'Buffer size is set to infinitive.'
                self.BUFFER = self.DATASETS

            elif isinstance(bufferSize,int) and bufferSize > 0:
                print 'Buffer size is %d.' % bufferSize
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
            
        #Instantiate an estimator class. Comment the code while testing to improve speed.
        self.Estimator = Estimators(BUFFER=self.BUFFER,estimator=estimatorType)


    #VISUALIZATION
    def timeSeries(self,parameter='mean'):

        if parameter == 'mean':
            #plt.ion()
            print 'Plotting mean...'
            pir2Mean = [np.average([float(i) for i in line[69:133]]) for line in self.DATASETS]
            pir2Mean = np.array(pir2Mean)
            mean = np.average(pir2Mean)
            std = np.std(pir2Mean)
            print mean
            print std
            #pir2Mean = np.array([(i-mean)/std for i in pir2Mean])
            #outliers = [i for i in range(len(pir2Mean)) if pir2Mean[i] > std]
            #print outliers
            timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
            timeStamp = np.array(timeStamp)
            plt.scatter(timeStamp,pir2Mean,marker='.',color='b')
            #plt.scatter(timeStamp[outliers], pir2Mean[outliers],marker='o',color='r')

            try:
                instances = [np.average([float(i) for i in line[69:133]]) 
                             for line in self.DATASETS if line[-1] == 1]
                instancesTime = [self.timeFormat(float(line[0]))+self.timeDiff 
                                 for line in self.DATASETS if line[-1] == 1]
                plt.scatter(instancesTime,instances,marker='o',color='r')
            except:
                print 'Labels not found.'

            plt.xlim([timeStamp[0],timeStamp[-1]])
            plt.xlabel('Time stamp')
            plt.ylabel('Mean of 64 pixels')
            plt.show(block=False)
            print 'Done.'
            
        elif parameter == 'variance':
            print 'Plotting variance...'
            pir2Var = [np.var([float(i) for i in line[69:133]]) for line in self.DATASETS]
            timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
            plt.plot(timeStamp, pir2Var)
            plt.xlabel('Time stamp')
            plt.ylabel('Variance of 64 pixels')
            plt.show(block=False)
            print 'Done.'

        elif parameter == 'difference':
            print 'Plotting difference...'
            pir2Mean = [np.average([float(i) for i in line[69:133]]) for line in self.DATASETS]
            pir2Diff = [0]
            for i in range(1,len(pir2Mean)):
                pir2Diff.append(pir2Mean[i] - pir2Mean[i-1])
            timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
            plt.plot(timeStamp,pir2Diff)
            plt.xlabel('Time stamp')
            plt.ylabel('Change of the mean of 64 pixels in time')
            plt.show(block=False)
            print 'Done.'

        elif parameter == 'ultrasonic':
            print 'Plotting ultrasonic sensor data...'
            ultrasonic = [line[134] for line in self.DATASETS]
            timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
            plt.plot(timeStamp, ultrasonic)
            plt.xlabel('Time stamp')
            plt.ylabel('Ultrasonic sensor reading')
            plt.show(block=False)
            print 'Done.'

        elif parameter == 'time': #Check the order and uniformity of the time data
            print 'Plotting time over time stamp...'
            time = [float(line[0]) for line in self.DATASETS]
            unitTime = [time[t]-time[t-1] for t in range(1,len(time))]
            timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
            print len(time)
            print len(unitTime)
            print len(timeStamp)
            fig1 = plt.figure(1)
            plt.plot(timeStamp,time)
            plt.xlabel('Time Stamp')
            plt.ylabel('Ordinal Time')
            fig2 = plt.figure(2)
            plt.plot(timeStamp[0:len(time)-1],unitTime)
            plt.xlabel('Time Stamp')
            plt.ylabel('Ordinal Time per Increment')

        else:
            sys.exit('ERROR: Parameter not defined.')
        plt.show()

    def meanTempHist(self):
        print 'Plotting mean temperature histogram...'
        pir2Mean = [np.average([float(i) for i in line[69:133]]) for line in self.DATASETS]
        pir2Mean = np.array(pir2Mean)
        #locMax = [argrelextrema(pir2Mean, np.greater)[0]]
        timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
        timeStamp = np.array(timeStamp)
        
        '''
        plt.figure()
        H = np.histogram(pir2Mean[locMax],bins=300)
        print len(H[1])
        h = H[0]
        plt.plot(H[1][:-1],h)
        hMax = np.argmax(h)
        norm0 = range(hMax)
        norm1 = range(hMax+1,2*hMax+1)[::-1]
        h[norm1] = h[norm1] - h[norm0]
        h[norm0] = 0
        h[hMax] = 0
        hMax1 = np.argmax(h)
        plt.plot(H[1][:-1],h)
        print (hMax1-hMax)*(H[1][1]-H[1][0])
        '''

        fig1 = plt.figure(1)
        plt.hist(pir2Mean,bins=250,histtype='stepfilled',color='r',label='All')
        #plt.hist(pir2Mean[locMax],bins=250,histtype='stepfilled',color='b',label='Peak')
        plt.xlabel('Temperature (C)')
        plt.ylabel('Point Count')
        plt.legend()
        
        try:
            fig2 = plt.figure(2)
            instances = np.array([np.average([float(i) for i in line[69:133]]) 
                                  for line in self.DATASETS if line[-1] == 1])
            instancesTime = np.array([self.timeFormat(float(line[0]))+self.timeDiff 
                                      for line in self.DATASETS if line[-1] == 1])
            locMax = [argrelextrema(instances, np.greater)[0]]
            plt.hist(list(set(pir2Mean)-set(instances)),bins=250,histtype='step',color='g',label='Noise')
            plt.hist(instances,bins=250,histtype='step',color='b',label='Labelled')
            plt.hist(instances[locMax],bins=250,histtype='step',color='r',alpha=0.5,label='Labelled Peak')
            plt.xlabel('Temperature (C)')
            plt.ylabel('Point Count')
            plt.legend()
        except:
            plt.close(fig2)
            print 'Labels not found.'


        plt.show(block=False)
        print 'Done.'
        plt.show()

    def heatMap(self,fps=8,saveFig=True):
        if not saveFig:
            print 'Note that the current version does not support title time stamp update!'

        pir2 = np.array([[float(i) for i in line[69:133]] for line in self.DATASETS])
        pir2 = np.array([np.flipud(line.reshape(4,16,order='F')) for line in pir2])
        timeStamp = [self.timeFormat(float(line[0]))+self.timeDiff for line in self.DATASETS]
        timeStamp = np.array(timeStamp)

        pir2Max = np.amax(pir2)
        pir2Min = np.amin(pir2)
        pir2MaxAvg = np.amax(np.average(pir2))
        pir2MinAvg = np.amin(np.average(pir2))
        pir2MaxVar = np.amax(np.var(pir2))
        pir2MinVar = np.amin(np.var(pir2))


        fig,(ax1,ax2) = plt.subplots(2, dpi=100)
        #ax2.set_xlim(pir2MinAvg,pir2MaxAvg)
        #ax2.set_ylim(pir2MinVar,pir2MaxVar)
        ax2.set_xlim(29,39)
        ax2.set_ylim(0,20)


        background1 = fig.canvas.copy_from_bbox(ax1.bbox)
        background2 = fig.canvas.copy_from_bbox(ax2.bbox)
        im = ax1.imshow(pir2[0],cmap=plt.get_cmap('jet'),aspect='auto',
                       interpolation='nearest',vmin=pir2Min,vmax=pir2Max-50)
        position=fig.add_axes([0.93,0.536,0.02,0.362])
        fig.colorbar(im,cax=position)
        ax1.set_title('Heat Map of PIR Signal at $t=$ '+ 
                     timeStamp[0].strftime('%Y-%m-%d %H:%M:%S'))
        pt, = ax2.plot(np.average(pir2[0]),np.var(pir2[0]),marker='x')
        ax2.set_title('Frame Mean/Var Correlation at $t=$ '+ 
                      timeStamp[0].strftime('%Y-%m-%d %H:%M:%S'))
        ax2.set_xlabel('Mean')
        ax2.set_ylabel('Variance')
        fig.show()
        fig.canvas.draw()

        for f in range(1,len(pir2)):
            time.sleep(1./fps)
            im.set_data(pir2[f])
            ax1.set_title('Heat Map of PIR Signal at $t=$ '+ 
                     timeStamp[f].strftime('%Y-%m-%d %H:%M:%S'))
            pt.set_data(np.average(pir2[f]),np.var(pir2[f]))
            #pt, = ax2.plot(np.average(pir2[f]),np.var(pir2[f]))
            ax2.set_title('Frame Mean/Var Correlation at $t=$ '+ 
                          timeStamp[f].strftime('%Y-%m-%d %H:%M:%S'))
            fig.canvas.restore_region(background1)
            fig.canvas.restore_region(background2)
            ax1.draw_artist(im)
            ax2.draw_artist(pt)
            #ax.get_figure().canvas.draw()
            if saveFig: #Output frames to folder. Users may later combine them into videos.
                fig.savefig('heatMaps_/'+'{:06}'.format(f))
            else:
                #fig.canvas.draw()
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)

        plt.close(fig)

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

    #Change time stamp format to Python datetime format
    def timeFormat(self,datenum):
        return datetime.datetime.fromtimestamp(float(datenum))

    #Update buffer
    def update(self,step=1):

        if mode == 'read': 
            for i in range(step):
                self.BUFFER.pop(0)
                self.BUFFER.append(self.DATASETS[self.curr])
            Estimator.update(self.BUFFER)

        else:  
            sys.exit('WARNING: Mode to be defined.') #Listen to serial

    #Label DATASETS
    def label(self):
    
        print 'Labelling data...'
        #Get the relative paths of data in data/.
        LABELS = []
        for root, dirs, files in os.walk('results/'):
            for file in sorted(files):
                if file.endswith(".csv"):
                    self.LABELS.append(os.path.join(root, file))
        print LABELS

        timeStamp = []
        for label in LABELS:
            with open(label,'r') as labelFile:
                labelReader = csv.reader(labelFile)
                labelReader.next()
                timeStamp = [datetime.datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S.%f') for line in labelReader]
        
        for line in self.DATASETS:
            for time in timeStamp:
                timeDiff = self.timeFormat(float(line[0]))+self.timeDiff-time
                if datetime.timedelta(milliseconds=-2500) < timeDiff < datetime.timedelta(milliseconds=1000):
                    #print timeDiff
                    line.append(1)
                    break
