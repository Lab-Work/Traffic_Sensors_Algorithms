'''
Author: Fangyu Wu
Date: 04/05/2015

The script contains the definition of the PIR temperature class.
'''

import csv
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import matplotlib.pyplot as plt
from datetime import datetime
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True)

class PIRTemp:
    
    def __init__(self, filename):
        file = open(filename, 'r')
        dataset = csv.reader(file)

        self.time = []
        self.isPass = []
        self.speed = []
        self.temp = []
        self.ambTemp = []
        self.count = []
        self.window = []
        self.array64 = []
        self.array16 = []
        self.array8 = []
        self.array4 = []

        for line in dataset:
            self.time.append(line[0])
            self.isPass.append(line[1])
            self.speed.append(line[2])
            self.temp.append(line[3:67])
            self.ambTemp.append(line[67])
            self.count.append(line[68])

        file.close()

        self.time = [datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f') for line in self.time]
        self.isPass = [int(i) for i in self.isPass]
        self.speed = [float(i) for i in self.speed]
        self.temp = [[float(i) for i in line] for line in self.temp]
        self.ambTemp = [float(i) for i in self.ambTemp]
        self.count = [int(i) for i in self.count]

        sample = [i for i in range(len(self.time)) if self.isPass[i]]
        ends = [i+1 for i in range(len(sample)-1) if sample[i+1]-sample[i] > 3]
        ends.insert(0,0)
        ends.append(len(sample))
        for i in range(len(ends)-1):
            self.window.append(sample[ends[i]:ends[i+1]])
        self.window = np.array(self.window)

        self.array64 = np.array([np.fliplr(np.array(line).reshape((4,16))) for line in self.temp])
        self.array16 = np.array([sum(line) for line in self.array64])
        self.array8 = np.array([[sum(line[0:2]),sum(line[2:4]),
                            sum(line[4:6]),sum(line[6:8]),
                            sum(line[8:10]),sum(line[10:12]),
                            sum(line[12:14]),sum(line[14:16])] 
                           for line in self.array16])
        self.array4 = np.array([[sum(line[0:2]),sum(line[2:4]),
                            sum(line[4:6]),sum(line[6:8])] 
                           for line in self.array8])

    def logisticRegression(self,window_len=8,window_type='hamming'):

        clf = linear_model.LogisticRegression()
        scores1 = cross_validation.cross_val_score(clf, self.temp, self.isPass, cv=10)
        print 'Naive Logistic Regression Accuracy: %.2f +/- %.2f' % (scores1.mean(), scores1.std()*2)
                
        temp0 = np.array(self.temp).reshape(len(self.temp),len(self.temp[0]))
        #plt.plot(temp0[:,0])
        temp = np.zeros((len(temp0)-window_len+1,len(temp0[0])))
        for i in range(len(self.temp[0])):
            temp[:,i] = smooth(temp0[:,i])
        isPass = np.array(self.isPass)[window_len/2-1:len(self.isPass)-window_len/2]
        #plt.plot(range(window_len/2-1,len(temp)+window_len/2-1),temp[:,0])
        #plt.show()

        scores2 = cross_validation.cross_val_score(clf, temp, isPass, cv=10)
        print 'Windowed Logistic Regression Accuracy: %.2f +/- %.2f' % (scores2.mean(), scores2.std()*2)

        plt.figure()
        clf.fit(temp, isPass)
        prediction = clf.predict(temp)
        plt.plot(range(1,len(isPass)+1), isPass, range(1,len(temp)+1), prediction+1)
        plt.show()

        
    def linearRegression(self):
        clf = linear_model.LinearRegression()
        scores1 = cross_validation.cross_val_score(clf, self.temp, self.speed, cv=10)
        print 'Naive Linear Regression Accuracy: %.2f +/- %.2f' % (scores1.mean(), scores1.std()*2)
        
        #plt.plot(range(1,len(self.time)+1),self.temp)
        #plt.plot(self.ambTemp)
        #plt.plot(range(1,len(self.time)+1),self.array16)
        #plt.plot(range(1,len(self.time)+1),self.array8)
        #plt.plot(range(1,len(self.time)+1),self.array4)
        #plt.show()

        arrayWindow = np.array([np.array([np.array(self.array4[t]) for t in window]) for window in self.window])
        #print len(self.window)
        timeStamp = np.array([np.argmax(line, axis=0) for line in arrayWindow])
        timeStamp = np.array([self.window[i][0] + timeStamp[i] for i in range(len(timeStamp))])
        timeStamp = np.array([[self.time[t] for t in line] for line in timeStamp])
        #print timeStamp
        timeDiff = np.array([[j-i for i, j in zip(k[:-1], k[1:])] for k in timeStamp])
        timeDiff = np.array([[td.total_seconds() for td in line] for line in timeDiff])
        timeDiff = np.array([[abs(1.0/i) for i in line] for line in timeDiff])
        #print timeDiff
        speed = np.array([np.array([np.array(self.speed[t]) for t in window])[0] for window in self.window])
        #print speed

        #scores2 = cross_validation.cross_val_score(clf, timeDiff, speed, cv=10)
        #print 'Linear Regression with Time Shift Accuracy: %.2f +/- %.2f' % (scores2.mean(), scores2.std()*2)

        #clf.fit(timeDiff, speed)
        #prediction = clf.predict(timeDiff)
        #plt.plot(range(len(speed)), speed, range(len(speed)), prediction)
        #plt.show()


    ### HELPER METHODS ###
    def updateSpeedWindows(self):
        sample = [i for i in range(len(self.time)) if self.isPass[i]]
        ends = [i+1 for i in range(len(sample)-1) if sample[i+1]-sample[i] > 3]
        ends.insert(0,0)
        ends.append(len(sample))
        for i in range(len(ends)-1):
            self.window.append(sample[ends[i]:ends[i+1]])

    def display(self, linNum):
        print 'Timestamp: ', self.time[linNum]
        print 'Is passing: ', self.isPass[linNum]
        print 'Speed: ', self.speed[linNum]
        print 'Temperature readings: ', self.array64[linNum]
        print 'Ambient temperature: ', self.ambTemp[linNum]
        print 'Count #: ', self.count[linNum]

    def timeSeries(self):
        fig = plt.figure(figsize=(16,8), dpi=100)
        plt.plot(self.time,self.temp)
        plt.plot(self.time,self.ambTemp)
        plt.title('Time Series of PIR Signal')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time stamp')
        fig.savefig('timeseries')

    def heatMap(self):
        for f in range(len(self.array64)):
            fig = plt.figure(figsize=(16,8), dpi=100)
            im = plt.imshow(self.array64[f],cmap=plt.get_cmap('jet'),interpolation='nearest')
            plt.colorbar(im,orientation="horizontal")
            plt.title('Heat Map of PIR Signal at $t=$ '+ self.time[f].strftime('%Y-%m-%d %H:%M:%S'))
            fig.savefig('/home/mianao/Dropbox/Courses/498C/Paper/heatMap/'+'{:06}'.format(f))
            f += 1
            plt.close(fig)

    def heatTopo(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        #maxTemp = 0
        #minTemp = 0
        #for line in self.temp:
        #    maxTemp = max(maxTemp,max(line))
        #    minTemp = max(minTemp,min(line))
        #print maxTemp
        #print minTemp

        for f in range(len(self.array64)):
            fig = plt.figure(figsize=(20,8), dpi=100)
            ax = fig.gca(projection='3d')
            X = np.arange(1,17)
            Y = np.arange(1,5)
            X, Y = np.meshgrid(X, Y)
            Z = self.array64[f]
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=25, vmax=32)
            ax.set_zlim(25, 32)
            fig.colorbar(surf)
            plt.title('Heat Topology of PIR Signal at $t=$ '+ self.time[f].strftime('%Y-%m-%d %H:%M:%S'),y=1.08)
            ax.set_xlabel('Horizontal direction')
            ax.set_ylabel('Vertical direction')
            ax.set_zlabel('Temperature ($^{\circ}C$)')
            fig.savefig('/home/mianao/Dropbox/Courses/498C/Paper/heatTopo/'+'{:06}'.format(f))
            plt.close(fig)

### HELPER FUNCTIONS ###
def smooth(x,window_len=8,window='hamming'):
    #Credit to http://wiki.scipy.org/Cookbook/SignalSmooth

    if x.ndim == 0:
        raise ValueError, 'Empty array!'
    if x.size < window_len:
        raise ValueError, 'Input vector needs to be bigger than window size.'
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'."

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = [np.dot(x[i:i+window_len],w/w.sum()) for i in range(len(x)-window_len+1)]

    return y

'''    
@TODO: realtime visualization
    def measure(self, timeseries=False, heatmap=False, heattopo=False):
        if timeseries:
            PIRTemp.TimeSeries(self)
        if heatmap:
            PIRTemp.HeatMap(self)
        if heattopo:
            PIRTemp.HeatTopo(self)

    def TimeSeries(self):
        
    def HeatMap(self):
        
    def HeatTopo(self):
'''    
