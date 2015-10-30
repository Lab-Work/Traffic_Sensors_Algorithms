"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu
Date: September 16th, 2015
A script that contains cross-correlation method.
////////////////////////////////////////////////////////////////////////////"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
from matplotlib import cm
import cookb_signalsmooth as ss 

"""
Crosscorrelation utility functions that return the cross correlation of already
normalized time series X and Y given a time delay delta. In addition, X and Y 
should be in form of list and equal in length, and delta should be an integer. 
______________________________________________________________________________"""
def crosscorrelation(X, Y, delta, col_shift):
    assert type(delta) is int, "Delay is not an integer: %s" %str(delta)
    if col_shift:
        N = len(Y) - 4*delta
        if N == 0:
            return 0
        cc = 0
        for i in range(N):
            cc += X[i]*Y[i+4*delta]
    else:
        N = len(Y) - delta
        if N == 0:
            return 0
        cc = 0
        for i in range(N):
            cc += X[i]*Y[i+delta]
    return cc/N

"""
A utility function that returns the most plausible delay between two signals
using the method of cross-correlation.
______________________________________________________________________________"""
def find_delay(X, Y, col_shift):
    if len(X) > len(Y):
        for i in range(len(X)-len(Y)):
            Y.append(0)
    elif len(X) < len(Y):
        for i in range(len(Y)-len(X)):
            X.append(0)
    if col_shift:
        positive_delay = []
        for i in range(len(Y)/4):
            positive_delay.append(crosscorrelation(X,Y,i, col_shift))
        negative_delay = []
        for i in range(len(X)/4):
            negative_delay.append(crosscorrelation(Y,X,i, col_shift))
    else:
        positive_delay = []
        for i in range(len(Y)):
            positive_delay.append(crosscorrelation(X,Y,i, col_shift))
        negative_delay = []
        for i in range(len(X)):
            negative_delay.append(crosscorrelation(Y,X,i, col_shift))

    if max(positive_delay) > max(negative_delay):
        if np.max(positive_delay) < 0.25:
            return [positive_delay, np.argmax(positive_delay),
                    np.max(positive_delay)]
            #return [positive_delay, -1,
            #        np.max(positive_delay)]
        else:
            return [positive_delay, np.argmax(positive_delay),
                    np.max(positive_delay)]
    else:
        if np.max(negative_delay) < 0.25:
            return [negative_delay, -np.argmax(negative_delay),
                    np.max(negative_delay)]
            #return [negative_delay, -1,
            #        np.max(negative_delay)]
        else:
            return [negative_delay, -np.argmax(negative_delay),
                    np.max(negative_delay)]

"""
Find and visualize the delay between the pixels of the PIR arrays.
___________________________________________________________________"""
def find_pixels_shifts(PIR_data, window=[965,985,0,15], col_major=True, norm=True,
        smooth=True, time_cc=True, center_cc=False, step=3, diff=False):
    print "Finding PIR delays..."
    WINDOW = np.array(PIR_data[window[0]:window[1]+1])
    WINDOW = WINDOW[:,window[2]:window[3]+1]
    
    if col_major and window[3] == 191:
        column_major = np.array([[[i*64+k*16+j for k in range(4)] for j in range(16)]
            for i in range(3)]).reshape(192)
        colormap_col = []
        for line in WINDOW:
            col = []
            for i in column_major:
                col.append(line[i])
            colormap_col.append(col)
        WINDOW = np.array(colormap_col)
   
    
    if norm:
        MEAN = []
        for i in range(len(WINDOW[0])):
            MEAN.append(np.mean([t[i] for t in WINDOW]))
        MEAN = np.array(MEAN)
        STDEV = []
        for i in range(len(WINDOW[0])):
            STDEV.append(np.std([t[i] for t in WINDOW]))
        STDEV = np.array(STDEV)
        WINDOW = (WINDOW - MEAN)/STDEV
    if smooth:
        #print WINDOW.shape
        WINDOW = ss.blur_image(WINDOW,3)
        #print WINDOW.shape
    
    if time_cc:
        WINDOW = np.transpose(WINDOW)
    
        DELAY = []
        n = len(WINDOW[:,1])
        print "Number of pixels analysed: %s" %str(n)
        
        if center_cc:
            for i in range(n):
                DELAY.append(find_delay(WINDOW[i,:], WINDOW[n/2,:],
                    col_shift=False))
        else:
            frames = np.arange(0,n-1,step)
            for i,j in zip(frames[:-1], frames[1:]):
                DELAY.append(find_delay(WINDOW[i,:],
                    WINDOW[j,:],col_shift=False))
    
        if diff:
            plt.figure()
            plt.plot([x[1] for x in DELAY])
            plt.plot([0]*len(DELAY), '--k')
            plt.title("Calculated Time Delays for Each Pixel")
            plt.xlabel("Nth Pixel")
            plt.ylabel("Time Delay (0.125 sec)")
            
            plt.figure()
            plt.imshow(WINDOW, interpolation="nearest")
            plt.title("Colormap of the Data")
            plt.show()
            
            return (DELAY[2][1] - DELAY[-3][1])*0.125
        else:
            plt.figure()
            plt.plot([x[1] for x in DELAY])
            plt.plot([0]*len(DELAY), '--k')
            plt.title("Calculated Time Delays for Each Pixel")
            plt.xlabel("Nth Pixel")
            plt.ylabel("Time Delay (0.125 sec)")
            plt.show()

    else:
        DELAY = []
        n = len(WINDOW[:,1])
        print "Number of instances analysed: %s" %str(n)
        
        frames = np.arange(0,n-1,step)
        for i,j in zip(frames[:-1], frames[1:]):
            DELAY.append(find_delay(WINDOW[i,:], WINDOW[j,:],
                         col_shift=True) + [i])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        X = range(len(WINDOW[0,:])/4)
        Y = [x[3] for x in DELAY]
        Z = np.ones((len(Y),len(X)))
        for i in range(len(Y)):
            Z[i,:] = DELAY[i][0]
        X, Y = np.meshgrid(X, Y)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=100, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        #ax.plot([x[1] for x in DELAY],
        #        [x[3] for x in DELAY],
        #        [x[2] for x in DELAY])
        ax.set_xlabel("Column Shift")
        ax.set_ylabel("Time (Frames)")
        ax.set_zlabel("Correlation Value")
        
        if False:
            i = 0
            for x in DELAY:
                plt.figure()
                plt.plot(x[0])
                plt.title("Cross-Correlation between Frames (%d)" % i)
                i += 1

    plt.figure()
    plt.imshow(WINDOW, interpolation="nearest")
    plt.title("Colormap of the Data")
    plt.show()

def cc_demo():
    SIN = []
    plt.figure()
    for n in np.arange(0,1,0.1):
        SIN.append([np.sin(x-n) for x in np.arange(0,2*np.pi,0.05)])
        plt.plot(SIN[-1])
    plt.plot(SIN[4],'k',linewidth=2.5)
    plt.title("A Series of Shifted Sine Signals")
    plt.xlabel("Time")
    plt.ylabel("Signal")

    SIN_delay = []
    for func in SIN:
        SIN_delay.append(find_delay(func, SIN[4]))
    plt.figure()
    plt.plot(SIN_delay)
    plt.title("Cross-correlation Demo")
    plt.xlabel("Nth Sine Curve")
    plt.ylabel("Delay w/ Respect to the 5th Sine")
    plt.show()
