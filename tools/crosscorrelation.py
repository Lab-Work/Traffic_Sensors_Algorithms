"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu
Date: September 16th, 2015
A script that contains cross-correlation method.
////////////////////////////////////////////////////////////////////////////"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
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
        return [positive_delay, np.argmax(positive_delay)]
    else:
        return [negative_delay, -np.argmax(negative_delay)]

"""
Find and visualize the delay between the pixels of the PIR arrays.
___________________________________________________________________"""
def find_pixels_shifts(PIR_data, window=[965,985,0,15], col_major=True, norm=True,
        smooth=True, time_cc=True, center_cc=False, step=4):
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
        WINDOW = ss.blur_image(WINDOW,3)
    
    if time_cc:
        WINDOW = np.transpose(WINDOW)
    
        DELAY = []
        plt.figure()
        n = len(WINDOW[:,1])
        print "Number of pixels analysed: %s" %str(n)
        
        if center_cc:
            for i in range(n):
                if i != n/2:
                    plt.plot(WINDOW[i,:], label="Pixel %s" %str(i))
                DELAY.append(find_delay(WINDOW[i,:], WINDOW[n/2,:],
                    col_shift=False))
            plt.plot(WINDOW[n/2,:], 'r', linewidth=2.5, label="Pixel %s" %str(n/2))
        else:
            frames = np.arange(0,n-1,step)
            for i,j in zip(frames[0:-1], frames[1:]):
                plt.plot(WINDOW[i,:], label="Pixel %s" %str(i))
                DELAY.append(find_delay(WINDOW[i,:],
                    WINDOW[j,:],col_shift=False))
            plt.plot(WINDOW[j,:], label="Pixel %s" %str(j))

        if n <= 16:
            plt.legend()
        plt.title("Time Series of PIR Pixels Signals")
        plt.xlabel("Elapsed Time (0.125 sec)")
        plt.ylabel("Normalizaed Quantity")
    
        plt.figure()
        plt.plot([x[1] for x in DELAY])
        plt.plot([0]*len(DELAY), '--k')
        plt.title("Calculated Time Delays for Each Pixel")
        plt.xlabel("Nth Pixel")
        plt.ylabel("Time Delay (0.125 sec)")
        if len(DELAY) <= 16:
            i = 0
            for x in DELAY:
                plt.figure()
                plt.plot(x[0])
                plt.title("Cross-Correlation between Pixels (%d)" % i)
                i += 1
    
    else:
        DELAY = []
        plt.figure()
        n = len(WINDOW[:,1])
        print "Number of instances analysed: %s" %str(n)
        
        if center_cc:
            for i in range(n):
                if i != n/2:
                    plt.plot(WINDOW[i,:], label="Instances %s"
                            %str(i),col_shift=True)
                DELAY.append(find_delay(WINDOW[i,:], WINDOW[n/2,:]))
            plt.plot(WINDOW[n/2,:], 'r', linewidth=2.5, label="Instances %s" %str(n/2))
        else:
            frames = np.arange(0,n-1,step)
            for i,j in zip(frames[0:-1], frames[1:]):
                plt.plot(WINDOW[i,:], label="Instances %s" %str(i))
                DELAY.append(find_delay(WINDOW[i,:], WINDOW[j,:],
                    col_shift=True))
            plt.plot(WINDOW[j,:], label="Instances %s" %str(j))
        
        if n <= 16:
            plt.legend()
        plt.title("Evolution of PIR Signals in Space")
        plt.xlabel("Pixels")
        plt.ylabel("Normalizaed Quantity")
        
        plt.figure()
        plt.plot([x[1] for x in DELAY])
        plt.plot([0]*len(DELAY), '--k')
        plt.title("Calculated Space Movement among Instances")
        plt.xlabel("Nth Instances")
        plt.ylabel("Space Shift")
        if len(DELAY) <= 16:
            i = 0
            for x in DELAY:
                plt.figure()
                plt.plot(x[0])
                plt.title("Cross-Correlation between Instances (%d)" % i)
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
