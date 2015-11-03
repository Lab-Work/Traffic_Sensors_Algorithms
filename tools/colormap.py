"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu (fwu10@illinois.edu)
Date: September 15th, 2015

Visualize data in form of colormap.
////////////////////////////////////////////////////////////////////////////"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
import cookb_signalsmooth as ss 

"""
Scan through the whole data space and take a snapshot of data every interval
amount of time.

@IN: an existing dataset supplied in lists form:
    PIR_data: PIR data 
    IMUU_reduced_data: frequency-reduced IMU and Uson data
    LOG_inflated_data: frequency-inflated label data
@OUT: consecutive colormap plots
_____________________________________________________________________________"""
def colormap_scan(PIR_data, IMUU_reduced_data, LOG_inflated_data, interval = 960):
    for t in np.arange(0,len(PIR),interval):
        colormap(PIR_data[t:t+interval], 
                 IMUU_reduced_data[t:t+interval], 
                 LOG_inflated_data[t:t+interval], 
                 t, save_fig=True)

"""
Specify synchronized PIR data, IMUU data, and LOG data and plot them in
colormap. To save the plots, set savefig=True. To display the plots, set
savefigs=False. By default, savefig is set to True. To apply smoothing, set
smooth=True.

@IN: an existing datasets supplied in lists form
@OUT: the corresponding colormap either display in gui or save to current
directory
_____________________________________________________________________________"""
def colormap(PIR_data, IMUU_reduced_data, LOG_inflated_data, label,
        save_fig=False, smooth=False):
    print "Generating colormaps"
    MEAN = []
    for i in range(len(PIR_data[0])):
        MEAN.append(np.mean([t[i] for t in PIR_data]))
    for i in range(6):
        MEAN.append(np.mean(IMUU_reduced_data))
    for i in range(6):
        MEAN.append(0)
    MEAN = np.array(MEAN)
    STDEV = []
    for i in range(len(PIR_data[0])):
        STDEV.append(np.std([t[i] for t in PIR_data]))
    for i in range(6):
        STDEV.append(np.std(IMUU_reduced_data))
    for i in range(6):
        STDEV.append(1)
    STDEV = np.array(STDEV)

    
    #plt.figure(figsize=(15,10), dpi=150)
    colormap_row = []
    for x,y,z in zip(PIR_data, IMUU_reduced_data, LOG_inflated_data):
        colormap_row.append((np.array(x+[y]*6+[z]*6)-MEAN)/STDEV)
    colormap_row = np.array(colormap_row)
    colormap_row = np.transpose(colormap_row)

    # Uncomment to generate colormap in row major
    '''
    plt.imshow(colormap_row, origin="lower", cmap=plt.get_cmap("jet"), aspect="auto",
               interpolation="nearest", vmin=-2, vmax=8)
    plt.colorbar(orientation="horizontal")
    plt.title("Colormap (Row Major) from the Data Collected on 09/03/15")
    plt.ylabel("Normalized Signal from PIR and Uson")
    plt.xlabel("Elapsed Time (0.125 sec)")
    if (save_fig):
        plt.savefig("./visualization/colormaps_row/"+"{:02}".format(label))
        plt.close()
    else:
        plt.show()
    '''
    
    plt.figure(figsize=(15,10), dpi=150)
    column_major = np.array([[[i*64+k*16+j for k in range(4)] for j in range(16)]
        for i in range(3)]).reshape(192)
    column_major = np.append(column_major, [192+i for i in range(12)])
    colormap_col = []
    colormap_row_t = np.transpose(colormap_row)
    for line in colormap_row_t:
        col = []
        for i in column_major:
            col.append(line[i])
        colormap_col.append(col)
    colormap_col = np.array(colormap_col)
    colormap_col = np.transpose(colormap_col)
    plt.imshow(colormap_col, origin="lower", cmap=plt.get_cmap("jet"), aspect="auto",
               interpolation="nearest", vmin=-2, vmax=8)
    plt.colorbar(orientation="horizontal")
    plt.title("Colormap (Col Major) from the Data Collected on 09/03/15")
    plt.ylabel("Normalized Signal from PIR and Uson")
    plt.xlabel("Elapsed Time (0.125 sec)")
    if (save_fig):
        plt.savefig("./visualization/colormaps_col/"+"{:02}".format(label))
        plt.close()
    else:
        plt.show()


    # Apply Guassian filter for data smoothing. This step is necessary to
    # obtain stable cross correlation results.
    if smooth:
        # Smoothing colormap_row
        plt.figure(figsize=(15,10), dpi=150)
        colormap_row = colormap_row[:-12]
        colormap_row_blur = ss.blur_image(colormap_row, 3)
        plt.imshow(colormap_row_blur, origin="lower", cmap=plt.get_cmap("jet"), aspect="auto",
                   interpolation="nearest", vmin=-2, vmax=8)
        plt.colorbar(orientation="horizontal")
        plt.title("Colormap (Row Major) from the Data Collected on 09/03/15")
        plt.ylabel("Normalized Signal from PIR and Uson")
        plt.xlabel("Elapsed Time (0.125 sec)")
        
        # Smoothing colormap
        plt.figure(figsize=(15,10), dpi=150)
        colormap_col = colormap_col[:-12]
        colormap_col_blur = ss.blur_image(colormap_col, 5)
        plt.imshow(colormap_col_blur, origin="lower", cmap=plt.get_cmap("jet"), aspect="auto",
                   interpolation="nearest", vmin=-2, vmax=8)
        plt.colorbar(orientation="horizontal")
        plt.title("Colormap (Col Major) from the Data Collected on 09/03/15")
        plt.ylabel("Normalized Signal from PIR and Uson")
        plt.xlabel("Elapsed Time (0.125 sec)")
        plt.show()
