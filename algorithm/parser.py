"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu, Will Barbour
Date: September 11th, 2015

The generic parser that takes in arbitary number of raw data files and returns 
an ordered list of struct, each of which contains one instance of data. Necess-
ary conversion and specific parsing rules are applied in the process.
////////////////////////////////////////////////////////////////////////////"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
import csv
from bisect import bisect_left
import time
from datetime import datetime
import os
import sys
from collections import namedtuple
sys.path.append('./tools')
from hex_converter import *
import cookb_signalsmooth as ss 

"""
Convert hex PIR data to dec data in Celsius.
Author: Will Barbour
Modified by: Fangyu Wu
______________________________________________"""

def convert_pir():

    converter = PIR3_converter()

    translate_list = []
    for f_name in os.listdir('./raw_data'):
        if f_name[0:6] == 'sensor' and f_name[-4:] == '.txt':
            translate_list.append(f_name)

    print translate_list

    skip_flag = False
    for txt_file in translate_list:

        trial_fname = txt_file
        output_fname = 'processed_' + trial_fname[0:-4] + '.csv'
        print "Output file: ", output_fname
        skip_line_counter = 1
        trans_file = open(trial_fname, 'r')
        csv_file = open(output_fname, 'w')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)

        for line in trans_file:
            if skip_line_counter > 8:
                parse = line.split(',')
                csv_line = []
                sensor = ''
                ts = parse[0]
                csv_line.append(ts)
                try:
                    sensor = parse[1]
                except:
                    print "Line does not correspond to IMU or PIR format"
                    print parse

                if sensor == '_IMU+Uson_':
                    csv_line.append(sensor[1:4])
                    csv_line.append(parse[2][2:])
                    csv_line.append(parse[3][1:])
                    csv_line.append(parse[4][1:-1])
                    csv_line.append(parse[5][2:])
                    csv_line.append(parse[6][1:])
                    csv_line.append(parse[7][1:-1])
                    csv_line.append(parse[8][1:-2])
                    #print csv_line

                else:
                    if sensor == '_PIR_':
                        csv_line.append(sensor[1:4])
                        #print parse[2]
                        if parse[2][:-1] == '-1' or len(parse[2])<800:
                            csv_line.append(-1)
                        else:
                            #print parse[2][2:-2]
                            csv_line += converter.convert(parse[2][2:-2])
                            #csv_line.append(PIR3_converter.convert(parse[3]))
                    else:
                        print "Second index read error"
                        skip_flag = True
                        
                if not skip_flag:
                    csv_writer.writerow(csv_line)
                else:
                    skip_flag = False
            else:
                skip_line_counter += 1

        csv_file.close()
        print csv_file.name, " conversion is completed."
        #time.sleep(5)


"""
Gramma of the data
    1) Raw data files are named as per "sensor*.txt"
    2) Processed data files are named as per "csv*.csv"
    3) IMU + rangefinder:
        timestamp,IMU,mag-x,mag-y,mag-z,accel-x,accel-y,accel-z,ultrasonic
    4) PIR:
        timestamp,PIR,pir1_milsec,pir2_milsec,pir3_milsec,pir1_ptat,pir2_ptat,pir3_ptat,
        pixel#0_degC,pixel#1_degC,...,pixel191_degC
Parse converted data into an ordered list of structs in time
Return or save to file
_____________________________________________________________________________________________"""

# A short format routine that converts unix timestamp to python datetime
def parse_time(datestr):
    return datetime.strptime(datestr, "%H_%M_%S_%f").replace(year=2015,
            month=9, day=3)

def parse(date="090315"):

    # Import PIR data to a struct (namedtuple)
    PIR_instance = namedtuple(
        "PIR_instance",
        """begin
           millisl millism millisr
           pirl pirm pirr"""
    )
    PIR = []

    # Import Acc/Gyro/Mag/Uson data to a struct (namedtuple)
    IMUU_instance = namedtuple(
        "IMUU_instance",
        """begin mag acc uson"""
    )
    IMUU = []
    
    print "Parsing:"
    
    for file in sorted(os.listdir("./datasets/tsa%s/csv" %date)):
        if "csv" in file and not "speed" in file:
            print file

            with open(("./datasets/tsa%s/csv/" %date) + file, 'r') as target:
                target_reader = csv.reader(target)
                for line in target_reader:
                    try:
                        if "PIR" in line and not "-1" in line:
                            PIR.append(
                                PIR_instance(
                                    parse_time(line[0]),                # Start time
                                    float(line[2]),                     # Millis time (L)
                                    float(line[3]),                     # Millis time (M)
                                    float(line[4]),                     # Millis time (R)
                                    [float(x) for x in line[8:72]],     # 64 pixels temperature (L)
                                    [float(x) for x in line[72:136]],   # 64 pixels temperature (M)
                                    [float(x) for x in line[136:200]],  # 64 pixels temperature (R)
                                )
                           )
                        elif "IMU" in line:
                            IMUU.append(
                                IMUU_instance(
                                    parse_time(line[0]),         # Instance time
                                    [float(line[2]), float(line[3]), float(line[4])],   
                                    # Magnetometer measurement
                                    [float(line[5]), float(line[6]), float(line[7])],
                                    # Accelerometer measurement
                                    float(line[8])
                                    # Ultrasonic measurement
                                )
                            )
                        elif "PIR" in line and "-1" in line:
                            pass
                    except:
                        print "\n\nInvalid read: "
                        print line
                        return
    PIRR_data = np.array([x.pirr for x in PIR])

    # Import speed log to a struct (namedtuple)
    LOG_instance = namedtuple(
        "LOG_instance",
        """begin observation count"""
    )
    LOG = []

    for file in sorted(os.listdir("./datasets/tsa%s/csv" %date)):
        if "speed" in file:
            print file

            with open(("./datasets/tsa%s/csv/" %date) + file, 'r') as target:
                target_reader = csv.reader(target)
                
                for line in target_reader:
                    if "timestamp" in line:
                        pass
                    else:
                        LOG.append(
                            LOG_instance(
                                datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S.%f'), # Instance time
                                line[1],                                            # Observ
                                int(line[2]),                                       # Count
                            )
                        )
    
    return PIR, IMUU, LOG

"""
Specify synchronized PIR data, IMUU data, and LOG data and plot them in
colormap. To save the plots, set savefig=True. To display the plots, set
savefigs=False. By default, savefig is set to True.
_____________________________________________________________________________"""

def colormap(PIR_data, IMUU_reduced_data, LOG_inflated_data, label, save_fig=True):
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

    
    plt.figure(figsize=(15,10), dpi=150)
    colormap_row = []
    for x,y,z in zip(PIR_data, IMUU_reduced_data, LOG_inflated_data):
        colormap_row.append((np.array(x+[y]*6+[z]*6)-MEAN)/STDEV)
    colormap_row = np.array(colormap_row)
    colormap_row = np.transpose(colormap_row)
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
        plt.show(block=False)

    
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
        plt.show(block=False)

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


if __name__ == "__main__":
    PIR, IMUU, LOG = parse()
    print "Generating plots..."
    
    if False:
        # General raw data inspection
        fig1 = plt.figure()
        plt.plot([t.begin for t in PIR], [i.pirm[16] for i in PIR])
        plt.plot([t.begin for t in IMUU], [i.uson for i in IMUU])
        plt.title("Data Inspection")
        plt.xlabel("Time Stamp")
        plt.ylabel("PIR Data (C)")
    
    if False:
        # Time stamp inspection
        fig2 = plt.figure()
        plt.plot([t.begin for t in PIR], range(len(PIR)), label="PIR")
        plt.plot([t.begin for t in IMUU], range(len(IMUU)), label="IMUU")
        plt.legend()
        plt.title("Time Stamp Inspection")
        plt.ylabel("# of Data Collected")
        plt.xlabel("Time Stamp")

    # Inspect data in colormap
    PIR_timestamps = [t.begin for t in PIR]
    IMUU_timestamps = [t.begin for t in IMUU]
    LOG_timestamps = [t.begin for t in LOG]
    IMUU_reduced_idx = [bisect_left(IMUU_timestamps, t) for t in PIR_timestamps]
    LOG_inflated_idx = [bisect_left(PIR_timestamps, t) for t in LOG_timestamps]
    PIR_data = [x.pirl+x.pirm+x.pirr for x in PIR]
    IMUU_reduced_data = [IMUU[i].uson for i in IMUU_reduced_idx]
    LOG_inflated_data = [0]*len(PIR_timestamps)
    for i in LOG_inflated_idx:
        LOG_inflated_data[i] = 8
    
    colormap(PIR_data[0:960*2], 
             IMUU_reduced_data[0:960*2], 
             LOG_inflated_data[0:960*2], 
             0, save_fig=False)
    plt.show()

    if False:
        interval = 960
        for t in np.arange(0,len(PIR),interval):
            colormap(PIR_data[t:t+interval], 
                     IMUU_reduced_data[t:t+interval], 
                     LOG_inflated_data[t:t+interval], 
                     t, save_fig=True)


