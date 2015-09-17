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
from crosscorrelation import find_pixels_shifts
import cookb_signalsmooth as ss
from colormap import colormap

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
   
    return PIR_data, IMUU_reduced_data, LOG_inflated_data

if __name__ == "__main__":
    
    PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
    find_pixels_shifts(PIR_data, [965,985,0,15], time_cc=False)
    colormap(PIR_data[960:1200], 
             IMUU_reduced_data[960:1200], 
             LOG_inflated_data[960:1200], 
             960, save_fig=False)
    plt.show()

