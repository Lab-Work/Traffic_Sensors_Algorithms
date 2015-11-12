"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu (fwu10@illinois.edu)
Date: September 11th, 2015

The generic parser that takes in arbitary number of raw data files and returns 
an ordered list of struct, each of which contains one instance of data. Necess-
ary conversion and specific parsing rules are applied in the process.
////////////////////////////////////////////////////////////////////////////"""

import numpy as np
import csv
from bisect import bisect_left
import time
from datetime import datetime
import os
from collections import namedtuple
from hex_converter import *

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

@IN: date of dataset
@OUT: fully parsed data stored in the following lists:
    PIR_data: PIR data, each line of which is a nametuple.
    IMUU_reduced_data: frequency-reduced IMU and Uson data, each line of which
    is a name tuple
    LOG_inflated_data: frequency-inflated speed log data, each line of which is
    a name tuple.
_____________________________________________________________________________________________"""

# A short format routine that converts unix timestamp to python datetime
# @IN: unix timestamp
# @OUT: converted time stored as Python datetime
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
    IMUU_reduced_data = []
    for i in IMUU_reduced_idx:
        if i < len(IMUU):
            IMUU_reduced_data.append(IMUU[i])
        else:
            IMUU_reduced_data.append(IMUU[-1])
    LOG_inflated_data = [0]*len(PIR_timestamps)
    for i in LOG_inflated_idx:
        if i < len(LOG_inflated_data):
            LOG_inflated_data[i] = 10
        else:
            LOG_inflated_data[-1] = 10
   
    return PIR_data, IMUU_reduced_data, LOG_inflated_data

if __name__ == "__main__":
    print "This is the parse module." 
