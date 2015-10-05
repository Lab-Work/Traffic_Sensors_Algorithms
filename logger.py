from sys import stdin
from datetime import datetime
import csv


startTime = datetime.now()
OUTPUT_FILENAME = 'log_%s.csv' % str(startTime.strftime("%Y_%m_%d_%H_%M_%S"))
print """Type observation (e.g. speed/vehicle type) and press [enter] to log it.
        Type "exit" to save and exit.\n"""


with open(OUTPUT_FILENAME, 'w') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'observation', 'count', 'duration'])
    count = 0

    while(True):
        line = raw_input('log:%s> ' %str(count+1))
        if('exit' in line):
            break
        
        try:
            speed = line
            timeStamp = datetime.now()
            count += 1
            duration = timeStamp - startTime
            w.writerow([timeStamp,speed,count,duration])
            print ('\tElapsed: \t%s' % str(duration))
            print ('\tObservation: \t%s' % speed)
            print ('\tVeh. Count: \t%d' % count)
        except:
            print('Invalid input!')
