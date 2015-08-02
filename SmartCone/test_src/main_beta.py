'''
@author: Fangyu Wu
@date:   07/23/15
@parameters:
    <none>
@return:
    <none>


This is the pseudo main function that serves testing the first real time sensor pack.
It should contain:
1. SensorsPack module, which includes:
    i.   Ultrasonic sensor
    ii.  PIR arrays
    iii. Accelerometer & gyroscope
    iv.  Magnetic sensors
2. ModelsPack module, which include:
    i.   static threshold
    ii.  adaptive threhold
3. Control module, which checks the following system status:
    i.   hardware irregularity
    ii.  model effectiveness
   And based on the system status, output control signals.
4. display() and save() routines:
   display() output to LCD and LED;
   save() stores data from cache to SD card.
'''

from SensorPack import Sensors
from ModelsPack import Models
from Utilities import Cache

FREQ = 8                    # frequency in Hz
CACHE_SIZE = FREQ*360       # cache contains 3 min of data
OSTREAM = 1

def reset():   # restart the system
    command = "/usr/bin/sudo /sbin/shutdown -r now"
    import subprocess
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    #output = process.communicate()[0]
    #print output

def main():
    print "Hello!"
    sensors = Sensors()
    sensors_cache = Cache(CACHE_SIZE)
    models_cache = Cache(CACHE_SIZE)
    while (sensors_cache.size() < CACHE_SIZE):
        sensors_cache.fetch(sensors.read(models.sensors_addr))
    models = Models(sensors_cache)

    while (True):
        sensors_cache.fetch(sensors.read(models.sensors_addr))
        models_cache.fetch(models.estimate(sensors_cache))
        if (OSTREAM):
            models_cache.display()
        if (models_cache.size() == CACHE_SIZE):
            models_cache.write_back()
        if (models.sensors_status = "error"):
            reset()
        #if (battery is low):
        #    hibernate 3 hr

if __name__ == "__main__":
    main()
