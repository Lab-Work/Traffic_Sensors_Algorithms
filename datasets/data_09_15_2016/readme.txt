One direction two lane field test.

There are three included folders in this directory:

cam1/ includes the raw and container files for video taken from the side of the road opposite to that of the PIR sensor.
cam2/ includes the raw and container files for video taken from the same side as the PIR sensor.


The timestamps for the start of the video are below. It might have taken some time for the cameras to start up, however it should make alignment much easier. Note that the timestamps are in different time zones: cam1 is GMT and cam2 is GMT-05. However, it should be easy to match the videos using the minute stamps. 

cam1: 2016-09-15 22:39:52.263389
cam2: 2016-09-15 17:40:00.237423

data/ include two separate folders:

RAW/ contains numpy files of the raw data in the following format per entry: - [timestamp, 4x16 pir array, ptat, cp]. They are split into 120 and 60 degree FOV, and 16,32,64,128, and 256 hz frequencies; the naming is fairly intuitive.
CALC_TEMP/ contains numpy files of the temperature calculated data in the following format per entry: [timestamp, 4x16 pir temperature array, ptat, cp]. The currently only include the 60 degree FOVs.

Alignment of the video and PIR data should be fairly simple since we have the timestamp for each frame. 
