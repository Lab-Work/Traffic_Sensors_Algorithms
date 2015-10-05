--------------------------------------------
README: TRANSLATION CODE FOR PIR SENSOR DATA
--------------------------------------------
PIR TRAFFIC SENSOR PLATFORM 
PROJECT BY: YANNING LI, WILLIAM BARBOUR, FANGYU WU (University of Illinois at Urbana-Champaign)
TESTING CONDUCTED: SEPTEMBER 03, 2015
CODE CREATED: SEPTEMBER 04, 2015
BY: WILLIAM BARBOUR

CODE DETAIL:
------------
This code package converts raw field data collected by the PIR sensor platform to meaningful values. Postprocessing is needed because the Raspberry Pi, in its current implementation, can not handle the conversion operations in real time without affecting the data collection frequency. Threading allows the two sensor sides to be collected independently and, likewise, a similar technique could probably be implemented in the future to convert some or all values in real time.
The Python scripts leverage portions of code written for the Raspberry Pi sensor package. The implementation of the conversion process is probably not ideal, but it was written for minimal usage.


NOTES:
------
The code runs ambiguously on files fitting the format "sensors_YYYY_MM_DD__HH_mm_ss.txt" in its same directory and converts them to the file "csv_sensors_YYYY_MM_DD__HH_mm_ss.csv". If the output needs to be customized, the conversion can easily be re-run.
The only data not included in the CSV files is the T(ambient) and T(cpix) for the PIR sensors. The code could easily be changed to include it, if needed.


INPUT:
------
The Raspberry Pi parses its readings somewhat by CSV as it saves them in .txt format. The data lines are different for the IMU and PIR data sides.


OUTPUT (IMU or PIR type):
-------------------------
IMU:
timestamp,IMU,mag-x,mag-y,mag-z,accel-x,accel-y,accel-z,ultrasonic

PIR:
timestamp,PIR,pir1_milsec,pir2_milsec,pir3_milsec,pixel#0_degC,pixel#1_degC,...,pixel191_degC

WHERE:
timestamp is recorded by Raspberry Pi (using inaccurate time)
format = hh_mm_ss_micros
pir#_milsec is recorded by Arduino using millis() method
