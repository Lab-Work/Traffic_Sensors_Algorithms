# =======================================================
# Upper-level framework for interface with sensor classes
# Sensor pack is called by and integrated with checker
# for data collection and sensor control
# =======================================================
# Version 2.01
# -version 2.xx began the integration of data with
#  real-time analysis; "checker" assesses validity of
#  data and controls its collection via periodic feedback
# -code has been consolidated and re-organized for
#  simplicity, readability, and further development
# =======================================================

__author__ = 'wbarbour1'

import RPi.GPIO as GPIO
from sensor_classes import ADS1x15, gyro, adxl, mag3110, PIR3

port_name = '/dev/ttyUSB0'
baud_rate = 230400
MEA_BUF_SIZE = 424

LED_pin1 = 4

class sensor_pack(object):

    def __init__(self):
        self.mag = mag3110(1)
        self.accel = adxl()
        self.gyro = gyro()
        self.adc = ADS1x15()
        self.PIRs = PIR3(port_name, baud_rate, MEA_BUF_SIZE)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_pin1, GPIO.OUT)

    def fetch(self):
        GPIO.output(LED_pin1, True)
 nce to look at it, let me know if you have any questions. We can discuss our
 next steps whenever you get some time. I’ll be out of the office
 Monday-Wednesday with Dan in Jacksonville, but I’m around the lab, otherwise.       full_read = []
        full_read.append(self.mag.read())
        full_read.append(self.accel.read())
        full_read.append(self.gyro.read())
        full_read.append(self.adc.read())
        full_read.append(self.PIRs.read())
        GPIO.output(LED_pin1, False)
        return full_read
# TODO allow configuration to turn on/off sensors selectively from outside sensor_pack class
# TODO add in any configuration to mag/accel/gyro/adc classes, controlled by sensor_pack set_config
    # config_str = [mag_config, accel_config, gyro_config, ADC_config, PIR_config]
        # PIR_config = [ PIR1 ON/OFF (1/0), PIR2 ON/OFF, PIR3 ON/OFF, pixel_str]
            # pixel_str = [[pir#,[px_row, px_col]], [pir#,[px_row, px_col]]]
    def set_config(self, config_str):
        self.mag.config(config_str[0])
        self.accel.config(config_str[1])
        self.gyro.config(config_str[2])
        self.adc.config(config_str[3])
        self.PIRs.config(config_str[4])
