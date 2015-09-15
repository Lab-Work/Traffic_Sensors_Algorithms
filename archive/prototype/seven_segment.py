# =======================================================
# Hardware interface class for 7-segment display via
# driver chip (SN74LS47N)
# =======================================================
# Included in software package v2.01 as explicit class
# Consider wrapping with status LED's in future versions
# =======================================================

__author__ = 'wbarbour1'

#
# Pin numbers for 7-segment display driver
#

# use BCM notation for pin numbers
pinBI = 5
# pinLT = 0  # not used - hardwired high at driver chip
pinA = 6
pinB = 13
pinC = 19
pinD = 26


import RPi.GPIO as GPIO

class sev_seg(object):
    def __init__(self, init_disp):
        self.disp_val = init_disp
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pinA, GPIO.OUT)
        GPIO.output(pinA, False)
        GPIO.setup(pinB, GPIO.OUT)
        GPIO.output(pinB, False)
        GPIO.setup(pinC, GPIO.OUT)
        GPIO.output(pinC, False)
        GPIO.setup(pinD, GPIO.OUT)
        GPIO.output(pinD, False)
        GPIO.setup(pinBI, GPIO.OUT)
        GPIO.output(pinBI, False)
        self.set_disp(self.disp_val)
        # GPIO.setup(pinLT, GPIO.OUT)
        # GPIO.output(pinLT, True)

    def inc_disp(self):
        new_disp = 1 + self.disp_val
        if (new_disp > 9):
            new_disp = 0
        self.disp_val = new_disp
        self.set_disp(self.disp_val)

    def set_disp(self, dig):
        if dig == 0:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, False)
            GPIO.output(pinB, False)
            GPIO.output(pinC, False)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 1:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, True)
            GPIO.output(pinB, False)
            GPIO.output(pinC, False)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 2:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, False)
            GPIO.output(pinB, True)
            GPIO.output(pinC, False)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 3:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, True)
            GPIO.output(pinB, True)
            GPIO.output(pinC, False)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 4:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, False)
            GPIO.output(pinB, False)
            GPIO.output(pinC, True)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 5:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, True)
            GPIO.output(pinB, False)
            GPIO.output(pinC, True)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 6:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, False)
            GPIO.output(pinB, True)
            GPIO.output(pinC, True)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 7:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, True)
            GPIO.output(pinB, True)
            GPIO.output(pinC, True)
            GPIO.output(pinD, False)
            GPIO.output(pinBI, True)
        elif dig == 8:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, False)
            GPIO.output(pinB, False)
            GPIO.output(pinC, False)
            GPIO.output(pinD, True)
            GPIO.output(pinBI, True)
        elif dig == 9:
            GPIO.output(pinBI, False)
            GPIO.output(pinA, True)
            GPIO.output(pinB, False)
            GPIO.output(pinC, False)
            GPIO.output(pinD, True)
            GPIO.output(pinBI, True)