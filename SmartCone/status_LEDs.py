# =======================================================
# Hardware interface class for access to LED's on
# Raspberry Pi GPIO
# Dev mode can be enabled which includes minimally-tested
# blink methods, running in separate threads in background
# =======================================================
# Included in software package v2.01 as explicit class
# Consider wrapping with seven-segment display in future
# =======================================================


__author__ = 'wbarbour1'

#
# Pin number for LED's
#

# use BCM notation for pin numbers
ledA = 4
ledB = 17

import RPi.GPIO as GPIO
import threading
import time

colour = "red"


class status_LEDs(object):
    def __init__(self, dev=False):
        """initialization of status LED's attached to Raspberry Pi
        optional "dev" mode to enable blinking functions on both LED's"""
        self.dev = dev
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ledA, GPIO.OUT)
        GPIO.output(ledA, False)
        GPIO.setup(ledB, GPIO.OUT)
        GPIO.output(ledB, False)
        if dev:
            self.eventA = threading.Event()
            self.eventA.clear()
            self.eventB = threading.Event()
            self.eventB.clear()
            self.threadA = threading.Thread(name='ledA', target=self.flashLED_A, args=self.eventA)
            self.threadB = threading.Thread(name='ledB', target=self.flashLED_B, args=self.eventB)
            self.threadA.start()
            self.threadB.start()
            self.blink_interval_A = 0.5
            self.blink_interval_B = 0.5


    def on(self, led_pin):
        if led_pin == ledA or ledB:
            GPIO.output(led_pin, True)
            return 1
        else:
            return -1

    def off(self, led_pin):
        if led_pin == ledA or ledB:
            GPIO.output(led_pin, False)
            return 1
        else:
            return -1

    def start_blink(self, led_pin, t=0.5):
        """starts blinking of specified LED at interval t="""
        if led_pin == ledA:
            self.eventA.set()   # set flag to FALSE, starts execution of flashLED loop
            self.blink_interval_A = t
            return 1
        elif led_pin == ledB:
            self.eventB.set()
            self.blink_interval_B = t
            return 1
        else:
            return -1

    def stop_blink(self, led_pin):
        """starts blinking of specified LED"""
        if led_pin == ledA:
            self.eventA.clear()     # set flag to TRUE, stops execution of flashLED loop
            return 1
        elif led_pin == ledB:
            self.eventB.clear()
            return 1
        else:
            return -1

    def flashLED_A(self, ev):
        """flash LED-A at interval t="""
        while ev.isSet():       # loops if flag set to FALSE
            GPIO.output(ledA, True)
            time.sleep(self.blink_interval_A)
            GPIO.output(ledA, False)
            time.sleep(self.blink_interval_A)
        else:
            time.sleep(0.1)     # slow down loop for resource conservation
            # does not execute if event is set and LED is blinking


    def flashLED_B(self, ev):
        """flashes LED-B at interval t="""
        while ev.isSet():       # loops if flag set to FALSE
            GPIO.output(ledB, True)
            time.sleep(self.blink_interval_B)
            GPIO.output(ledB, False)
            time.sleep(self.blink_interval_B)
        else:
            time.sleep(0.1)     # slow down loop for resource conservation
            # does not execute if event is set and LED is blinking