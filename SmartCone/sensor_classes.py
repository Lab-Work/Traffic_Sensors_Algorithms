# =======================================================
# Lower-level classes for interface with sensor hardware
# includes import statements for hardware classes
# Direct dependent of sensor pack (sensor_pack.py)
# =======================================================
# Order:
#   1. Import statements
#   2. Constants and register values for sensor interface
#   3. PIR constants and register values
#   4. PIR classes
#   5. ADXL accelerometer class (IMU digital combo)
#   6. ITG-3200 gyroscope class (IMU digital combo)
#   7. MAG3110 magnetometer class
#   8. ADS1015 Analog-to-digital converter class
#   9. Adafruit_I2C class, hardware interface for ADC (Ultson)
#   10. PY_I2C class, hardware interface for Accel., Mag., Gyr.

__author__ = 'wbarbour1'


# =======================================================
#
# 1. Import statements
#
# =======================================================


from datetime import datetime
import numpy as np
import serial
import smbus
import re
import time
import RPi.GPIO as GPIO




# =======================================================
#
# 2. Constants and register values for sensor interface
#
# =======================================================


#
# I2C addresses
#

magaddress = 0x0e  # magnetometer address
accaddress = 0x53  # accelerometer address
gyraddress = 0x68  # gyro address


#
# Magnetometer registers and settings
#

# most significant byte register addresses for x,y,z parameters
# readS16 and U16 automatically increment address by 1
xmsb = 0x01
# xlsb=0x02
ymsb = 0x03
# ylsb=0x04
zmsb = 0x05
# zlsb=0x06

mode = 0x08  # register address for system mode (standby/active/active corrected)
dietemp = 0x0f
ctrlreg1 = 0x10
ctrlreg2 = 0x11  # control register 2 address; includes mag. sensor reset and output correction bits

# opmode=0x81 #5Hz, oversample=16, 16-bit, normal, active (see data sheet)
opmode = 0x01  # 80Hz, oversample=16, 16-bit, normal, active
# opmode=0x51 #5Hz, oversample=32, 16-bit, normal, active
# opmode=0x05 #80Hz, oversample=16, 8-bit fast read, normal, active
# note: 16-bit read functions must be substituted below

mrstraw = 0xA0  # mag. sensor reset enabled, raw data output (no correction)
mrstnrm = 0x80  # mag. sensor reset enabled, normal data output (corrected)


#
# Accelerometer registers and settings
#

powerctl = 0x2d  # power control register address
dataformat = 0x31  # data format register address
datax0 = 0x32  # LSB (x0,y0,z0) and MSB (x1,y1,z1) registers for x,y,z data
datax1 = 0x33
datay0 = 0x34
datay1 = 0x35
dataz0 = 0x36
dataz1 = 0x37

pwrsetting = 0x08  # auto-sleep disabled, measure mode, sleep mode off
dataformsetting = 0x01  # g range set to +/-4g
# possible that setting justify bit = 1 could fix readS16 issue


#
# Gyro registers and settings
#

smplrtdiv = 0x15  # sample rate divider register
dlpffs = 0x16  # digital low pass filter full scale register
intcfg = 0x17  # interrupt configuration register
pwrmgm = 0x3e  # power management register

xouth = 0x1d  # x,y,z output registers, each with high and low bytes
xoutl = 0x1e
youth = 0x1f
youtl = 0x20
zouth = 0x21
zoutl = 0x22

dlpfcfg0 = 0x00  # set digital low pass filter to 256Hz w/ 8kHz internal sample rate
dlpfcfg1 = 0x01  # set DLPF to 188Hz w/ 1kHz internal sample rate
dlpffssel = 0x18  # set full scale select to recommended (00011000)
pwrmgmclkselx = 0x01  # byte set for x gyro reference
intcfgitgrdyen = 1 << 2  # enable interrupt when device ready
intcfgrawrdyen = 1 << 0  # enable interrupt when data available


#
# ADC settings
#

gain = 6144  # +/- 6.144 Vdc
# acceptable values: +/- 0.256, 0.512, 1.024, 2.048, 4.096, 6.144 Vdc
sps = 64  # 64 samples per second collected
# acceptable values: 8, 16, 32, 64, 128, 250, 475, 860




# =======================================================
#
# 3. PIR constants and register values
#
# =======================================================


# Yes, really, it's not 0x60, the EEPROM has its own address (I've never seen this before)
MLX90620_EEPROM_WRITE = 0xA0
MLX90620_EEPROM_READ = 0xA1

# The sensor's I2C address is 0x60. So 0b.1100.000W becomes 0xC0
MLX90620_WRITE = 0xC0
MLX90620_READ = 0xC1

# Commands
CMD_READ_REGISTER = 0x02

# Begin registers
CAL_ACP = 0xD4
CAL_BCP = 0xD5
CAL_alphaCP_L = 0xD6
CAL_alphaCP_H = 0xD7
CAL_TGC = 0xD8
CAL_BI_SCALE = 0xD9

VTH_L = 0xDA
VTH_H = 0xDB
KT1_L = 0xDC
KT1_H = 0xDD
KT2_L = 0xDE
KT2_H = 0xDF

# Common sensitivity coefficients
CAL_A0_L = 0xE0
CAL_A0_H = 0xE1
CAL_A0_SCALE = 0xE2
CAL_DELTA_A_SCALE = 0xE3
CAL_EMIS_L = 0xE4
CAL_EMIS_H = 0xE5

# Config register = 0xF5-F6

OSC_TRIM_VALUE = 0xF7

# Bits within configuration register 0x92

POR_TEST = 10

# first PIR EEPROM
EEPROM_PIR1 = [195, 195, 195, 194, 203, 203, 204, 203, 208, 211, 210, 205, 214, 216, 215, 209,
               219, 219, 217, 213, 221, 221, 220, 215, 223, 224, 223, 218, 226, 227, 225, 222,
               229, 229, 228, 224, 231, 230, 230, 228, 230, 232, 231, 228, 232, 233, 233, 231,
               232, 233, 233, 234, 233, 234, 234, 233, 232, 231, 233, 233, 231, 232, 232, 229,
               171, 188, 188, 179, 196, 196, 196, 188, 188, 205, 205, 205, 205, 205, 205, 196,
               213, 213, 213, 188, 213, 213, 213, 213, 213, 222, 222, 205, 222, 222, 222, 213,
               222, 222, 222, 222, 230, 230, 222, 222, 222, 222, 222, 222, 230, 222, 239, 222,
               222, 222, 222, 230, 222, 222, 230, 222, 222, 222, 222, 230, 239, 222, 222, 222,
               0, 23, 30, 7, 23, 57, 57, 44, 47, 74, 87, 70, 60, 97, 101, 97,
               74, 117, 124, 107, 80, 124, 127, 127, 84, 121, 134, 127, 80, 117, 134, 117,
               70, 114, 117, 117, 67, 104, 117, 111, 60, 97, 111, 117, 50, 87, 94, 94,
               44, 80, 94, 77, 33, 70, 84, 74, 23, 64, 70, 57, 3, 37, 44, 44,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 40, 0,
               0, 2, 0, 224, 221, 205, 29, 10, 0, 8, 103, 26, 85, 91, 79, 27,
               15, 146, 41, 34, 0, 128, 0, 0, 255, 18, 0, 62, 26, 190, 18, 25,
               191, 27, 103, 26, 255, 30, 116, 87, 166, 32, 143, 67, 4, 33, 0, 80]

alpha_ij_PIR1 = [1.70035E-8, 1.83422E-8, 1.87497E-8, 1.74109E-8, 1.83422E-8, 2.03213E-8, 2.03213E-8, 1.95646E-8,
                 1.97392E-8, 2.13108E-8, 2.20675E-8, 2.10780E-8, 2.04959E-8, 2.26496E-8, 2.28824E-8, 2.26496E-8,
                 2.13108E-8, 2.38138E-8, 2.42212E-8, 2.32317E-8, 2.16601E-8, 2.42212E-8, 2.43958E-8, 2.43958E-8,
                 2.18929E-8, 2.40466E-8, 2.48033E-8, 2.43958E-8, 2.16601E-8, 2.38138E-8, 2.48033E-8, 2.38138E-8,
                 2.10780E-8, 2.36391E-8, 2.38138E-8, 2.38138E-8, 2.09034E-8, 2.30571E-8, 2.38138E-8, 2.34645E-8,
                 2.04959E-8, 2.26496E-8, 2.34645E-8, 2.38138E-8, 1.99138E-8, 2.20675E-8, 2.24750E-8, 2.24750E-8,
                 1.95646E-8, 2.16601E-8, 2.24750E-8, 2.14854E-8, 1.89243E-8, 2.10780E-8, 2.18929E-8, 2.13108E-8,
                 1.83422E-8, 2.07287E-8, 2.10780E-8, 2.03213E-8, 1.71781E-8, 1.91571E-8, 1.95646E-8, 1.95646E-8]

# second PIR EEPROM
EEPROM_PIR2 = [202, 202, 202, 205, 206, 204, 207, 206, 209, 210, 211, 210, 213, 213, 212, 210,
               216, 216, 215, 212, 217, 216, 216, 213, 218, 219, 218, 214, 220, 220, 219, 218,
               222, 221, 221, 221, 224, 224, 224, 224, 225, 227, 226, 225, 226, 226, 228, 228,
               228, 229, 229, 230, 229, 231, 230, 232, 228, 229, 230, 231, 227, 229, 228, 230,
               188, 179, 188, 188, 188, 188, 188, 188, 205, 205, 196, 205, 196, 205, 205, 205,
               205, 205, 205, 196, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
               205, 205, 205, 205, 213, 222, 222, 205, 213, 222, 222, 222, 222, 222, 222, 222,
               222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222,
               0, 50, 60, 43, 37, 97, 107, 94, 70, 124, 144, 127, 97, 157, 187, 160,
               134, 194, 211, 184, 150, 214, 224, 204, 167, 227, 241, 221, 174, 237, 254, 221,
               174, 247, 254, 231, 167, 237, 247, 221, 164, 227, 241, 221, 147, 217, 227, 194,
               134, 190, 211, 174, 124, 170, 190, 154, 104, 150, 167, 140, 77, 124, 134, 107,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 222, 217, 205, 213, 5, 0, 8, 135, 26, 154, 91, 79, 27,
               163, 67, 40, 34, 0, 128, 0, 0, 255, 19, 0, 89, 26, 190, 49, 25,
               224, 27, 135, 26, 255, 30, 116, 85, 166, 32, 143, 67, 4, 33, 0, 170]

alpha_ij_PIR2 = [1.57479E-8, 1.86583E-8, 1.92404E-8, 1.82509E-8, 1.79016E-8, 2.13941E-8, 2.19762E-8, 2.12195E-8,
                 1.98225E-8, 2.29657E-8, 2.41298E-8, 2.31403E-8, 2.13941E-8, 2.48865E-8, 2.66328E-8, 2.50612E-8,
                 2.35478E-8, 2.70402E-8, 2.80298E-8, 2.64582E-8, 2.44791E-8, 2.82044E-8, 2.87865E-8, 2.76223E-8,
                 2.54686E-8, 2.89611E-8, 2.97760E-8, 2.86118E-8, 2.58761E-8, 2.95432E-8, 3.05327E-8, 2.86118E-8,
                 2.58761E-8, 3.01252E-8, 3.05327E-8, 2.91939E-8, 2.54686E-8, 2.95432E-8, 3.01252E-8, 2.86118E-8,
                 2.52940E-8, 2.89611E-8, 2.97760E-8, 2.86118E-8, 2.43045E-8, 2.83790E-8, 2.89611E-8, 2.70402E-8,
                 2.35478E-8, 2.68074E-8, 2.80298E-8, 2.58761E-8, 2.29657E-8, 2.56432E-8, 2.68074E-8, 2.47119E-8,
                 2.18015E-8, 2.44791E-8, 2.54686E-8, 2.38970E-8, 2.02299E-8, 2.29657E-8, 2.35478E-8, 2.19762E-8]


# third PIR EEPROM
EEPROM_PIR3 = [223, 224, 224, 224, 223, 225, 226, 227, 225, 226, 228, 227, 228, 228, 226, 226,
               229, 229, 229, 226, 229, 228, 230, 228, 229, 229, 230, 230, 230, 231, 232, 231,
               233, 233, 233, 233, 234, 234, 235, 236, 233, 235, 235, 236, 234, 235, 237, 238,
               233, 235, 236, 238, 234, 236, 236, 240, 234, 236, 237, 239, 231, 234, 233, 237,
               154, 154, 154, 154, 154, 154, 171, 171, 154, 171, 171, 171, 171, 188, 171, 154,
               171, 171, 188, 171, 188, 171, 188, 171, 171, 188, 188, 188, 188, 188, 188, 188,
               188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 205,
               188, 188, 188, 205, 188, 188, 188, 222, 188, 188, 188, 205, 188, 188, 188, 188,
               0, 50, 70, 63, 40, 93, 107, 100, 67, 127, 140, 133, 97, 160, 180, 167,
               127, 197, 203, 200, 147, 223, 227, 217, 160, 230, 247, 223, 163, 230, 240, 227,
               157, 230, 247, 233, 153, 227, 233, 220, 160, 220, 233, 217, 150, 213, 213, 210,
               137, 197, 210, 197, 120, 177, 190, 177, 103, 150, 163, 150, 73, 123, 140, 120,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 225, 223, 154, 242, 4, 0, 9, 84, 26, 137, 90, 183, 63,
               1, 125, 41, 34, 0, 128, 0, 0, 255, 34, 0, 44, 26, 190, 4, 25,
               171, 27, 84, 26, 255, 30, 116, 92, 166, 32, 143, 67, 4, 33, 1, 48]

alpha_ij_PIR3 = [1.45524E-8, 1.74628E-8, 1.86269E-8, 1.82195E-8, 1.68807E-8, 1.99657E-8, 2.07806E-8, 2.03731E-8,
                 1.84523E-8, 2.19447E-8, 2.27014E-8, 2.22940E-8, 2.01985E-8, 2.38656E-8, 2.50297E-8, 2.42730E-8,
                 2.19447E-8, 2.60193E-8, 2.63685E-8, 2.61939E-8, 2.31089E-8, 2.75327E-8, 2.77655E-8, 2.71834E-8,
                 2.38656E-8, 2.79401E-8, 2.89297E-8, 2.75327E-8, 2.40402E-8, 2.79401E-8, 2.85222E-8, 2.77655E-8,
                 2.36910E-8, 2.79401E-8, 2.89297E-8, 2.81148E-8, 2.34581E-8, 2.77655E-8, 2.81148E-8, 2.73581E-8,
                 2.38656E-8, 2.73581E-8, 2.81148E-8, 2.71834E-8, 2.32835E-8, 2.69506E-8, 2.69506E-8, 2.67760E-8,
                 2.25268E-8, 2.60193E-8, 2.67760E-8, 2.60193E-8, 2.15373E-8, 2.48551E-8, 2.56118E-8, 2.48551E-8,
                 2.05478E-8, 2.32835E-8, 2.40402E-8, 2.32835E-8, 1.88015E-8, 2.17119E-8, 2.27014E-8, 2.15373E-8]

test_line = '$$$$00010460FFBCFFBFFFBAFFBCFFCAFFC8FFC5FFC0FFCCFFCDFFCAFFC9FFCDFFCCFFCFFFC8FFD6FFD3FFD3FFCEFFD5FFD9FFD8FFD1FFD8FFDDFFD8FFD3FFDFFFDCFFDCFFD8FFDDFFE4FFE0FFE1FFDEFFEAFFE2FFDCFFE4FFE1FFE3FFDCFFE3FFE6FFE2FFE0FFE5FFE6FFE2FFDEFFE0FFE0FFE1FFE4FFE5FFE8FFE3FFE0FFDEFFE6FFE1FFE11A78FFD1000000010460FFFFFFBCFFFFFFBFFFFFFFBAFFFFFFBCFFFFFFCAFFFFFFC8FFFFFFC5FFFFFFC0FFFFFFCCFFFFFFCDFFFFFFCAFFFFFFC9FFFFFFCDFFFFFFCCFFFFFFCFFFFFFFC8FFFFFFD6FFFFFFD3FFFFFFD3FFFFFFCEFFFFFFD5FFFFFFD9FFFFFFD8FFFFFFD1FFFFFFD8FFFFFFDDFFFFFFD8FFFFFFD3FFFFFFDFFFFFFFDCFFFFFFDCFFFFFFD8FFFFFFDDFFFFFFE4FFFFFFE0FFFFFFE1FFFFFFDEFFFFFFEAFFFFFFE2FFFFFFDCFFFFFFE4FFFFFFE1FFFFFFE3FFFFFFDCFFFFFFE3FFFFFFE6FFFFFFE2FFFFFFE0FFFFFFE5FFFFFFE6FFFFFFE2FFFFFFDEFFFFFFE0FFFFFFE0FFFFFFE1FFFFFFE4FFFFFFE5FFFFFFE8FFFFFFE3FFFFFFE0FFFFFFDEFFFFFFE6FFFFFFE1FFFFFFE11A78FFD1000000010460FFBCFFBFFFBAFFBCFFCAFFC8FFC5FFC0FFCCFFCDFFCAFFC9FFCDFFCCFFCFFFC8FFD6FFD3FFD3FFCEFFD5FFD9FFD8FFD1FFD8FFDDFFD8FFD3FFDFFFDCFFDCFFD8FFDDFFE4FFE0FFE1FFDEFFEAFFE2FFDCFFE4FFE1FFE3FFDCFFE3FFE6FFE2FFE0FFE5FFE6FFE2FFDEFFE0FFE0FFE1FFE4FFE5FFE8FFE3FFE0FFDEFFE6FFE1FFE11A78FFD100000001'
# $PIR1,1135,21.95,23.78,21.25,22.59,25.32,24.43,22.55,20.44,23.92,23.14,22.36,24.02,21.62,20.72,22.48,22.03,23.58,22.60,23.45,22.96,22.28,24.24,24.25,23.46,22.76,24.61,23.10,23.08,24.48,22.99,23.88,23.41,22.17,25.39,24.20,26.18,21.66,27.39,24.20,22.54,24.87,22.85,24.18,22.59,23.40,24.51,22.81,22.83,24.35,24.48,22.83,20.41,21.24,21.25,21.85,23.57,24.25,26.24,23.09,21.55,20.74,24.81,22.39,23.86,25.74,1*




# =======================================================
#
# 4. PIR classes
#
# =======================================================

class PIR3:
    def __init__(self, port_name, baud_rate, Meas_buf_size):
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.buf_size = Meas_buf_size
        # print 'self.buf_size: {0}'.format(self.buf_size)

        self.pir_on_off = 1
        self.pixel_str = []     # contains list of pixels to be converted to temperatures
        self.convert_flag = False

        # save all incoming data into those lists
        self.all_data = []
        self.data_to_save = []
        self.all_imu_data = []
        self.imu_to_save = []

        # parse each strings or byte buffers into those matrix and array
        # HEX values
        self.temp_pir1_4x16 = np.zeros((4, 16))
        self.temp_pir2_4x16 = np.zeros((4, 16))
        self.temp_pir3_4x16 = np.zeros((4, 16))

        self.ptat_pir1 = 0
        self.ptat_pir2 = 0
        self.ptat_pir3 = 0

        self.cpix_pir1 = 0
        self.cpix_pir2 = 0
        self.cpix_pir3 = 0

        # initialize the three pir sensors
        self.pir_1 = PIRSensor(1, alpha_ij_PIR1, EEPROM_PIR1)
        self.pir_2 = PIRSensor(2, alpha_ij_PIR2, EEPROM_PIR2)
        self.pir_3 = PIRSensor(3, alpha_ij_PIR3, EEPROM_PIR3)

        self.pir_1.const_init()
        self.pir_2.const_init()
        self.pir_3.const_init()

        # open serial
        self.ser = serial.Serial(self.port_name, self.baud_rate)

        if self.ser.isOpen():
            print "opened serial port {0}".format(self.port_name)

    # config_str = [ PIR123 ON/OFF (1/0), pixel_str]
    # pixel_str = [px1, px2, ..., pxn]
    # assuming px = [0, 191]
    def config(self, config_str):
        # parse config_str to pixel_str and other settings
        self.pir_on_off = config_str[0]
        pixel_list = config_str[1]
        # set convert_flag to appropriate value T/F
        if pixel_list == []:
            self.convert_flag = False
        else:
            self.convert_flag = True
        for pixel in pixel_list:
            px = [(pixel // 64) + 1, [(pixel % 64) // 16, (pixel % 64) % 16]]       # [PIR123, [px row, px col]]
            self.pixel_str.append(px)

    # the following function is for testing the sensor
    # For real-time plotting, the byte stream is sent as a char string with the same format
    # read, save and real-time plotting
    # set save_every_min as inf if do not want to save
    # T_min and T_max are respectively the temperature for plotting
    def read(self):

        self.ser.flushInput()
        time.sleep(0.5)

        read_flag = False

        # read new line of data
        while read_flag is False and self.ser.isOpen():

            line = self.ser.readline()

            t_now = time.time()
            print 'loop_time: {0}'.format(t_now)

            # append every line of inputs to the matrices
            self.data_to_save.append((t_now, line))
            self.all_data.append((t_now, line))

            # wait until received data, then process and plot
            # print line
            # first four characters are $, last char is *
            # but in python, last two chars are \n, skip those.
            if line[0] != '$' or line[1] != '$' or line[2] != '$' or line[3] != '$' or line[-3] != '*':
                continue

            read_flag = True

            if self.convert_flag is True:
                calc_temp_return = []
                # parse the data string and extract the data to structures
                self.parse_IR_string_line(line)
                for i in self.pixel_str:
                    pixel_row = i[1][0]
                    pixel_col = i[1][1]
                    if i[0] == 1:
                        calc_temp_return.append(self.pir_1.calculate_single_pixel_temperature(self.ptat_pir1, [[pixel_row, pixel_col], self.temp_pir1_4x16[pixel_row, pixel_col]], self.cpix_pir1))
                    if i[0] == 2:
                        calc_temp_return.append(self.pir_2.calculate_single_pixel_temperature(self.ptat_pir2, [[pixel_row, pixel_col], self.temp_pir2_4x16[pixel_row, pixel_col]], self.cpix_pir2))
                    if i[0] == 3:
                        calc_temp_return.append(self.pir_3.calculate_single_pixel_temperature(self.ptat_pir3, [[pixel_row, pixel_col], self.temp_pir3_4x16[pixel_row, pixel_col]], self.cpix_pir3))

        return [line, calc_temp_return]


    # parse the IR data string line
    # extract the IRdata matrix, and update temperature 4x16
    # line length 406 bytes; each () is 4 char
    # ($$$$)[(ID)(time)(64x2 bytes)(ptat)(cpix)(ultra)][pir2][pir3](counter)
    # new
    # ($$$$)[(IDID)(time 4 characters in millis())(64x4 char)(ptat 4 char)(cpix 4 char)][pir2][pir3](counter 4 char)
    def parse_IR_string_line(self, line):

        line_list = [line[i:i + 4] for i in range(0, len(line), 4)]

        # parse the string
        ################################################
        # Parse the first sensor
        print line_list
        index = 2
        # index +=1   # omit the sensor id

        t_pir1 = int(line_list[index], 16)
        index += 1
        for col in range(0, 16):
            for row in range(0, 4):
                self.temp_pir1_4x16[row, col] = self.sign_int(int(line_list[index], 16))
                index += 1

        self.ptat_pir1 = int(line_list[index], 16)
        index += 1
        self.cpix_pir1 = self.sign_int(int(line_list[index], 16))
        index += 1

        ################################################
        # make sure at the second sensor
        if int(line_list[index], 16) != 2:
            print 'Parsing error at second sensor'
            return 1
        else:
            index += 1

        t_pir2 = int(line_list[index], 16)
        index += 1
        for col in range(0, 16):
            for row in range(0, 4):
                self.temp_pir2_4x16[row, col] = self.sign_int(int(line_list[index], 16))
                index += 1

        self.ptat_pir2 = int(line_list[index], 16)
        index += 1
        self.cpix_pir2 = self.sign_int(int(line_list[index], 16))
        index += 1


        ################################################
        # make sure at the third sensor
        if int(line_list[index], 16) != 3:
            print 'Parsing error at third sensor'
            return 1
        else:
            index += 1

        t_pir3 = int(line_list[index], 16)
        index += 1
        for col in range(0, 16):
            for row in range(0, 4):
                self.temp_pir3_4x16[row, col] = self.sign_int(int(line_list[index], 16))
                index += 1

        self.ptat_pir3 = int(line_list[index], 16)
        index += 1
        self.cpix_pir3 = self.sign_int(int(line_list[index], 16))
        index += 1

        print 'counter:'
        print int(line_list[index], 16)


    # convert hex to signed int from -32767~32767
    def sign_int(self, val):
        if val > 32767:
            return val - 65536
        else:
            return val

    # convert hex to signed byte
    def sign_byte(self, val):
        if val > 127:
            return val - 256
        else:
            return val


# this is the library for PIR90620 sensor.
# construct object with EEPROM constants,
# then given the IRraw data, it should compute the temperature matrix in C
class PIRSensor:
    def __init__(self, pir_id, alpha_ij, eepromData):
        # alpha matrix for each PIR sensor
        self.pir_id = pir_id
        self.alpha_ij = np.copy(alpha_ij)
        self.eepromData = np.copy(eepromData)

        # constants needs to be first computed and then used
        self.v_th = 0
        self.a_cp = 0
        self.b_cp = 0
        self.tgc = 0
        self.b_i_scale = 0
        self.k_t1 = 0
        self.k_t2 = 0
        self.emissivity = 0

        self.a_ij = np.zeros(64)  # 64 array
        self.b_ij = np.zeros(64)

        # raw data to be received
        # a list of 64x3 arrays. each array is a time series temperature data for one pixel
        self.all_temperatures = []
        for i in range(0, 64):
            self.all_temperatures.append([])

        self.temperatures = np.zeros((4, 16))
        self.Tambient = 0

    # initialize the constant parameters
    def const_init(self):
        # =============================================
        # read and calculate constants for PIR instance
        # =============================================
        self.v_th = 256 * self.eepromData[VTH_H] + self.eepromData[VTH_L]
        self.k_t1 = (256 * self.eepromData[KT1_H] + self.eepromData[KT1_L]) / 1024.0
        self.k_t2 = (256 * self.eepromData[KT2_H] + self.eepromData[KT2_L]) / 1048576.0  # 2^20 = 1,048,576
        self.emissivity = (256 * self.eepromData[CAL_EMIS_H] + self.eepromData[CAL_EMIS_L]) / 32768.0

        self.a_cp = self.eepromData[CAL_ACP]
        if (self.a_cp > 127):
            self.a_cp -= 256  # These values are stored as 2's compliment. This coverts it if necessary.

        self.b_cp = self.eepromData[CAL_BCP]
        if (self.b_cp > 127):
            self.b_cp -= 256

        self.tgc = self.eepromData[CAL_TGC]
        if (self.tgc > 127):
            self.tgc -= 256

        self.b_i_scale = self.eepromData[CAL_BI_SCALE]

        # =============================================
        # read and calculate constants for PIR instance
        # =============================================
        for i in range(0, 64):
            # Read the individual pixel offsets
            self.a_ij[i] = self.eepromData[i]
            if (self.a_ij[i] > 127):
                self.a_ij[i] -= 256  # These values are stored as 2's compliment. This coverts it if necessary.

            # Read the individual pixel offset slope coefficients
            self.b_ij[i] = self.eepromData[0x40 + i]  # Bi(i,j) begins 64 bytes into EEPROM at 0x40
            if (self.b_ij[i] > 127):
                self.b_ij[i] -= 256

            # print 'finished initializing values\n'
            # print 'v_th: {0}\n'.format(self.v_th)
            # print 'k_t1: {0}\n'.format(self.k_t1)
            # print 'k_t2: {0}\n'.format(self.k_t2)
            # print 'emissivity: {0}\n'.format(self.emissivity)
            # print 'a_cp: {0}\n'.format(self.a_cp)
            # print 'b_cp: {0}\n'.format(self.b_cp)
            # print 'tgc: {0}\n'.format(self.tgc)
            # print 'b_i_scale: {0}\n'.format(self.b_i_scale)
            # print 'a_ij:'
            # for i in range(0, 64):
            #     print self.a_ij[i]
            # print '\n b_ij:'
            # for i in range(0, 64):
            #     print self.b_ij[i]

    # calculate TA
    def calculate_TA(self, ptat):
        self.Tambient = (-self.k_t1 + np.sqrt(np.power(self.k_t1, 2) -
                                              (4 * self.k_t2 * (self.v_th - ptat)))) / (2 * self.k_t2) + 25
        print 'ambient temperature: '
        print self.Tambient

    # calculate single pixel
    # iraData = [[2,3], Hex]
    def calculate_single_pixel_temperature(self, ptat, irData, cpix):

        self.calculate_TA(ptat)

        if self.Tambient <= -100:
            self.Tambient = 0  # reset, do not proceed to compute the To
        else:
            self.calculate_single_pixel_TO(irData, cpix)


    # iraData = [[2,3], Hex]
    def calculate_single_pixel_TO(self, irData, cpix):

        # Calculate the offset compensation for the one compensation pixel
        # This is a constant in the TO calculation, so calculate it here.
        v_cp_off_comp = cpix - (self.a_cp + (self.b_cp / np.power(2, self.b_i_scale)) * (self.Tambient - 25))

        col = irData[0][1]
        row = irData[0][0]
        hexData = irData[1]

        i = col * 4 + row
        v_ir_off_comp = hexData - (self.a_ij[i] + (self.b_ij[i] / np.power(2, self.b_i_scale)) * (
                    self.Tambient - 25))  # 1: Calculate Offset Compensation

        v_ir_tgc_comp = v_ir_off_comp - (
                    (self.tgc / 32) * v_cp_off_comp)  # 2: Calculate Thermal Gradien Compensation (TGC)

        v_ir_comp = v_ir_tgc_comp / self.emissivity  # 3: Calculate Emissivity Compensation

        temperature = np.sqrt(
                    np.sqrt((v_ir_comp / self.alpha_ij[i]) + np.power(self.Tambient + 273.15, 4))) - 273.15

        return temperature


    # calculate the mean and std of the measurement of each pixel
    def calculate_std(self):

        # skip the first a few that have non sense values due to transmission corruption
        std = []
        for i in range(0, 64):
            # print 'all_temperatures:{0}'.format(self.all_temperatures[i])

            if self.all_temperatures[i]:  # if not empty
                # print 'before_std'
                std.append(np.std(self.all_temperatures[i]))
                # print 'after_std'

        # print 'std of PIR {0}: {1} '.format(self.pir_id, std)
        print 'mean_std of PIR {0}: {1}'.format(self.pir_id, np.mean(std))





# =======================================================
#
# 5. ADXL accelerometer class (IMU digital combo)
#
# =======================================================
# Modified from sparkfun IMU Digital Combo Board code
# Inspiration source: https://github.com/sparkfun/IMU_Digital_Combo_Board
# Liscence info: Creative Commons Attribution-ShareAlike 4.0 Intl.
#                 Modified from original
# ===================================================

class adxl(object):
    def __init__(self):
        self.i2c = PY_I2C(accaddress, smbus.SMBus(1))
        self.address = accaddress
        self.i2c.write8(dataformat, dataformsetting)
        self.i2c.write8(powerctl, pwrsetting)

    def read(self):
        "Reads 16-bit values for x,y,z acceleration and returns in list"
        x0 = self.i2c.readS8(datax0)  # can't use 'readS16' b/c x0 is LSB
        x1 = self.i2c.readS8(datax1)  # see above for possible fix (justify)
        y0 = self.i2c.readS8(datay0)
        y1 = self.i2c.readS8(datay1)
        z0 = self.i2c.readS8(dataz0)
        z1 = self.i2c.readS8(dataz1)
        if x0 == -1 or x1 == -1 or y0 == -1 or y1 == -1 or z0 == -1 or z1 == -1:
            return [-1, -1, -1]
        xf = x0 | (x1 << 8)  # shift MSB and add LSB
        yf = y0 | (y1 << 8)
        zf = z0 | (z1 << 8)
        return [xf, yf, zf]

    def config(self):
        pass




# =======================================================
#
# 6. ITG-3200 gyroscope class (IMU digital combo)
#
# =======================================================
# Modified from sparkfun IMU Digital Combo Board code
# Inspiration source: https://github.com/sparkfun/IMU_Digital_Combo_Board
# Liscence info: Creative Commons Attribution-ShareAlike 4.0 Intl.
#                 Modified from original
# ===================================================

class gyro(object):
    def __init__(self):
        self.i2c = PY_I2C(gyraddress, smbus.SMBus(1))
        self.address = gyraddress
        self.i2c.write8(dlpffs, dlpffssel | dlpfcfg1)
        # set to full-scale and 1kHz internal sampling, 188Hz filter
        self.i2c.write8(smplrtdiv, 9)
        # divides 1kHz internal by 10 for 100Hz operation
        self.i2c.write8(intcfg, intcfgrawrdyen | intcfgitgrdyen)
        # set up both interrupts
        self.i2c.write8(pwrmgm, pwrmgmclkselx)  # PLL x gyro reference

    def read(self):
        xh = self.i2c.readS8(xouth)  # could use readS16 here bc 1st
        xL = self.i2c.readS8(xoutl)  # register is MSB/high byte
        yh = self.i2c.readS8(youth)
        yL = self.i2c.readS8(youtl)
        zh = self.i2c.readS8(zouth)
        zL = self.i2c.readS8(zoutl)
        if xh == -1 or xL == -1 or yh == -1 or yL == -1 or zh == -1 or zL == -1:
            return [-1, -1, -1, -1, -1, -1]
        xf = xL | (xh << 8)  # full ADC value
        xe = xf / 14.375  # convert to g-value
        yf = yL | (yh << 8)
        ye = yf / 14.375
        zf = zL | (zh << 8)
        ze = zf / 14.375
        return [xf, yf, zf, "%.3f" % xe, "%.3f" % ye, "%.3f" % ze]  # round to .000

    def config(self):
        pass




# =======================================================
#
# 7. MAG3110 magnetometer class
#
# =======================================================

class mag3110(object):
    def __init__(self):
        self.i2c = PY_I2C(magaddress, smbus.SMBus(1))
        self.address = magaddress
        self.i2c.write8(ctrlreg1, 0x00)  # initially set in standby mode
        self.i2c.write8(ctrlreg2, mrstraw)  # set to raw output mode
        self.i2c.write8(ctrlreg1, opmode)  # set to active mode (see above)

    def read(self):
        "Reads 16-bit values for x,y,z magnetic and returns in list"
        xmag = self.i2c.readS16(xmsb)
        ymag = self.i2c.readS16(ymsb)
        zmag = self.i2c.readS16(zmsb)
        # temp=self.i2c.readS8(dietemp)         # implementation omitted
        if xmag == -1 or ymag == -1 or zmag == -1:
            return [-1, -1, -1]
        return [xmag, ymag, zmag]

    def config(self):
        pass



# =======================================================
#
# 8. ADS1015 Analog-to-digital converter class
#
# =======================================================

class ADS1x15:
    i2c = None

    # IC Identifiers
    __IC_ADS1015 = 0x00

    # Pointer Register
    __ADS1015_REG_POINTER_MASK = 0x03
    __ADS1015_REG_POINTER_CONVERT = 0x00
    __ADS1015_REG_POINTER_CONFIG = 0x01
    __ADS1015_REG_POINTER_LOWTHRESH = 0x02
    __ADS1015_REG_POINTER_HITHRESH = 0x03

    # Config Register
    __ADS1015_REG_CONFIG_OS_MASK = 0x8000
    __ADS1015_REG_CONFIG_OS_SINGLE = 0x8000  # Write: Set to start a single-conversion
    __ADS1015_REG_CONFIG_OS_BUSY = 0x0000  # Read: Bit = 0 when conversion is in progress
    __ADS1015_REG_CONFIG_OS_NOTBUSY = 0x8000  # Read: Bit = 1 when device is not performing a conversion

    __ADS1015_REG_CONFIG_MUX_MASK = 0x7000
    __ADS1015_REG_CONFIG_MUX_DIFF_0_1 = 0x0000  # Differential P = AIN0, N = AIN1 (default)
    __ADS1015_REG_CONFIG_MUX_DIFF_0_3 = 0x1000  # Differential P = AIN0, N = AIN3
    __ADS1015_REG_CONFIG_MUX_DIFF_1_3 = 0x2000  # Differential P = AIN1, N = AIN3
    __ADS1015_REG_CONFIG_MUX_DIFF_2_3 = 0x3000  # Differential P = AIN2, N = AIN3
    __ADS1015_REG_CONFIG_MUX_SINGLE_0 = 0x4000  # Single-ended AIN0
    __ADS1015_REG_CONFIG_MUX_SINGLE_1 = 0x5000  # Single-ended AIN1
    __ADS1015_REG_CONFIG_MUX_SINGLE_2 = 0x6000  # Single-ended AIN2
    __ADS1015_REG_CONFIG_MUX_SINGLE_3 = 0x7000  # Single-ended AIN3

    __ADS1015_REG_CONFIG_PGA_MASK = 0x0E00
    __ADS1015_REG_CONFIG_PGA_6_144V = 0x0000  # +/-6.144V range
    __ADS1015_REG_CONFIG_PGA_4_096V = 0x0200  # +/-4.096V range
    __ADS1015_REG_CONFIG_PGA_2_048V = 0x0400  # +/-2.048V range (default)
    __ADS1015_REG_CONFIG_PGA_1_024V = 0x0600  # +/-1.024V range
    __ADS1015_REG_CONFIG_PGA_0_512V = 0x0800  # +/-0.512V range
    __ADS1015_REG_CONFIG_PGA_0_256V = 0x0A00  # +/-0.256V range

    __ADS1015_REG_CONFIG_MODE_MASK = 0x0100
    __ADS1015_REG_CONFIG_MODE_CONTIN = 0x0000  # Continuous conversion mode
    __ADS1015_REG_CONFIG_MODE_SINGLE = 0x0100  # Power-down single-shot mode (default)

    __ADS1015_REG_CONFIG_DR_MASK = 0x00E0
    __ADS1015_REG_CONFIG_DR_128SPS = 0x0000  # 128 samples per second
    __ADS1015_REG_CONFIG_DR_250SPS = 0x0020  # 250 samples per second
    __ADS1015_REG_CONFIG_DR_490SPS = 0x0040  # 490 samples per second
    __ADS1015_REG_CONFIG_DR_920SPS = 0x0060  # 920 samples per second
    __ADS1015_REG_CONFIG_DR_1600SPS = 0x0080  # 1600 samples per second (default)
    __ADS1015_REG_CONFIG_DR_2400SPS = 0x00A0  # 2400 samples per second
    __ADS1015_REG_CONFIG_DR_3300SPS = 0x00C0  # 3300 samples per second (also 0x00E0)

    __ADS1015_REG_CONFIG_CMODE_MASK = 0x0010
    __ADS1015_REG_CONFIG_CMODE_TRAD = 0x0000  # Traditional comparator with hysteresis (default)
    __ADS1015_REG_CONFIG_CMODE_WINDOW = 0x0010  # Window comparator

    __ADS1015_REG_CONFIG_CPOL_MASK = 0x0008
    __ADS1015_REG_CONFIG_CPOL_ACTVLOW = 0x0000  # ALERT/RDY pin is low when active (default)
    __ADS1015_REG_CONFIG_CPOL_ACTVHI = 0x0008  # ALERT/RDY pin is high when active

    __ADS1015_REG_CONFIG_CLAT_MASK = 0x0004  # Determines if ALERT/RDY pin latches once asserted
    __ADS1015_REG_CONFIG_CLAT_NONLAT = 0x0000  # Non-latching comparator (default)
    __ADS1015_REG_CONFIG_CLAT_LATCH = 0x0004  # Latching comparator

    __ADS1015_REG_CONFIG_CQUE_MASK = 0x0003
    __ADS1015_REG_CONFIG_CQUE_1CONV = 0x0000  # Assert ALERT/RDY after one conversions
    __ADS1015_REG_CONFIG_CQUE_2CONV = 0x0001  # Assert ALERT/RDY after two conversions
    __ADS1015_REG_CONFIG_CQUE_4CONV = 0x0002  # Assert ALERT/RDY after four conversions
    __ADS1015_REG_CONFIG_CQUE_NONE = 0x0003  # Disable the comparator and put ALERT/RDY in high state (default)


    # Dictionaries with the sampling speed values
    # These simplify and clean the code (avoid the abuse of if/elif/else clauses)
    spsADS1015 = {
        128: __ADS1015_REG_CONFIG_DR_128SPS,
        250: __ADS1015_REG_CONFIG_DR_250SPS,
        490: __ADS1015_REG_CONFIG_DR_490SPS,
        920: __ADS1015_REG_CONFIG_DR_920SPS,
        1600: __ADS1015_REG_CONFIG_DR_1600SPS,
        2400: __ADS1015_REG_CONFIG_DR_2400SPS,
        3300: __ADS1015_REG_CONFIG_DR_3300SPS
    }
    # Dictionariy with the programable gains
    pgaADS1x15 = {
        6144: __ADS1015_REG_CONFIG_PGA_6_144V,
        4096: __ADS1015_REG_CONFIG_PGA_4_096V,
        2048: __ADS1015_REG_CONFIG_PGA_2_048V,
        1024: __ADS1015_REG_CONFIG_PGA_1_024V,
        512: __ADS1015_REG_CONFIG_PGA_0_512V,
        256: __ADS1015_REG_CONFIG_PGA_0_256V
    }

    # Constructor
    def __init__(self, address=0x48, ic=__IC_ADS1015, debug=False):
        # Depending on if you have an old or a new Raspberry Pi, you
        # may need to change the I2C bus.  Older Pis use SMBus 0,
        # whereas new Pis use SMBus 1.  If you see an error like:
        # 'Error accessing 0x48: Check your I2C address '
        # change the SMBus number in the initializer below!
        self.i2c = Adafruit_I2C(address)
        self.address = address
        self.debug = debug

        # Make sure the IC specified is valid
        if ic < self.__IC_ADS1015:
            if self.debug:
                print "ADS1x15: Invalid IC specfied: %dh" % ic
        else:
            self.ic = ic

        # Set pga value, so that getLastConversionResult() can use it,
        # any function that accepts a pga value must update this.
        self.pga = 6144

    def read(self, channel=0, pga=6144, sps=250):
        "Gets a single-ended ADC reading from the specified channel in mV. \
    The sample rate for this mode (single-shot) can be used to lower the noise \
    (low sps) or to lower the power consumption (high sps) by duty cycling, \
    see datasheet page 14 for more info. \
    The pga must be given in mV, see page 13 for the supported values."

        ret_val = 0

        # With invalid channel return -1
        if channel > 3:
            if self.debug:
                print "ADS1x15: Invalid channel specified: %d" % channel
            return -1

        # Disable comparator, Non-latching, Alert/Rdy active low
        # traditional comparator, single-shot mode
        config = self.__ADS1015_REG_CONFIG_CQUE_NONE | \
                 self.__ADS1015_REG_CONFIG_CLAT_NONLAT | \
                 self.__ADS1015_REG_CONFIG_CPOL_ACTVLOW | \
                 self.__ADS1015_REG_CONFIG_CMODE_TRAD | \
                 self.__ADS1015_REG_CONFIG_MODE_SINGLE

        # Set sample per seconds, defaults to 250sps
        # If sps is in the dictionary (defined in init) it returns the value of the constant
        # otherwise it returns the value for 250sps. This saves a lot of if/elif/else code!
        if self.ic == self.__IC_ADS1015:
            config |= self.spsADS1015.setdefault(sps, self.__ADS1015_REG_CONFIG_DR_1600SPS)

        # Set PGA/voltage range, defaults to +-6.144V
        if (pga not in self.pgaADS1x15) & self.debug:
            print "ADS1x15: Invalid pga specified: %d, using 6144mV" % sps
        config |= self.pgaADS1x15.setdefault(pga, self.__ADS1015_REG_CONFIG_PGA_6_144V)
        self.pga = pga

        # Set the channel to be converted
        if channel == 3:
            config |= self.__ADS1015_REG_CONFIG_MUX_SINGLE_3
        elif channel == 2:
            config |= self.__ADS1015_REG_CONFIG_MUX_SINGLE_2
        elif channel == 1:
            config |= self.__ADS1015_REG_CONFIG_MUX_SINGLE_1
        else:
            config |= self.__ADS1015_REG_CONFIG_MUX_SINGLE_0

        # Set 'start single-conversion' bit
        config |= self.__ADS1015_REG_CONFIG_OS_SINGLE

        # Write config register to the ADC
        bytes = [(config >> 8) & 0xFF, config & 0xFF]
        if self.i2c.writeList(self.__ADS1015_REG_POINTER_CONFIG, bytes) == -1:
            return -1

        # Wait for the ADC conversion to complete
        # The minimum delay depends on the sps: delay >= 1/sps
        # We add 0.1ms to be sure
        delay = 1.0 / sps + 0.0001
        time.sleep(delay)

        # Read the conversion results
        result = self.i2c.readList(self.__ADS1015_REG_POINTER_CONVERT, 2)
        if result == -1:
            return result
        if self.ic == self.__IC_ADS1015:
            # Shift right 4 bits for the 12-bit ADS1015 and convert to mV
            return (((result[0] << 8) | (result[1] & 0xFF)) >> 4) * pga / 2048.0
        else:
            # Return a mV value for the ADS1115
            # (Take signed values into account as well)
            val = (result[0] << 8) | (result[1])
            if val > 0x7FFF:
                ret_val = (val - 0xFFFF) * pga / 32768.0
            else:
                ret_val = ((result[0] << 8) | (result[1])) * pga / 32768.0

        ret_val = ret_val / 1000 * 0.0123
        return ret_val

    def config(self):
        pass




# =======================================================
#
# 9. Adafruit_I2C class, hardware interface for ADC (Ultson)
#
# =======================================================
# Both read and write methods will return (-1) if error occurs
# ============================================================

class Adafruit_I2C(object):
    @staticmethod
    def getPiRevision():
        "Gets the version number of the Raspberry Pi board"
        # Revision list available at: http://elinux.org/RPi_HardwareHistory#Board_Revision_History
        try:
            with open('/proc/cpuinfo', 'r') as infile:
                for line in infile:
                    # Match a line of the form "Revision : 0002" while ignoring extra
                    # info in front of the revsion (like 1000 when the Pi was over-volted).
                    match = re.match('Revision\s+:\s+.*(\w{4})$', line)
                    if match and match.group(1) in ['0000', '0002', '0003']:
                        # Return revision 1 if revision ends with 0000, 0002 or 0003.
                        return 1
                    elif match:
                        # Assume revision 2 if revision ends with any other 4 chars.
                        return 2
                # Couldn't find the revision, assume revision 0 like older code for compatibility.
                return 0
        except:
            return 0

    @staticmethod
    def getPiI2CBusNumber():
        # Gets the I2C bus number /dev/i2c#
        return 1 if Adafruit_I2C.getPiRevision() > 1 else 0

    def __init__(self, address, busnum=-1, debug=False):
        self.address = address
        # By default, the correct I2C bus is auto-detected using /proc/cpuinfo
        # Alternatively, you can hard-code the bus version below:
        # self.bus = smbus.SMBus(0); # Force I2C0 (early 256MB Pi's)
        # self.bus = smbus.SMBus(1); # Force I2C1 (512MB Pi's)
        self.bus = smbus.SMBus(busnum if busnum >= 0 else Adafruit_I2C.getPiI2CBusNumber())
        self.debug = debug

    def reverseByteOrder(self, data):
        "Reverses the byte order of an int (16-bit) or long (32-bit) value"
        # Courtesy Vishal Sapre
        byteCount = len(hex(data)[2:].replace('L', '')[::2])
        val = 0
        for i in range(byteCount):
            val = (val << 8) | (data & 0xff)
            data >>= 8
        return val

    def errMsg(self):
        print "Error accessing 0x%02X: Check your I2C address" % self.address
        return -1

    def write8(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        try:
            self.bus.write_byte_data(self.address, reg, value)
            if self.debug:
                print "I2C: Wrote 0x%02X to register 0x%02X" % (value, reg)
        except IOError, err:
            return self.errMsg()

    def write16(self, reg, value):
        "Writes a 16-bit value to the specified register/address pair"
        try:
            self.bus.write_word_data(self.address, reg, value)
            if self.debug:
                print ("I2C: Wrote 0x%02X to register pair 0x%02X,0x%02X" %
                       (value, reg, reg + 1))
        except IOError, err:
            return self.errMsg()

    def writeRaw8(self, value):
        "Writes an 8-bit value on the bus"
        try:
            self.bus.write_byte(self.address, value)
            if self.debug:
                print "I2C: Wrote 0x%02X" % value
        except IOError, err:
            return self.errMsg()

    def writeList(self, reg, list):
        "Writes an array of bytes using I2C format"
        try:
            if self.debug:
                print "I2C: Writing list to register 0x%02X:" % reg
                print list
            self.bus.write_i2c_block_data(self.address, reg, list)
        except IOError, err:
            return self.errMsg()

    def readList(self, reg, length):
        "Read a list of bytes from the I2C device"
        try:
            results = self.bus.read_i2c_block_data(self.address, reg, length)
            if self.debug:
                print ("I2C: Device 0x%02X returned the following from reg 0x%02X" %
                       (self.address, reg))
                print results
            return results
        except IOError, err:
            return self.errMsg()

    def readU8(self, reg):
        "Read an unsigned byte from the I2C device"
        try:
            result = self.bus.read_byte_data(self.address, reg)
            if self.debug:
                print ("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" %
                       (self.address, result & 0xFF, reg))
            return result
        except IOError, err:
            return self.errMsg()

    def readS8(self, reg):
        "Reads a signed byte from the I2C device"
        try:
            result = self.bus.read_byte_data(self.address, reg)
            if result > 127: result -= 256
            if self.debug:
                print ("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" %
                       (self.address, result & 0xFF, reg))
            return result
        except IOError, err:
            return self.errMsg()

    def readU16(self, reg, little_endian=True):
        "Reads an unsigned 16-bit value from the I2C device"
        try:
            result = self.bus.read_word_data(self.address, reg)
            # Swap bytes if using big endian because read_word_data assumes little
            # endian on ARM (little endian) systems.
            if not little_endian:
                result = ((result << 8) & 0xFF00) + (result >> 8)
            if (self.debug):
                print "I2C: Device 0x%02X returned 0x%04X from reg 0x%02X" % (self.address, result & 0xFFFF, reg)
            return result
        except IOError, err:
            return self.errMsg()

    def readS16(self, reg, little_endian=True):
        "Reads a signed 16-bit value from the I2C device"
        try:
            result = self.readU16(reg, little_endian)
            if result > 32767: result -= 65536
            return result
        except IOError, err:
            return self.errMsg()




# ============================================================
#
# 10. PY_I2C class, hardware interface for Accel., Mag., Gyr.
#
# ============================================================
# I2C modified class
# Credit to: Antonio Casini, Adafruit
# Includes basic bitwise operations for sign/unsigned bytes,
#   as well as 16 bit registers
# Inspiration source:
#        https://github.com/tokask/python/blob/master/mag3110/tk_i2c.py
# Liscence: subject to Github terms and conditions (public)
# ============================================================
# Both read and write methods will return (-1) if error occurs
# ============================================================

class PY_I2C:
    def __init__(self, address, bus=smbus.SMBus(1)):  # SMBus(1)-bus# on RPi 2
        print address
        print bus
        self.address = address
        self.bus = bus

    def write8(self, reg, value):
        "Writes an 8-bit value to an register/address"
        try:
            self.bus.write_byte_data(self.address, reg, value)
        except IOError, err:
            print "Error accessing address"
            return -1

    def readU8(self, reg):
        "Reads an unsigned byte from the I2C device"
        try:
            result = self.bus.read_byte_data(self.address, reg)
            return result
        except IOError, err:
            print "Error accessing address"
            return -1

    def readS8(self, reg):
        "Reads a signed byte from the I2C device"
        try:
            result = self.bus.read_byte_data(self.address, reg)
            if (result > 127):
                return result - 256
            else:
                return result
        except IOError, err:
            print "Error accessing address"
            return -1

    def readU16(self, reg):
        "Reads a unsigned 16-bit value from the I2C device"
        # automatically increments register address by 1 for LSB read
        try:
            hibyte = self.bus.read_byte_data(self.address, reg)
            result = (hibyte << 8) + self.bus.read_byte_data(self.address, reg + 1)
            return result
        except IOError, err:
            print "Error accessing address"
            return -1

    def readS16(self, reg):
        "Reads a signed 16-bit value from the I2C device"
        # automatically increments register address by 1 for LSB read
        try:
            hibyte = self.bus.read_byte_data(self.address, reg)
            if (hibyte > 127):
                hibyte -= 256
            result = (hibyte << 8) + self.bus.read_byte_data(self.address, reg + 1)
            return result
        except IOError, err:
            print "Error accessing address"
            return -1



