"""////////////////////////////////////////////////////////////////////////////
Author: Will Barbour
Date: October 2, 2015

Convert orginal hexidecimal PIR data to decimal format. This version may
contain bugs. Please check with Will (wbarbour1@illinois.edu) should any
problems occur.
////////////////////////////////////////////////////////////////////////////"""

__author__ = 'wbarbour1'

import numpy as np

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

class PIR3_converter:
    def __init__(self):

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

        self.ts_pir1 = 0
        self.ts_pir2 = 0
        self.ts_pir3 = 0

        # initialize the three pir sensors
        self.pir_1 = PIRSensor(1, alpha_ij_PIR1, EEPROM_PIR1)
        self.pir_2 = PIRSensor(2, alpha_ij_PIR2, EEPROM_PIR2)
        self.pir_3 = PIRSensor(3, alpha_ij_PIR3, EEPROM_PIR3)

        self.pir_1.const_init()
        self.pir_2.const_init()
        self.pir_3.const_init()

        self.config()

    # config_str = [ PIR123 ON/OFF (1/0), pixel_list]
    # pixel_list = [px1, px2, ..., pxn]
    # assuming px = [0-191]
    def config(self):
        pixel_list = range(0, 191)
        # set convert_flag to appropriate value T/F
        self.convert_flag = True
        print "pixel conversion ON"
        # changes pixel index to [row, col] and appends to pixel_str, which is called in read()
        for pixel in pixel_list:
            px = [(pixel // 64) + 1, [(pixel % 64) // 16, (pixel % 64) % 16]]       # [PIR123, [px row, px col]]
            self.pixel_str.append(px)
            #print "convert pixel ", px
        print self.pixel_str

    # reads the full serial string from the Arduino
    # if conversion is requested by the class configuration,
    # the string will be parsed and the specified pixels will be converted
    def convert(self, line):

        if self.convert_flag is True:
            calc_temp_return = []
            # parse the data string and extract the data to structures
            self.parse_IR_string_line(line)
            if not self.ptat_pir1 == 0:
                for i in self.pixel_str:
                    pixel_row = i[1][0]
                    pixel_col = i[1][1]
                    if i[0] == 1:
                        calc_temp_return.append(self.pir_1.calculate_single_pixel_temperature(self.ptat_pir1, [[pixel_row, pixel_col], self.temp_pir1_4x16[pixel_row, pixel_col]], self.cpix_pir1))
                    if i[0] == 2:
                        calc_temp_return.append(self.pir_2.calculate_single_pixel_temperature(self.ptat_pir2, [[pixel_row, pixel_col], self.temp_pir2_4x16[pixel_row, pixel_col]], self.cpix_pir2))
                    if i[0] == 3:
                        calc_temp_return.append(self.pir_3.calculate_single_pixel_temperature(self.ptat_pir3, [[pixel_row, pixel_col], self.temp_pir3_4x16[pixel_row, pixel_col]], self.cpix_pir3))
                return [self.ts_pir1, self.ts_pir2, self.ts_pir3] + calc_temp_return
        return [-1]


    # parse the IR data string line
    # extract the IRdata matrix, and update temperature 4x16
    # line length 406 bytes; each () is 4 char
    # ($$$$)[(ID)(time)(64x2 bytes)(ptat)(cpix)(ultra)][pir2][pir3](counter)
    # new
    # ($$$$)[(IDID)(time 4 characters in millis())(64x4 char)(ptat 4 char)(cpix 4 char)][pir2][pir3](counter 4 char)
    def parse_IR_string_line(self, line):
        #print line
        line_list = [line[i:i + 4] for i in range(0, len(line), 4)]
        #print line_list
        ################################################
        # Parse the first sensor
        index = 2

        self.ts_pir1 = int(line_list[index], 16)
        index += 1
        # line string comes in with pixels grouped by column
        # temp_pir#_4x16 indexed properly by (row, col)
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
            return -1
        else:
            index += 1

        self.ts_pir2 = int(line_list[index], 16)
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
            return -1
        else:
            index += 1

        self.ts_pir3 = int(line_list[index], 16)
        index += 1
        for col in range(0, 16):
            for row in range(0, 4):
                self.temp_pir3_4x16[row, col] = self.sign_int(int(line_list[index], 16))
                index += 1

        self.ptat_pir3 = int(line_list[index], 16)
        index += 1
        self.cpix_pir3 = self.sign_int(int(line_list[index], 16))
        index += 1


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

    # calculate TA
    def calculate_TA(self, ptat):
        self.Tambient = (-self.k_t1 + np.sqrt(np.power(self.k_t1, 2) -
                                              (4 * self.k_t2 * (self.v_th - ptat)))) / (2 * self.k_t2) + 25
        #print "ambient temperature: ", self.Tambient

    # calculate single pixel
    # iraData = [[2,3], Hex]
    def calculate_single_pixel_temperature(self, ptat, irData, cpix):
        #print "calculating single pixel temp ", irData
        self.calculate_TA(ptat)

        if self.Tambient <= -100:
            self.Tambient = 0  # reset, do not proceed to compute the To
        else:
            return self.calculate_single_pixel_TO(irData, cpix)
        return -1


    # iraData = [[2,3], Hex]
    def calculate_single_pixel_TO(self, irData, cpix):

        #print "making temp calculations"
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
