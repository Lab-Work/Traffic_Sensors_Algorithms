
#Yes really, it's not 0x60, the EEPROM has its own address (I've never seen this before)
MLX90620_EEPROM_WRITE = 0xA0
MLX90620_EEPROM_READ = 0xA1

# The sensor's I2C address is 0x60. So 0b.1100.000W becomes 0xC0
MLX90620_WRITE = 0xC0
MLX90620_READ = 0xC1

#These are commands
CMD_READ_REGISTER = 0x02

#Begin registers

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

#Common sensitivity coefficients
CAL_A0_L = 0xE0
CAL_A0_H = 0xE1
CAL_A0_SCALE = 0xE2
CAL_DELTA_A_SCALE = 0xE3
CAL_EMIS_L = 0xE4
CAL_EMIS_H = 0xE5

#KSTA

#Config register = 0xF5-F6

OSC_TRIM_VALUE = 0xF7

#Bits within configuration register 0x92

POR_TEST = 10










