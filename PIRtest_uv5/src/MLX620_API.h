/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_API.h,v $
*  $Author: dal $
*  $Date: 2013/04/19 09:52:47 $
*  $Revision: 1.9 $
*/

#ifndef _MLX620_API_H_
#define _MLX620_API_H_

#include "MLX620_I2C_Device.h"
#include "MLX620_I2C_Driver.h"


/** \mainpage MLX90620 Firmware API

\section intro_sec Contents

  This is a Firmware API for MLX90620.\n

  It could be used with any given MCU and integrated in a project. Only minimal modifications in MLX620_I2C_Driver.h are required as the driver is MCU defendant. The rest of the code is MCU and tool chain independent.
  The API is compiled and tested on EVB90620, which is using Atmel's AT91SAM7S265 MCU. GCC compiler was used to build the API, but any commercial development environment could be used as well.

  \subsection I2C_driver I2C Driver
   - Software implementation of I2C driver for writing and reading data to and from the sensor.\n
   - The software driver is almost MCU independent as it only uses the general purpose input/output controller (module) of the particular MCU.\n
   For more detailed description please refer to the source code MLX620_I2C_Driver.c and MLX620_I2C_Driver.h

  \subsection I2C_interface I2C interface and configuration bits
  - I2C communication interface description - MLX620_I2C_Device.h
  - Configuration register bits description - MLX620_I2C_Device.h

  \subsection RAM_EEPROM RAM and EEPROM memory descriptions
  - RAM memory map and description - MLX620_RAM.h
  - EEPROM memory map and description - MLX620_EEPROM.h

  \subsection API_functions API Functions
  - configuring the sensor (device)
  - starting measurement
  - reading measurement data
  - calculating the Object temperature (compensation).
  For detailed description of the source code please refer to MLX620_API.c and MLX620_API.h

  \subsection API_Demo API Demonstration
  - initializing the sensor and reporting (printing) the error
  - printing the values of the "Configuration" and "Trimming" Registers, after the initialization is done
  - reading raw IR temperature data for each pixel, as well as reading Ambient Temperature sensor
  - compensating the printing the Ambient Temperature [Kelvin]
  - compensating the printing the Object's Temperature for each pixel from 1 to 64\n
  Go to the source files description for more information MLX620_Demo.c

For more information please refer to the sensor's data sheet: http://www.melexis.com/Asset/Datasheet-IR-thermometer-16X4-sensor-array-MLX90620-DownloadLink-6099.aspx

To see the code documentation please click on the "Files" tab above.


*/

/**
 * \file MLX620_API.h
 * \brief MLX90620 API header file
*/

#include <stdint.h>
#include "MLX620_I2C_Device.h"
#include "MLX620_I2C_Driver.h"
#include "MLX620_RAM.h"
#include "MLX620_EEPROM.h"

/**
  \def MLX620_IR_ROWS
  Number of rows in IR array.
*/
#define MLX620_IR_ROWS (4U)
/**
  \def MLX620_IR_COLUMNS
  Number of columns in IR array.
*/
#define MLX620_IR_COLUMNS (16U)
/**
  \def MLX620_IR_SENSORS
  Number of sensors in IR array.
*/
#define MLX620_IR_SENSORS (MLX620_IR_ROWS * MLX620_IR_COLUMNS)
/**
  \def MLX620_IR_SENSOR_IDX(row, col)
  Index of a IR sensor in a 1D array.
*/
#define MLX620_IR_SENSOR_IDX(row, col) ((col) * MLX620_IR_COLUMNS + (row))
/**
  \var MLX620_RAMbuff
  \brief Buffer of the device RAM memory.
*/
extern uint16_t MLX620_RAMbuff[MLX620_RAM_SIZE_WORDS];
/**
  \var IRtempK[MLX620_IR_SENSORS]
  \brief Compensated IR .
*/

double IRtempK[MLX620_IR_SENSORS];
/**
  \var MLX620_EEbuff
  \brief A buffer of the device EEPROM memory.
*/
extern uint8_t MLX620_EEbuff[MLX620_EE_SIZE_BYTES];
/** \fn uint8_t MLX620_ReadRAM(uint8_t startAddr, uint8_t addrStep, uint8_t nWords, uint8_t *pData)
* \brief Reads device RAM memory.
* \details It should be used to read measurement data from the device.
* \param[in] startAddr Address of the first word to be read.
* \param[in] addrStep Address increment value.
* \param[in] nWords Number of words to be read.
* \param[out] *pData Pointer to the buffer address to store the read data. The buffer should be at least equal to nWords * 2 bytes;
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_ReadRAM(uint8_t startAddr,
                      uint8_t addrStep,
                      uint8_t nWords,
                      uint8_t *pData);
/** \fn uint8_t MLX620_ReadEEPROM(uint8_t startAddr, uint16_t nBytes, uint8_t *pData)
* \brief Reads the EEPROM memory inside the 90620 device.
* \details It should be used to read the calibration constants.
* \param[in] startAddr Address of the first byte to be read.
* \param[in] nBytes Number of bytes to be read.
* \param[out] *pData Pointer to the buffer address to store the read data.
*             The buffer should be at least equal to nBytes;
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_ReadEEPROM(uint8_t startAddr,
                         uint16_t nBytes,
                         uint8_t *pData);
/** \fn uint8_t MLX620_WriteConfig(uint16_t configReg)
* \brief Writes the configuration register.
* \param[in] configReg Configuration register value.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_WriteConfig(uint16_t configReg);
/** \fn uint8_t MLX620_ReadConfig (uint16_t *pConfigReg)
* \brief Reads the configuration register.
* \param[out] *pConfigReg Pointer to a variable which should contain the configuration register value.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_ReadConfig (uint16_t *pConfigReg);
/** \fn uint8_t MLX620_WriteTrim(uint16_t trimReg)
* \brief Writes the trim register.
* \param[in] trimReg Trimming calibration coefficients read from EEPROM.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_WriteTrim(uint16_t trimReg);
/** \fn uint8_t MLX620_ReadTrim (uint16_t *pTrimReg)
* \brief Reads the trimming register.
* \param[out] *pTrimReg Pointer to a variable which should contain the trimming register value.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_ReadTrim (uint16_t *pTrimReg);
/** \fn uint8_t MLX620_StartSingleMeasurement(void)
* \brief Sends a start measurement command. Doesn't check if the device is in step measurement mode.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX620_StartSingleMeasurement(void);
/** \fn uint32_t MLX620_Initialize(void)
* \brief Initialize MLX90620 according to this procedure.
* \details Read the full EEPROM;
* \details Initialize the Configuration and Trimming registers;
* \details Calculate the common parameters;
* \retval err I2C acknowledge bit; '0' - ACK received; '1' - ACK NOT received.
*/
uint32_t MLX620_Initialize(void);
/** \fn int16_t MLX620_GetRawIR(uint8_t row, uint8_t column)
* \brief Returns the raw IR measurement result at the given position, taken from the device RAM buffer.
* \details Raw data is without any compensation.
* \retval rawIR Raw IR measurement result.
*/
int16_t MLX620_GetRawIR(uint8_t row, uint8_t column);
/** \fn uint16_t MLX620_GetPTAT(void)
* \brief Returns the measurement result of PTAT sensor.
* \retval Ta Result from ambient temperature measurement.
*/
uint16_t MLX620_GetPTAT(void);
/** \fn int16_t MLX620_GetTGC(void)
* \brief Returns the measurement result of the Temperature Gradient Compensation, taken from the current RAM buffer.
* \retval tgc TGC measurement result.
*/
int16_t MLX620_GetTGC(void);
/** \fn uint32_t MLX620_CalcTa(int16_t ptat)
* \brief Calculates the last measured ambient temperature and store it to \a MLX620_DLastTa.
* \retval err error if '1', no error if '0'.
*/
uint32_t MLX620_CalcTa(int16_t ptat);
/** \fn double MLX620_CalcToKelvin(int idxIr, int16_t data)
* \brief Calculates the real object temperature for a single pixel in Kelvin.
* \param[in] idxIr IR sensor index.
* \param[in] data IR raw data.
* \retval To Object temperature.
*/
double MLX620_CalcToKelvin(int idxIr, int16_t data);
/** \fn void MLX620_CalcCommonParams(void)
* \brief Calculates the common parameters(coefficients) and store them in global variables.
* \details It updates the following coefficients:
* \details MLX620_Ai[MLX620_IR_SENSORS];
* \details MLX620_Bi[MLX620_IR_SENSORS];
* \details MLX620_Alphai[MLX620_IR_SENSORS];
* \details MLX620_DVtho;
* \details MLX620_DKT1;
* \details MLX620_DKT2;
* \details MLX620_DLastTa;
* \details MLX620_TGC;
* \details MLX620_CyclopsAlpha;
* \details MLX620_CyclopsA;
* \details MLX620_CyclopsB;
* \details MLX620_Ke;
* \details MLX620_KsTa;
* \details MLX620_CyclopsData;
*/
void MLX620_CalcCommonParams(void);
/** \fn void MLX620_CompensateIR(int16_t* pFrame, int start, int step, int count, double *pIR)
* \brief Performs compensation over measurement samples.
* \details Depending on the \a start \a step and \a count parameters, it compensates To and/or Ta
* \details The same buffer is used for input and output data.
* \details Returned values are signed int (16-bit) multiplied by 50.
* \param[in,out] *pFrame pointer to RAM buffer.
* \param[in] start RAM start address.
* \param[in] step address step.
* \param[in] count RAM addresses count.
* \param[out] *pIR pointer to a buffer to store real temperature values in Kelvin.
* \retval To Object temperature.
*/
void MLX620_CompensateIR(int16_t* pFrame, int start, int step, int count, double *pIR);
/** \fn void MLX620_CalcTGC(int16_t tgc)
 * \brief Calculates the Temperature Gradient Compensation sensor data \a MLX620_CyclopsData.
 * \param[in] tgc TGC (Cyclop) sensor raw data.
 */
void MLX620_CalcTGC(int16_t tgc);
/** \fn double MLX620_GetTaKelvin(int16_t ptat)
 * \brief Calculates the real ambient temperature in Kelvin.
 * \param[in] ptat PTAT sensor raw data.
 * \retval Ta Ambient temperature in Kelvin.
 */
double MLX620_GetTaKelvin (int16_t ptat);

#endif  /* _MLX620_API_H_ */

