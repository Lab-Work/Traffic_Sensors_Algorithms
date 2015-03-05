/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_I2C_Device.h,v $
*  $Author: dal $
*  $Date: 2012/11/08 16:39:46 $
*  $Revision: 1.5 $
************************************************************* */

#ifndef _MLX620_I2C_Device_H_
#define _MLX620_I2C_Device_H_

/**
 * \file MLX620_I2C_Device.h
 * \brief MLX90620 device I2C related definitions
 * \details It contains the following definitions:
 * - I2C slave addresses;
 * - I2C commands;
 * - Configuration register bit masks.
 * \details  For more detailed information please refer to the product data sheet.
 */

/**
  \def MLX620_ADDR
  I2C slave address of the MLX90620 device (RAM memory).
*/
#define MLX620_ADDR	(0x60U)
/**
  \def MLX620_EEPROM_ADDR
  I2C slave address of the MLX90620 EEPROM memory.
*/
#define MLX620_EEPROM_ADDR	(0x50U)
/**
  \def MLX620_CMD_START
  I2C command used to start a single shot measurement. Available only in Step mode
*/
#define MLX620_CMD_START (0x801U)
/**
  \def MLX620_CMD_READ
  I2C command used to read measurement, configuration and other data from the chip.
*/
#define MLX620_CMD_READ (2U)
/**
  \def MLX620_CMD_WRITE_CONFIG
  I2C command used to write the configuration register.
*/
#define MLX620_CMD_WRITE_CONFIG (3U)
/**
  \def MLX620_CMD_WRITE_TRIM
  I2C command used to write the trimming register.
*/
#define MLX620_CMD_WRITE_TRIM (4U)
/**
  \def MLX620_CONFIG_FPS_IR_MASK(fps)
  \a fps bit mask for the IR measurement.
*/
#define MLX620_CONFIG_FPS_IR_MASK(fps)	(((fps) & 0x0F) << 0)
/**
  \def MLX620_CONFIG_MEAS_STEP_CONT_MASK
  This parameter defined the measurement mode: continuous (active level – '0') or step mode (active level – '1').
*/
#define MLX620_CONFIG_MEAS_STEP_CONT_MASK 	(1 << 6)
/**
  \def MLX620_CONFIG_SLEEP_REQUEST_MASK
  Writing '1' to this bit puts the chip in sleep mode. Writing 0 has no effect.
*/
#define MLX620_CONFIG_SLEEP_REQUEST_MASK 	(1 << 7)
/**
  \def MLX620_CONFIG_TA_MEAS_RUNNING_MASK
  Shows if there is ambient temperature measurement running with active level  '1'. Write to it has no effect.
*/
#define MLX620_CONFIG_TA_MEAS_RUNNING_MASK 	(1 << 8)
/**
  \def MLX620_CONFIG_IR_MEAS_RUNNING_MASK
  Shows if there is IR measurement running with active level '1'. Write to it has no effect.
*/
#define MLX620_CONFIG_IR_MEAS_RUNNING_MASK 	(1 << 9)
/**
  \def MLX620_CONFIG_POR_BROUT_BIT_MASK
  Should be written to '1' during configuration.
  If it's read '0' - POR or Brown-out occurred, need to reload CONFIG register again.
*/
#define MLX620_CONFIG_POR_BROUT_BIT_MASK 	(1 << 10)
/**
  \def MLX620_CONFIG_FMPLUS_ENABLE_MASK
  Enable/Disable the I2C fast mode plus (1MHz) clock speed. '0' - FMPLUS is enabled; '1' - FMPLUS is disabled.
*/
#define MLX620_CONFIG_FMPLUS_ENABLE_MASK (1 << 11)
/**
  \def MLX620_CONFIG_FPS_PTAT_MASK(fps)
  \a fps bit mask for the PTAT measurement.
*/
#define MLX620_CONFIG_FPS_PTAT_MASK(fps)	(((fps) & 0x03) << 12)
/**
  \def MLX620_CONFIG_ADC_REFERENCE_MASK
  Enable the ADC HIGH or LOW reference. '0' - ADC HIGH reference is enabled; '1' - ADC LOW reference is enabled.
*/
#define MLX620_CONFIG_ADC_REFERENCE_MASK (1 << 14)

#endif  /* _MLX620_I2C_Device_H_ */

