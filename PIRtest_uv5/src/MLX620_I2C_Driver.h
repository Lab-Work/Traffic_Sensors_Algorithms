/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_I2C_Driver.h,v $
*  $Author: dal $
*  $Date: 2013/04/19 09:52:48 $
*  $Revision: 1.9 $
************************************************************* */

/*
This is a revised version of the driver using the I2C in STM32F407
Yanning Li
March 02, 2015
*/

#ifndef _MLX620_I2C_Driver_H_
#define _MLX620_I2C_Driver_H_

#include <stdint.h>

/** \file MLX620_I2C_Driver.h
 *  \brief I2C communication driver header file.
 *  \details Modify this file in order to use the general purpose IO functions of the particular MCU that you're using.
 *  \details The definitions which should be modified are marked with <b>MODIFY HERE</b>
 */


extern uint32_t MLX620_I2C_clock;
extern uint32_t MLX620_I2C_start;
extern uint32_t MLX620_I2C_stop;
extern uint32_t MLX620_I2C_W_R;

/**
  \def PIN_SCL
  \brief Hardware pin name at which the I2C clock line is mapped.
  \details For example AT91C_PA3_TWD for AT91SAM7Sxxxx. <b>MODIFY HERE</b>
*/
// #define PIN_SCL AT91C_PA4_TWCK
// Yanning
#define PIN_SCL GPIO_Pin_10

/**
  \def PIN_SDA
  \brief Hardware pin name at which the I2C data line is mapped.
  \details For example AT91C_PA3_TWD for AT91SAM7Sxxxx. <b>MODIFY HERE</b>
  
*/
// #define PIN_SDA AT91C_PA3_TWD
// Yanning
#define PIN_SDA GPIO_Pin_11

// Yanning
// Modified as delay(2us) in MLX620_I2C_Driver.c
// #ifndef _NOP
/*
  \def _NOP()
  \brief MCU No OPeration instruction definition.
  \details "MODIFY THIS"
*/
//  #define _NOP() asm volatile ("mov r0, r0");
// #endif
/**
  \def MLX620_ACK
  Acknowledge bit value.
*/
#define MLX620_ACK  (0U)
/**
  \def MLX620_NACK
  Not acknowledge bit value.
*/
#define MLX620_NACK (1U)
/**
  \def MLX620_TRUE
  TRUE value.
*/
#define MLX620_TRUE  (1U)
/**
  \def MLX620_FALSE
  FALSE value.
*/
#define MLX620_FALSE       (0U)
/**
  \def MLX620_I2C_SET_PIN(pin)
 * \param[in] pin to be set
 * \brief Sets output pin to '1'.
 * \details <b>MODIFY HERE</b>
*/


// #define MLX620_I2C_SET_PIN(pin) AT91F_PIO_SetOutput(AT91C_BASE_PIOA, (pin))
// Yanning
// GPIOB is for SCL and SDA
#define MLX620_I2C_SET_PIN(pin) GPIO_SetBits(GPIOB, (pin))

/**
  \def MLX620_I2C_CLR_PIN(pin)
 * \param[in] pin to be cleared
 * \brief Clears output pin to '0'.
 * \details <b>MODIFY HERE</b>
*/
// #define MLX620_I2C_CLR_PIN(pin) AT91F_PIO_ClearOutput(AT91C_BASE_PIOA, (pin))
// Yanning
// GPIOB is for SCL and SDA
#define MLX620_I2C_CLR_PIN(pin) GPIO_ResetBits(GPIOB, (pin))

/**
  \def MLX620_I2C_GET_PIN(pin)
 * \param[in] pin to be read
 * \brief Read the status of an input pin.
 * \details <b>MODIFY HERE</b>
*/
// #define MLX620_I2C_GET_PIN(pin) AT91F_PIO_IsInputSet(AT91C_BASE_PIOA, (pin))
// Yanning
// GPIOB only for SCL and SDA
#define MLX620_I2C_GET_PIN(pin) GPIO_ReadInputDataBit(GPIOB, (pin))

/** \fn void MLX620_I2C_Driver_SendStart (void)
 *  \brief Send a Start condition over the I2C line.
*/
void MLX620_I2C_Driver_SendStart (void);

/** \fn void MLX620_I2C_Driver_SendStop (void)
 *  \brief Send a Stop condition over the I2C line.
*/
void MLX620_I2C_Driver_SendStop (void);

/** \fn void MLX620_I2C_Driver_NOPdelay (uint32_t nops)
 *  \brief Perform a CPU delay by using NOPs.
 *  \param[in] nops Number of NOPs to be executed.
*/
void MLX620_I2C_Driver_NOPdelay (uint32_t nops);

/** \fn void MLX620_I2C_Driver_Init (uint32_t clock, uint32_t start, uint32_t stop, uint32_t w_r)
* \brief Initialize the I2C Driver.
* \param[in] clock Clock delay.
* \param[in] start Start delay.
* \param[in] stop Start delay.
* \param[in] w_r Delay between write and read transmissions.
*/
void MLX620_I2C_Driver_Init (uint32_t clock,
                            uint32_t start,
                            uint32_t stop,
                            uint32_t w_r);
/** \fn uint32_t MLX620_I2C_Driver_WriteByte(uint8_t byte)
*   \brief Write a single byte into a slave device.
*   \param[in] byte to be written.
*   \retval ack I2C acknowledge bit.
*/
uint32_t MLX620_I2C_Driver_WriteByte(uint8_t byte);
/** \fn uint8_t MLX620_I2C_Driver_ReadByte(uint32_t ack)
 *  \brief Read single byte from a slave device.
 *  \param[in] ack Acknowledge bit to be sent.
 *  \retval byte Data byte read from the slave device.
*/
uint8_t MLX620_I2C_Driver_ReadByte(uint32_t ack);
/** \fn uint32_t MLX620_I2C_Driver_Write(uint8_t slaveAddr, uint32_t nBytes, uint8_t *pData)
* \brief Write multiple data bytes into a slave device.
* \param[in] slaveAddr Address of the slave device.
* \param[in] nBytes Number of bytes to write.
* \param[in] *pData Pointer to a data buffer.
* \retval ack I2C acknowledge bit.
*/
uint32_t MLX620_I2C_Driver_Write(uint8_t slaveAddr,
                                uint32_t nBytes,
                                uint8_t *pData);
/** \fn uint32_t MLX620_I2C_Driver_Read(uint8_t slaveAddr, uint32_t nBytes, uint8_t *pData)
* \brief Read multiple data from a slave device.
* \param[in] slaveAddr Address of the slave device.
* \param[in] nBytes Number of bytes to read.
* \param[out] *pData Pointer to a data buffer.
* \retval ack I2C acknowledge bit.
*/
uint32_t MLX620_I2C_Driver_Read(uint8_t slaveAddr,
                               uint32_t nBytes,
                               uint8_t *pData);
/** \fn uint32_t MLX620_I2C_Driver_WriteRead(uint8_t slaveAddr, uint32_t nBytesWrite, uint8_t *pWriteData, uint32_t nBytesRead, uint8_t *pReadData)
* \brief Write and then read multiple data to and from a slave device.
* \details Write only and Read only operations are also possible.
* \param[in] slaveAddr Address of the slave device.
* \param[in] nBytesWrite Number of bytes to write.
* \param[in] *pWriteData Pointer to a write data buffer.
* \param[in] nBytesRead Number of bytes to read.
* \param[out] *pReadData Pointer to a read data buffer.
* \retval ack I2C acknowledge bit.
*/
uint32_t MLX620_I2C_Driver_WriteRead(uint8_t slaveAddr,
                                    uint32_t nBytesWrite,
                                    uint8_t *pWriteData,
                                    uint32_t nBytesRead,
                                    uint8_t *pReadData);

#endif	/* _MLX620_I2C_Driver_H_ */
