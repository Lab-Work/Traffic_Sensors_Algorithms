/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_I2C_Driver.c,v $
*  $Author: dal $
*  $Date: 2013/04/19 09:52:47 $
*  $Revision: 1.8 $
************************************************************* */

/*
This is a revised version of the driver using the I2C in STM32F407
Yanning Li
March 02, 2015
*/


/** \file MLX620_I2C_Driver.c
 *  \brief I2C communication driver source code
 *  \details Do not modify this file if not needed.
 */

#include "MLX620_I2C_Driver.h"
// #include <AT91SAM7S256.h>
// #include <lib_AT91SAM7S256.h>
//Yanning
#include <stm32f4xx_gpio.h>
#include "dingke_delay.h"
#include "dingke_smbus.h"

/** \var  MLX620_I2C_clock
 * Clock delay in MCU cycles. not sure, but try 3000 us*/
uint32_t MLX620_I2C_clock;
/** \var  MLX620_I2C_start
 * Start delay in MCU cycles. 2 us*/
uint32_t MLX620_I2C_start;
/** \var  MLX620_I2C_stop
 * Stop delay in MCU cycles. 2 us*/
uint32_t MLX620_I2C_stop;
/** \var  MLX620_I2C_W_R
 * Delay between write and read transmissions, in MCU cycles.  6 us */
uint32_t MLX620_I2C_W_R;


// Yanning
// Modified function originally from MLX620_I2C_Dirver.h
#ifndef _NOP
/**
  \def _NOP()
  \brief MCU No OPeration instruction definition.
  \details "MODIFY THIS"
*/
  #define _NOP() delay_us(2);
#endif


void MLX620_I2C_Driver_SendStart (void)
{
  uint32_t delay = MLX620_I2C_start;

  MLX620_I2C_Driver_NOPdelay(delay);
  MLX620_I2C_SET_PIN(PIN_SCL);
  MLX620_I2C_Driver_NOPdelay(delay);

  MLX620_I2C_CLR_PIN(PIN_SDA);
  MLX620_I2C_Driver_NOPdelay(delay);

}
void MLX620_I2C_Driver_SendStop (void)
{
  uint32_t delay = MLX620_I2C_stop;

  MLX620_I2C_CLR_PIN(PIN_SDA);
  MLX620_I2C_Driver_NOPdelay(delay);

  MLX620_I2C_SET_PIN(PIN_SCL);
  MLX620_I2C_Driver_NOPdelay(delay);

  MLX620_I2C_SET_PIN(PIN_SDA);
  MLX620_I2C_Driver_NOPdelay(delay);
}

// Yanning
// originally written as nops, change to us
/*void MLX620_I2C_Driver_NOPdelay (uint32_t nops)
{
  uint32_t i;

  for(i = 0; i < nops; i++)
  {
    _NOP();
  }

}
*/
// delay in us
void MLX620_I2C_Driver_NOPdelay (uint32_t us)
{
  delay_ms(us);
}

/*void MLX620_I2C_Driver_Init (uint32_t clock,
                            uint32_t start,
                            uint32_t stop,
                            uint32_t w_r)
{
  AT91F_PIO_CfgOutput(AT91C_BASE_PIOA, PIN_SCL|PIN_SDA);
  AT91C_BASE_PIOA->PIO_MDER = PIN_SCL|PIN_SDA;
  AT91F_PIO_SetOutput(AT91C_BASE_PIOA, PIN_SCL|PIN_SDA);
  AT91C_BASE_PIOA->PIO_PPUDR = PIN_SDA|PIN_SCL;

  MLX620_I2C_clock = clock;
  MLX620_I2C_start = start;
  MLX620_I2C_stop = stop;
  MLX620_I2C_W_R = w_r;
}
*/

// Yanning
void MLX620_I2C_Driver_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
						     
	SMBus_poweron();
	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
	
	//I2C_SCL PB.8   I2C_SDA PB.9 
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10 |GPIO_Pin_11 ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
		
	GPIO_SetBits(GPIOB,GPIO_Pin_10);//SCL  
	GPIO_SetBits(GPIOB,GPIO_Pin_11);//SDA
	
	// those are defined as us
	// The following values are to be tested
		MLX620_I2C_clock = 3000;
  	MLX620_I2C_start = 5;
  	MLX620_I2C_stop = 5;
  	MLX620_I2C_W_R = 10;
}



uint32_t MLX620_I2C_Driver_WriteByte(uint8_t byte)
{
  uint32_t i, clock;

  clock = MLX620_I2C_clock;

  for (i = 0; i < 8; i ++)
  {
    MLX620_I2C_CLR_PIN(PIN_SCL);
    if (byte & (1<<7))
    {
      MLX620_I2C_SET_PIN(PIN_SDA);
    }
    else
    {
      MLX620_I2C_CLR_PIN(PIN_SDA);
    }
    MLX620_I2C_Driver_NOPdelay(clock);
    MLX620_I2C_SET_PIN(PIN_SCL);
    MLX620_I2C_Driver_NOPdelay(clock);
    byte <<= 1;
  }

  _NOP();
  _NOP();

  MLX620_I2C_CLR_PIN(PIN_SCL);
  MLX620_I2C_SET_PIN(PIN_SDA);
  _NOP();
  MLX620_I2C_Driver_NOPdelay(clock);
  _NOP();
  _NOP();
  _NOP();
  MLX620_I2C_SET_PIN(PIN_SCL);

  if (MLX620_I2C_GET_PIN(PIN_SDA))
    i = MLX620_NACK;  //reuse i as acknowledge status
  else
    i = MLX620_ACK;

  MLX620_I2C_Driver_NOPdelay(clock);
  _NOP();

  MLX620_I2C_CLR_PIN(PIN_SCL);

  return i;
}

uint8_t MLX620_I2C_Driver_ReadByte(uint32_t ack)
{
  uint32_t i, clock, data;

   clock = MLX620_I2C_clock;
   data = 0;

   MLX620_I2C_CLR_PIN(PIN_SCL);
   MLX620_I2C_SET_PIN(PIN_SDA);

   for (i = 0; i < 8; i ++)
   {
     MLX620_I2C_CLR_PIN(PIN_SCL);
     _NOP();
     _NOP();
     _NOP();
     data <<= 1;

     MLX620_I2C_Driver_NOPdelay(clock);

     MLX620_I2C_SET_PIN(PIN_SCL);

     if (MLX620_I2C_GET_PIN(PIN_SDA))
       data |= 1;
     else
       data &= ~1;

     MLX620_I2C_Driver_NOPdelay(clock);
   }

   _NOP();
   MLX620_I2C_CLR_PIN(PIN_SCL);
   if (ack)
     MLX620_I2C_CLR_PIN(PIN_SDA);
   else
     MLX620_I2C_SET_PIN(PIN_SDA);

   _NOP();
   MLX620_I2C_Driver_NOPdelay(clock);

   MLX620_I2C_SET_PIN(PIN_SCL);
   _NOP();
   _NOP();
   _NOP();
   MLX620_I2C_Driver_NOPdelay(clock);
   _NOP();
   _NOP();

   MLX620_I2C_CLR_PIN(PIN_SCL);
   MLX620_I2C_SET_PIN(PIN_SDA);

   return ((uint8_t)data);
}

uint32_t MLX620_I2C_Driver_Write(uint8_t slaveAddr,
                                  uint32_t nBytes,
                                  uint8_t *pData)
{
	uint32_t i, ack;

	slaveAddr <<= 1;
	slaveAddr &= ~1;    //W/R bit = W
	ack = 0;
	
	MLX620_I2C_Driver_SendStart();

	ack = MLX620_I2C_Driver_WriteByte(slaveAddr);

	for(i = 0; i < nBytes; i++)
	{
		ack |= MLX620_I2C_Driver_WriteByte(*pData++);
	}

	MLX620_I2C_Driver_SendStop();

	return ack;
}
uint32_t MLX620_I2C_Driver_Read(uint8_t slaveAddr,
                                 uint32_t nBytes,
                                 uint8_t *pData)
{
	uint32_t i, ack;

	slaveAddr <<= 1;
	slaveAddr |= 1;    //W/R bit = R
	
	MLX620_I2C_Driver_SendStart();

	ack = MLX620_I2C_Driver_WriteByte(slaveAddr);


	for(i = 0; i < nBytes - 1; i++)
	{
		*pData++ = MLX620_I2C_Driver_ReadByte(1);
		
	}
	*pData = MLX620_I2C_Driver_ReadByte(0);

	MLX620_I2C_Driver_SendStop();

	return ack;
}
uint32_t MLX620_I2C_Driver_WriteRead(uint8_t slaveAddr,
                                    uint32_t nBytesWrite,
                                    uint8_t *pWriteData,
                                    uint32_t nBytesRead,
                                    uint8_t *pReadData)
{
	uint32_t i, ack, delay;

	slaveAddr <<= 1;
	slaveAddr &= ~1;  //W/R bit = W
	ack = 0;
	delay = MLX620_I2C_W_R;

	if (nBytesWrite)
	{
		MLX620_I2C_Driver_SendStart();

		ack = MLX620_I2C_Driver_WriteByte(slaveAddr);

		for(i = 0; i < nBytesWrite; i++)
		{
			ack |= MLX620_I2C_Driver_WriteByte(*pWriteData++);
		}
	}

	if (nBytesRead)
	{
		MLX620_I2C_Driver_NOPdelay(delay);

		slaveAddr |= 1;  //W/R bit = R
		MLX620_I2C_Driver_SendStart();

		ack |= MLX620_I2C_Driver_WriteByte(slaveAddr);

		for(i = 0; i < nBytesRead - 1; i++)
		{
			*pReadData++ = MLX620_I2C_Driver_ReadByte(1);
		}
    *pReadData = MLX620_I2C_Driver_ReadByte(0);
	}

	if (nBytesWrite || nBytesRead)
		MLX620_I2C_Driver_SendStop();

	return ack;

}

