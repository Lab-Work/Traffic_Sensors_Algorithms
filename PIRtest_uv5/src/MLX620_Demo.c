/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_Demo.c,v $
*  $Author: dal $
*  $Date: 2013/04/19 09:52:47 $
*  $Revision: 1.6 $
*/

#define MLXDEMONULL 0

#define MLXDEMO 1

#define RUNMLXDEMO MLXDEMONULL

#if RUNMLXDEMO == MXLDEMO

#include "MLX620_API.h"
#include <stdlib.h >

/**
 * \file MLX620_Demo.C
 * \brief MLX90620 API functions demonstration usage.
 * \details This demonstration usage of the API functions performing the most necessary operations in order to get full frame of the sensor printed out using standard 'printf' function.\n
 * The users of this demo should implement the 'printf' function for the particular MCU and compiler that is used. If the printing functionality is not needed, it should be disabled.
 * This demo is performing the following operations:
 * - initializing the sensor and reporting (printing) the error
 * - printing the values of the "Configuration" and "Trimming" Registers, after the initialization is done
 * - reading raw IR temperature data for each pixel, as well as reading Ambient Temperature sensor
 * - compensating the printing the Ambient Temperature [Kelvin]
 * - compensating the printing the Object's Temperature for each pixel from 1 to 64
 */

/** \fn uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf)
* \brief Initialize the sensor.
* \details It should be used only when the sensor is supplied.
* \param[out] *trim Trimming register value after the configuration is done.
* \param[out] *conf Configuration register value after the configuration is done.
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf);

/** \fn uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta)
* \brief Read measurement data from the sensor and calculate ambient and Infra Red (object's) temperature in Kelvin.
* \details The temperature results for each pixel is saved in .\n
* \param[in] *pIRtempK Pointer to buffer where the temperature results for each pixel would be stored
* \param[out] *Ta Ambient temperature in Kelvin
* \retval ack I2C acknowledge bit.
*/
uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta);


uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf)
{
  uint8_t ack;

  ack = MLX620_Initialize();      //initialize the sensor
  if (ack == MLX620_ACK)
  {
    ack = MLX620_ReadTrim(trim);    //read the Trimming register and return it
    ack |= MLX620_ReadConfig(conf); //read the Configuration register and return it
  }

  return ack;
}

uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta)
{
  uint8_t ack;
  int16_t ptat, tgc;

  //get RAW (not compensated) ambient temperature sample (PTAT sensor)
  ack = MLX620_ReadRAM(MLX620_RAM_PTAT, 0, 1, (uint8_t*)&ptat);

  if (ack == MLX620_ACK)
  {
    //compensate ambient temperature; get absolute temperature in Kelvin
    *Ta = MLX620_GetTaKelvin (ptat);

    ack = MLX620_ReadRAM(MLX620_RAM_TGC, 0, 1, (uint8_t*)&tgc);

    if (ack == MLX620_ACK)
    {
      MLX620_CalcTGC(tgc);

      ack = MLX620_ReadRAM(MLX620_RAM_IR_BEG, 1, MLX620_IR_SENSORS, (uint8_t*)MLX620_RAMbuff);

      if (ack == MLX620_ACK)
      {
        MLX620_CompensateIR((int16_t*)MLX620_RAMbuff, MLX620_RAM_IR_BEG, 1, MLX620_IR_SENSORS, pIRtempK);

      }
    }
  }

  return ack;
}
/** \fn int main(void)
* \brief main
*/
int main(void)
{
	

  uint8_t ack;      //I2C acknowledge bit
  double Ta;        //Ambient Temperature
  uint8_t pixIdx;   //pixel index
  uint16_t trimReg, confReg;

  ack = MLX90620_InitializeSensor(&trimReg, &confReg);

  if (ack == MLX620_ACK)
  {
//    printf("Sensor initialized successfully\n")
//    printf("Triming Register = %X\n, trimReg");
//    printf("Configuration Register = %X\n, confReg");
  }
  else
  {
//    printf("ERROR: Sensor initiazation failed!\n");
  }

	for (;;)
	{
	  if(ack == MLX620_ACK)
	  {
	    ack = MLX90620_MeasureTemperature(IRtempK, &Ta);

	    if(ack == MLX620_ACK)
	    {
//        printf("Ambient Temperature = %4.2 \n", Ta);

        for(pixIdx = 0; pixIdx < MLX620_IR_SENSORS; pixIdx++)
        {
//          printf("Infrared Temperature pixel[%u] = %4.2 \n", pixIdx, IRtempK[pixIdx]);
        }
	    }
	    else
	    {
//	      printf("ERROR: Reading data from the sensor failed!\n");
	    }
	  }
	  else
	  {
	    return 0;
	  }

	  MLX620_I2C_Driver_NOPdelay(10000);   //wait some time
	}

}

#endif


