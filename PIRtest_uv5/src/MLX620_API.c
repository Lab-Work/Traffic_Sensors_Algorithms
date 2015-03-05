/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_API.c,v $
*  $Author: dal $
*  $Date: 2013/05/10 08:29:24 $
*  $Revision: 1.10 $
*/

#include "MLX620_API.h"
#include <string.h>
#include <math.h>

/**
 * \file MLX620_API.c
 * \brief MLX90620 API source code.
 * */

uint16_t MLX620_RAMbuff[MLX620_RAM_SIZE_WORDS];
uint8_t MLX620_EEbuff[MLX620_EE_SIZE_BYTES];
/**
* \var MLX620_Ai[MLX620_IR_SENSORS]
* \brief IR pixels individual offset coefficients - Ai.
* \details It will be used for the calculations.
*/
int8_t MLX620_Ai[MLX620_IR_SENSORS];
/**
* \var MLX620_Bi[MLX620_IR_SENSORS]
* \brief Individual Ta dependence (slope) of IR pixels offset - Bi.
* \details It will be used for the calculations.
*/
double MLX620_Bi[MLX620_IR_SENSORS];
/**
* \var MLX620_Alphai[MLX620_IR_SENSORS]
* \brief Individual sensitivity coefficients - Alphai.
* \details It will be used during calculations.
*/
double MLX620_Alphai[MLX620_IR_SENSORS];
/**
* \var MLX620_DVtho
* \brief VTH0 of absolute temperature sensor.
* \details It will be used during calculations.
*/
double MLX620_DVtho;
/**
* \var MLX620_DKT1
* \brief KT1 of absolute temperature sensor.
* \details It will be used during calculations.
*/
double MLX620_DKT1;
/**
* \var MLX620_DKT2
* \brief KT2 of absolute temperature sensor.
* \details It will be used during calculations.
*/
double MLX620_DKT2;
/**
* \var MLX620_DLastTa
* \brief Last measured ambient temperature.
* \details It will be used during calculations.
*/
double MLX620_DLastTa;
/**
* \var MLX620_TGC
* \brief Thermal Gradient Coefficient.
* \details It will be used during calculations.
*/
double MLX620_TGC;
/**
* \var MLX620_CyclopsAlpha
* \brief Sensitivity coefficient of the compensation pixel.
* \details It will be used during calculations.
*/
double MLX620_CyclopsAlpha;
/**
* \var MLX620_CyclopsA
* \brief Compensation pixel individual offset.
* \details It will be used during calculations.
*/
double MLX620_CyclopsA;
/**
* \var MLX620_CyclopsB
* \brief Individual Ta dependence (slope) of compensation pixel offse.
* \details It will be used during calculations.
*/
double MLX620_CyclopsB;
/**
* \var MLX620_Ke
* \brief Emissivity.
* \details It will be used during calculations.
*/
double MLX620_Ke;
/**
* \var MLX620_CyclopsData
* \brief Last measured data from the compensation pixel (Cyclop).
* \details It will be used during calculations.
*/
double MLX620_CyclopsData;

/* ************************************************************* */
uint8_t MLX620_ReadRAM(uint8_t startAddr,
                        uint8_t addrStep,
                        uint8_t nWords,
                        uint8_t *pData)
{
  uint8_t writeBuffer[] = {MLX620_CMD_READ,
                           startAddr,
                           addrStep,
                           nWords};

  return MLX620_I2C_Driver_WriteRead(MLX620_ADDR,
                                    sizeof(writeBuffer),
                                    writeBuffer,
                                    nWords*sizeof(uint16_t),
                                    pData);
}
/* ************************************************************* */
uint8_t MLX620_ReadEEPROM(uint8_t startAddr,
                           uint16_t nBytes,
                           uint8_t *pData)
{
/**
  \def EE_I2C_CLOCK
  I2C clock delay in MCU cycles for EEPROM.
*/
#define EE_I2C_CLOCK 200
/**
  \def EE_I2C_START
  I2C start delay in MCU cycles for EEPROM.
*/
#define EE_I2C_START 400
/**
  \def EE_I2C_STOP
  I2C stop delay in MCU cycles for EEPROM.
*/
#define EE_I2C_STOP 400
/**
  \def EE_I2C_W_R
  I2C delay between write and read transmissions in MCU cycles for EEPROM.
*/
#define EE_I2C_W_R 400

  uint32_t err;
  uint32_t m_clock = MLX620_I2C_clock;
  uint32_t m_start = MLX620_I2C_start;
  uint32_t m_stop = MLX620_I2C_stop;
  uint32_t m_w_r = MLX620_I2C_W_R;

  MLX620_I2C_clock = EE_I2C_CLOCK;
  MLX620_I2C_start = EE_I2C_START;
  MLX620_I2C_stop = EE_I2C_STOP;
  MLX620_I2C_W_R = EE_I2C_W_R;

/** \note
 * Decrease the I2C clock speed temporary to read the EE.
 *
 */

  err = MLX620_I2C_Driver_WriteRead(MLX620_EEPROM_ADDR,
                                   sizeof(startAddr),
                                   &startAddr,
                                   nBytes,
                                   pData);

  MLX620_I2C_clock = m_clock;
  MLX620_I2C_start = m_start;
  MLX620_I2C_stop = m_stop;
  MLX620_I2C_W_R = m_w_r;

  return err;
}
/* ************************************************************* */
uint8_t MLX620_WriteConfig(uint16_t configReg)
{
  uint8_t pOut[] = {MLX620_CMD_WRITE_CONFIG,
                    (uint8_t)((configReg & 0xFF) - 0x55),
                    (uint8_t)(configReg & 0xFF),
                    (uint8_t)((configReg >> 8) - 0x55),
                    (uint8_t)(configReg >> 8)};
  return MLX620_I2C_Driver_Write(MLX620_ADDR,
                                sizeof(pOut),
                                pOut);
}
/* ************************************************************* */
uint8_t MLX620_ReadConfig (uint16_t *pConfigReg)
{
   return (MLX620_ReadRAM(MLX620_RAM_CONFIG,
                          0,
                          sizeof(uint16_t),
                          (uint8_t*)pConfigReg));
}
/* ************************************************************* */
uint8_t MLX620_WriteTrim(uint16_t trimReg)
{
  uint8_t pOut[] = {MLX620_CMD_WRITE_TRIM,
      (uint8_t)((trimReg & 0xFF) - 0xAA),
      (uint8_t)(trimReg & 0xFF),
      (uint8_t)((trimReg >> 8) - 0xAA),
      (uint8_t)(trimReg >> 8)};

  return MLX620_I2C_Driver_Write(MLX620_ADDR,
                      sizeof(pOut),
                      pOut);
}
/* ************************************************************* */
uint8_t MLX620_ReadTrim (uint16_t *pTrimReg)
{
  return (MLX620_ReadRAM(MLX620_RAM_TRIM,
                         0,
                         sizeof(uint16_t),
                         (uint8_t*)pTrimReg));
}
/* ************************************************************* */
uint8_t MLX620_StartSingleMeasurement(void)
{
  uint8_t pOut[2];
  pOut[0] = (MLX620_CMD_START & 0xFF);
  pOut[1] = (MLX620_CMD_START >> 8);

  return MLX620_I2C_Driver_Write(MLX620_ADDR,
                                sizeof(pOut),
                                pOut);
}
/* ************************************************************* */
uint32_t MLX620_Initialize(void)
{
 uint16_t confReg = 0;
 uint8_t trimReg = 0;
 uint32_t err;

  err = 0;

  err = MLX620_ReadEEPROM(MLX620_EE_IROffsetAi00,
                          MLX620_EE_SIZE_BYTES,
                          MLX620_EEbuff);
  if (err == 0)
  {
    memcpy(&trimReg, MLX620_EEbuff + MLX620_EE_OscTrim, sizeof(trimReg));
    if (trimReg == 0 || trimReg == 0xFF)
      err = MLX620_WriteTrim(85);
    else
      err = MLX620_WriteTrim(trimReg);

    if (err == 0)
    {
      memcpy(&confReg, MLX620_EEbuff + MLX620_EE_ConfReg, sizeof(confReg));

      if(confReg == 0 || confReg == 0xFFFF)
        err = MLX620_WriteConfig(MLX620_CONFIG_ADC_REFERENCE_MASK|
                                 MLX620_CONFIG_FPS_PTAT_MASK(3)|
                                 MLX620_CONFIG_POR_BROUT_BIT_MASK|
                                 MLX620_CONFIG_FPS_IR_MASK(0xE));
      else
        err = MLX620_WriteConfig(confReg);

      if(err == 0)
      {
        MLX620_CalcCommonParams();
      }
      else
        return 3;
    }
    else
      return 2;
  }
  else
    return 1;

  return err;
}
/* ************************************************************* */
void MLX620_CompensateIR(int16_t* pFrame, int start, int step, int count, double *pIR)
{
  //compensates full RAM buffer

  if (step == 0)
    step = 1;
  int end = start + count * step;

  // check which part of the the IR array is read -> compensate To
  if (start < MLX620_RAM_IR_BEG + MLX620_IR_SENSORS && end > MLX620_RAM_IR_BEG)
  {
    int irStart = (MLX620_RAM_IR_BEG - start) / step;
    if (irStart < 0)
      irStart = 0;
    int irEnd = (MLX620_RAM_IR_BEG + MLX620_IR_SENSORS - start) / step;
    if (irEnd > count)
      irEnd = count;
    int caddr = start + irStart * step;
    irEnd -= irStart;

    for (int i = 0; i < irEnd; ++i)
    {
      int16_t iv = pFrame[i];
      double vir_comp = MLX620_CalcToKelvin(caddr - MLX620_RAM_IR_BEG, iv);
      pIR[i] = vir_comp;
      iv = (uint16_t)(vir_comp * 50 + 0.5);
      pFrame[i] = iv;
      caddr += step;
    }

  }
}
/* ************************************************************* */
double MLX620_GetTaKelvin (int16_t ptat)
{
  MLX620_CalcTa(ptat);
  return MLX620_DLastTa + 273.15;
}
/* ************************************************************* */
void MLX620_CalcTGC(int16_t tgc)
{
  MLX620_CyclopsData = MLX620_TGC * (tgc - (MLX620_CyclopsA + MLX620_CyclopsB * (MLX620_DLastTa - 25)));
}
/* ************************************************************* */
uint32_t MLX620_CalcTa(int16_t ptat)
{
  if (fabs(MLX620_DKT2) < 1e-10 || fabs(MLX620_DVtho) < 1e-10)
    return 1;

  double d = (MLX620_DKT1*MLX620_DKT1 - 4 * MLX620_DKT2 * (1 - ptat / MLX620_DVtho));

  if (d < 0)
    return 1;

  MLX620_DLastTa = ((-MLX620_DKT1 +  sqrt(d)) / (2 * MLX620_DKT2)) + 25;

  return 0;
}
/* ************************************************************* */
double MLX620_CalcToKelvin(int idxIr, int16_t data)
{
  if (fabs(MLX620_Alphai[idxIr]) < 1e-10)
    return 0;

  double dt = MLX620_DLastTa + 273.15;    //convert to Kelvin
  dt *= dt;
  dt *= dt;   //power of 4

  double d = (data - (MLX620_Ai[idxIr] + MLX620_Bi[idxIr] * (MLX620_DLastTa - 25)) - MLX620_CyclopsData) /(MLX620_Ke * MLX620_Alphai[idxIr]) + dt;

  if (d < 0)
    return 0;

  double dTo = pow(d, 0.25);    //4th square

  return dTo;
}
/* ************************************************************* */
void MLX620_CalcCommonParams(void)
{
  uint16_t val16;
  uint8_t val8;
  int8_t s_val8;
  int16_t s_val16;
  uint8_t *ptr8;
  // Scale coefficients
  double dOffsetBScale;
  double dScaleComAlpha = 1;
  double dScaleAlpha = 1;
  // Common alpha
  double dCommonAlpha;

  ptr8 = MLX620_EEbuff;

  //  Vth(25) 16 signed
  memcpy(&val16, ptr8 + MLX620_EE_PtatVtho, sizeof(val16));
  if (val16 == 0)
    MLX620_DVtho = 1;
  else
    MLX620_DVtho = (double)((int16_t)val16);

  //  kT1 16 bit signed * 2^10
  memcpy(&val16, ptr8 + MLX620_EE_PtatKT1, sizeof(val16));
  MLX620_DKT1 = (double)((int16_t)val16) / MLX620_DVtho / (1 << 10);

  //  kT2 16 bit signed * 2^20
  memcpy(&val16, ptr8 + MLX620_EE_PtatKT2, sizeof(val16));
  MLX620_DKT2 = (double)((int16_t)val16) / MLX620_DVtho / (1 << 20);

  // Thermal Gradient Gompensation coefficient - MLX620_TGC
  memcpy(&s_val8, ptr8 + MLX620_EE_TGC, sizeof(s_val8));
  MLX620_TGC = s_val8 / 32.0;

  // Cyclops alpha
  memcpy(&val16, ptr8 + MLX620_EE_CyclopsAlpha, sizeof(val16));
  MLX620_CyclopsAlpha = val16;

  memcpy(&val8, ptr8 + MLX620_EE_IROffsetBScale, sizeof(val8));
  dOffsetBScale = pow(2, val8);

 // cyclops A & B
  memcpy(&s_val8, ptr8 + MLX620_EE_CyclopsA, sizeof(s_val8));
  MLX620_CyclopsA = s_val8;

  memcpy(&s_val8, ptr8 + MLX620_EE_CyclopsB, sizeof(s_val8));
  MLX620_CyclopsB = s_val8 / dOffsetBScale;

  memcpy(&val8, ptr8 + MLX620_EE_IRCommonSensScale, sizeof(val8));
  dScaleComAlpha = pow(2, val8);

  memcpy(&val8, ptr8 + MLX620_EE_IRSensScale, sizeof(val8));
  dScaleAlpha = pow(2, val8);

  memcpy(&val16, ptr8 + MLX620_EE_Ke, sizeof(val16));
  MLX620_Ke = val16 / 32768.0;

  memcpy(&val16, ptr8 + MLX620_EE_IRCommonSens, sizeof(val16));
  dCommonAlpha = ((double)val16 - (MLX620_TGC * MLX620_CyclopsAlpha)) / dScaleComAlpha;

  // Ai, Bi and MLX620_Alphai

  for (int i = 0; i < MLX620_IR_SENSORS; ++i)
  {
    memcpy(&val8, ptr8 + (MLX620_EE_IROffsetAi00 + i), sizeof(val8));
    MLX620_Ai[i] = (int8_t)val8;

    memcpy(&val8, ptr8 + (MLX620_EE_IROffsetBi00 + i), sizeof(val8));
    MLX620_Bi[i] = (double)((int8_t)val8) / dOffsetBScale;

    memcpy(&val8, ptr8 + (MLX620_EE_IRSens00 + i), sizeof(val8));
    MLX620_Alphai[i] = (double)val8 / dScaleAlpha + dCommonAlpha;

  }

}
/* ************************************************************* */
int16_t MLX620_GetRawIR(uint8_t row, uint8_t column)
{
  return ((int16_t) MLX620_RAMbuff[MLX620_RAM_IR_BEG + MLX620_IR_SENSOR_IDX(row, column)]);
}
/* ************************************************************* */
uint16_t MLX620_GetPTAT(void)
{
  return ((uint16_t) MLX620_RAMbuff[MLX620_RAM_PTAT]);
}
/* ************************************************************* */
int16_t MLX620_GetTGC(void)
{
  return ((int16_t) MLX620_RAMbuff[MLX620_RAM_TGC]);
}
/* ************************************************************* */

