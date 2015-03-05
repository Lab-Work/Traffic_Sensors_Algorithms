/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_EEPROM.h,v $
*  $Author: dal $
*  $Date: 2012/11/09 14:15:07 $
*  $Revision: 1.6 $
************************************************************* */

#ifndef _MLX620_EEPROM_H_
#define _MLX620_EEPROM_H_

/**
 * \file MLX620_EEPROM.h
 * \brief MLX90620 EEPROM memory map.
 * \details  For more detailed information please refer to the product data sheet.
 * */

/**
  \def MLX620_EE_SIZE_BYTES
  Device EEPROM memory size expressed in bytes.
*/
#define MLX620_EE_SIZE_BYTES (256U)
/**
  \def MLX620_EE_IROffsetAi00
  IR pixels individual offset coefficients - Ai.
*/
#define MLX620_EE_IROffsetAi00   (0x0U)
/**
  \def MLX620_EE_IROffsetBi00
  Individual Ta dependence (slope) of IR pixels offset - Bi.
*/
#define MLX620_EE_IROffsetBi00   (0x40U)
/**
  \def MLX620_EE_IRSens00
  Individual sensitivity coefficients.
*/
#define MLX620_EE_IRSens00      (0x80U)
/**
  \def MLX620_EE_MLX_Reserved1
  Melexis reserved part 1.
*/
#define MLX620_EE_MLX_Reserved1  (0xC0U)
/**
  \def MLX620_EE_CyclopsA
  Compensation pixel (Cyclop)individual offset.
*/
#define MLX620_EE_CyclopsA      (0xD4U)
/**
  \def MLX620_EE_CyclopsB
  Individual Ta dependence (slope) of compensation pixel (Cyclop) offset.
*/
#define MLX620_EE_CyclopsB      (0xD5U)
/**
  \def MLX620_EE_CyclopsAlpha
  Sensitivity coefficient of the compensation pixel (16-bit).
*/
#define MLX620_EE_CyclopsAlpha  (0xD6U)
/**
  \def MLX620_EE_TGC
  Thermal Gradient Coefficient.
*/
#define MLX620_EE_TGC             (0xD8U)
/**
  \def MLX620_EE_IROffsetBScale
  Scaling coefficient for slope of IR pixels offset.
*/
#define MLX620_EE_IROffsetBScale  (0xD9U)
/**
  \def MLX620_EE_PtatVtho
  VTH0 of absolute temperature sensor (16-bit).
*/
#define MLX620_EE_PtatVtho        (0xDAU)
/**
  \def MLX620_EE_PtatKT1
  KT1 of absolute temperature sensor (16-bit).
*/
#define MLX620_EE_PtatKT1         (0xDCU)
/**
  \def MLX620_EE_PtatKT2
  KT2 of absolute temperature sensor (16-bit).
*/
#define MLX620_EE_PtatKT2         (0xDEU)
/**
  \def MLX620_EE_IRCommonSens
  Common sensitivity coefficient of IR pixels (16-bit).
*/
#define MLX620_EE_IRCommonSens    (0xE0U)
/**
  \def MLX620_EE_IRCommonSensScale
  Scaling coefficient for common sensitivity.
*/
#define MLX620_EE_IRCommonSensScale (0xE2U)
/**
  \def MLX620_EE_IRSensScale
  Scaling coefficient for individual sensitivity.
*/
#define MLX620_EE_IRSensScale       (0xE3U)
/**
  \def MLX620_EE_Ke
  Emissivity  (16-bit).
*/
#define MLX620_EE_Ke                (0xE4U)
/**
  \def MLX620_EE_KsTa
  KsTa  (16-bit).
*/
#define MLX620_EE_KsTa              (0xE6U)
/**
  \def MLX620_EE_MLX_Reserved2
  Melexis reserved part 2.
*/
#define MLX620_EE_MLX_Reserved2   (0xE8U)
/**
  \def MLX620_EE_ConfReg
  Configuration register  (16-bit).
*/
#define MLX620_EE_ConfReg         (0xF5U)
/**
  \def MLX620_EE_OscTrim
  Oscillator trim register (16-bit).
*/
#define MLX620_EE_OscTrim         (0xF7U)
/**
  \def MLX620_EE_ChipID
  Chip ID (64-bit).
*/
#define MLX620_EE_ChipID         (0xF8U)

#endif  /* _MLX620_EEPROM_H_ */
        
