/*
*  Copyright (C) 2011 Melexis N.V.
*  $RCSfile: MLX620_RAM.h,v $
*  $Author: dal $
*  $Date: 2012/11/07 17:39:23 $
*  $Revision: 1.4 $
************************************************************* */

#ifndef _MLX620_RAM_H_
#define _MLX620_RAM_H_

/**
 * \file MLX620_RAM.h
 * \brief MLX90620 RAM memory map.
 * \details For more detailed information please refer to the product data sheet.
 * */

/**
  \def MLX620_RAM_SIZE_WORDS
  Device RAM memory size expressed in 16-bit words.
  The device's RAM memory is organized in 16-bit words.
*/
#define MLX620_RAM_SIZE_WORDS (0xFFU)
/**
  \def MLX620_RAM_SIZE_BYTES
  Device RAM memory size expressed in bytes.
*/
#define MLX620_RAM_SIZE_BYTES (MLX620_RAM_SIZE_WORDS * 2)
/**
  \def MLX620_RAM_IR_BEG
  Begging of area to store measurement result for each IR sensor.
*/
#define	MLX620_RAM_IR_BEG		(0x0U)
/**
  \def MLX620_RAM_IR_END
  End of area to store measurement result for each IR sensor.
*/
#define	MLX620_RAM_IR_END		(0x3FU)
/**
  \def MLX620_RAM_PTAT
  Measurement result of the PTAT sensor.
*/
#define MLX620_RAM_PTAT			(0x90U)
/**
  \def MLX620_RAM_TGC
  Measurement result of the Temperature Gradient Compensation sensor.
*/
#define MLX620_RAM_TGC			(0x91U)
/**
  \def MLX620_RAM_CONFIG
  Configuration register.
*/
#define	MLX620_RAM_CONFIG		(0x92U)
/**
  \def MLX620_RAM_TRIM
  Trim register.
*/
#define	MLX620_RAM_TRIM			(0x93U)

#endif  /* _MLX620_RAM_H_ */
