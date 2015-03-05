/*
    Copyright (C) 2009 Libelium Comunicaciones Distribuidas S.L.
    http://www.libelium.com
 
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 2.1 of the License, or
 *  (at your option) any later version.
   
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
  
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
    Version:		0.14

    Design:		David Gascón

    Implementation:	David Cuartielles, Alberto Bielsa
*/
 
  


#ifndef __WASPCONSTANTS_H__
#define __WASPCONSTANTS_H__

/******************************************************************************
 * Includes
 ******************************************************************************/
 


/******************************************************************************
 * Definitions & Declarations
 ******************************************************************************/

// internal peripherals flag (IPF) register
// it just re-arranges the PRR0 and PRR1 registers from the MCU:
// MSB	-	7	6	5	4	3	2	1	  0
// ---------------------------------------------------------------------------------- 
// PRR0	-	PRTWI	PRTIM2 	PRTIM0	?	PRTIM1	PRSPI	PRUSART0  PRADC
// PRR1	-	?	?	?	?	PRTIM3 	?	? 	  PRUSART1

// IPRA register
// has the three MSB positions empty
#define IPADC 1
#define IPTWI 2
#define IPSPI 4
#define IPUSART0 8
#define IPUSART1 16

// IPRB register
// has the four MSB positions empty
#define IPTIM0 1
#define IPTIM1 2
#define IPTIM2 4
#define IPTIM3 8

// external peripherals flag (EPF) register and constants
// '1' marks should be present, '0' marks not installed
// the EPF should be hardcoded in EEPROM on manufacturing
// a function will then check through the flags if the peripheral is
// installed on a certain WASP device, there is room for others
// that could come on the possible add-on boards

// EPRA register
#define PGPS 1
#define PSD 2
#define PACC 4
#define PXBEE 8
#define PRTC 16
#define PUSB 32
#define PSID 64
#define PSLOW_CLK 128

// EPRB register - for GPRS and add-on boards
#define PGPRS 1


// Analog Pins
#define	ANALOG0	0
#define	ANALOG1	1
#define	ANALOG2	2
#define	ANALOG3	3
#define	ANALOG4	4
#define	ANALOG5	5
#define	ANALOG6	6
#define	ANALOG7	7

// WASP interrupt vector
// this is a 16 bits interrupt vector to contain
// the flags for the different HW/SW interrupts
// 0 will denote not-active, 1 will denote active
// the HW interrupts and default callback functions are
// stored inside WInterrupts.c

#define	HAI_INT		1 // High Active Interrupt
#define	LAI_INT		2
#define ACC_INT   	4
#define BAT_INT   	8
//#define RTC_INT   	16
#define WTD_INT   	32
#define TIM0_INT  	64
#define TIM1_INT  	128
#define TIM2_INT  	256
#define PIN_INT   	512
#define UART0_INT 	1024
#define UART1_INT 	2048
#define	SENS_INT	4096
#define	ANE_INT		8192
#define	PLV_INT		16384
#define	HIB_INT		32768

// Interrupt Counter Vector
#define	HAI_POS		0
#define	LAI_POS		1
#define	ACC_POS		2
#define	BAT_POS		3
#define	RTC_POS		4
#define	WTD_POS		5
#define	TIM0_POS	6
#define	TIM1_POS	7
#define	TIM2_POS	8
#define	PIN_POS		9
#define	UART0_POS	10
#define	UART1_POS	11
#define	SENS_POS	12
#define	SENS2_POS	13

// sensor's interrupt pin
#define ACC_INT_ACT 		2 
#define	ACC_INT_PIN_MON		RDY_ACC 

#define	HAI_INT_ACT		2 
#define	HAI_INT_PIN_MON		I2C_SDA 

#define	LAI_INT_ACT		3 
#define	LAI_INT_PIN_MON		I2C_SDA 

#define	BAT_INT_ACT		3 
#define	BAT_INT_PIN_MON		LOW_BAT_MON

#define	RTC_INT_ACT		2 
#define	RTC_INT_PIN_MON		RST_RTC

#define	WTD_INT_ACT		4 
#define	WTD_INT_PIN_MON		DIGITAL0

#define	UART1_INT_ACT		2
#define	UART1_INT_PIN_MON	GPRS_PIN 

// Event Sensor Board
#define	SENS_INT_ACT		2
#define	SENS_INT_PIN_MON 	DIGITAL2 
#define	SENS_INT_CLK_REG	DIGITAL7
#define	SENS_INT_DO		DIGITAL1
#define	SENS_INT_ENABLE		DIGITAL8
#define	SENS_INT_CLK_INH	DIGITAL3

// Smart Metering Sensor Board
#define	SENS_INT_SMART_ACT		2
#define	SENS_INT_SMART_PIN_MON 		DIGITAL2 
#define	SENS_INT_SMART_CLK_REG		19
#define	SENS_INT_SMART_DO		18
#define	SENS_INT_SMART_ENABLE		20
#define	SENS_INT_SMART_CLK_INH		17

// Meteo Sensor Board
#define	ANE_INT_ACT		2
#define	PLV_INT_ACT		3
#define SENS2_INT_PIN_MON	DIGITAL5 
#define SENS2_INT_PIN2_MON	DIGITAL3

#define MAX_ARGS 20    		// max amount of arguments in Wasp
#define MAX_ARG_LENGTH 16    	// max length for Wasp arguments

#endif
