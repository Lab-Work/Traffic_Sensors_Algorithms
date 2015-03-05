/*! \file WaspXBee.h
    \brief Library for managing the UART related with the XBee
    
    Copyright (C) 2009 Libelium Comunicaciones Distribuidas S.L.
    http://www.libelium.com
 
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.
   
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
  
    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
    Version:		0.1

    Design:		David Gascón

    Implementation:	David Cuartielles, Alberto Bielsa

 */
 
 
/*! \def XBee_h
    \brief The library flag
    
 */
#ifndef Eeprom_h
#define Eeprom_h

/******************************************************************************
 * Includes
 ******************************************************************************/
 
#include <inttypes.h>
#include "dingke_i2c.h"



/******************************************************************************
 * Class
 ******************************************************************************/
 
  //! WaspXBee Class
/*!
	WaspXBee Class defines all the variables and functions used to manage the UART related with the XBee
 */
class WaspEEPROM
{
  private:
	//uint8_t _uart;

	//void printNumber(unsigned long n, uint8_t base);
	uint8_t SlaveAdress;
  public:


	  
	//! class constructor

	WaspEEPROM();
	
	void ON();
	void OFF();
	void begin();
	void start();
	void WriteProtection(int flag);
	void SoftwareReset (void);

	int setSlaveAddress(uint8_t SlaveAdress);
	int writeEEPROM (int address, uint8_t value);
	int  readEEPROM (int address);

	int writeEEPROM (uint8_t SlaveAdress,int address, uint8_t value);
	int readEEPROM (uint8_t SlaveAdress,int address);

	int writeEEPROMStr(uint8_t slaveadress,uint16_t writeaddr,uint8_t *pbuffer,uint16_t strlen);
	int  readEEPROMStr(uint8_t slaveadress,uint16_t  readaddr,uint8_t *pbuffer,uint16_t strlen);
};

extern WaspEEPROM Eeprom;

#endif

