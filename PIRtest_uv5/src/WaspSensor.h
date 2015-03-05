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
#ifndef Sensor_h
#define Sensor_h

/******************************************************************************
 * Includes
 ******************************************************************************/
 
#include <inttypes.h>




/******************************************************************************
 * Class
 ******************************************************************************/
 
  //! WaspXBee Class
/*!
	WaspXBee Class defines all the variables and functions used to manage the UART related with the XBee
 */
class WaspSensor
{
  private:
	uint8_t _uart;

	//void printNumber(unsigned long n, uint8_t base);
	//uint8_t SlaveAdress;
  public:


	  
	//! class constructor

	WaspSensor();
	void init();

	int readSensor (uint8_t slaveadress,uint8_t address);
	int readSensor (uint8_t slaveadress,int address);
	int writeSensor (uint8_t slaveadress,uint8_t address, uint8_t value);
	int writeSensor (uint8_t slaveadress,int address, uint8_t value);


};

extern WaspSensor Sensor;

#endif

