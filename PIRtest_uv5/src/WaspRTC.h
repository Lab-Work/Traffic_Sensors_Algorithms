/*! \file WaspRTC.h
    \brief Library for managing the RTC
    
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
  
    Version:		0.8

    Design:		David Gasc髇

    Implementation:	Alberto Bielsa, David Cuartielles, Marcos Yarza

*/

 /*! \def WaspRTC_h
    \brief The library flag
    
  */
#ifndef WaspRTC_h
#define WaspRTC_h

/******************************************************************************
 * Includes
 ******************************************************************************/

#include <inttypes.h>
// #include <time.h>
/******************************************************************************
 * Definitions & Declarations
 ******************************************************************************/

// RTC ADDRESSES CONSTANTS
/*! \def RTC_SECONDS_ADDRESS
    \brief RTC Addresses constants. Seconds register in this case
 */
/*! \def RTC_MINUTES_ADDRESS
    \brief RTC Addresses constants. Minutes register in this case
 */
/*! \def RTC_HOURS_ADDRESS
    \brief RTC Addresses constants. Hours register in this case
 */
/*! \def RTC_DAYS_ADDRESS
    \brief RTC Addresses constants. Days register in this case
 */
/*! \def RTC_DATE_ADDRESS
    \brief RTC Addresses constants. Date register in this case
 */
/*! \def RTC_MONTH_ADDRESS
    \brief RTC Addresses constants. Month register in this case
 */
/*! \def RTC_YEAR_ADDRESS
    \brief RTC Addresses constants. Year register in this case
 */
/*! \def RTC_ALM1_START_ADDRESS
    \brief RTC Addresses constants. Alarm1 start address in this case
 */
/*! \def RTC_ALM1_SECONDS_ADDRESS
    \brief RTC Addresses constants. Alarm1 seconds register in this case
 */
/*! \def RTC_ALM1_MINUTES_ADDRESS
    \brief RTC Addresses constants. Alarm1 minutes register in this case
 */
/*! \def RTC_ALM1_HOURS_ADDRESS
    \brief RTC Addresses constants. Alarm1 hours register in this case
 */
/*! \def RTC_ALM1_DAYS_ADDRESS
    \brief RTC Addresses constants. Alarm1 days register in this case
 */
/*! \def RTC_ALM2_START_ADDRESS
    \brief RTC Addresses constants. Alarm2 start address in this case
 */
/*! \def RTC_ALM2_MINUTES_ADDRESS
    \brief RTC Addresses constants. Alarm2 minutes register in this case
 */
/*! \def RTC_ALM2_HOURS_ADDRESS
    \brief RTC Addresses constants. Alarm2 hours register in this case
 */
/*! \def RTC_ALM2_DAYS_ADDRESS
    \brief RTC Addresses constants. Alarm2 days register in this case
 */
/*! \def RTC_CONTROL_ADDRESS
    \brief RTC Addresses constants. Control register in this case
 */
/*! \def RTC_STATUS_ADDRESS
    \brief RTC Addresses constants. Status register in this case
 */
/*! \def RTC_MSB_TEMP_ADDRESS
    \brief RTC Addresses constants. MSB Temperature register in this case
 */
/*! \def RTC_LSB_TEMP_ADDRESS
    \brief RTC Addresses constants. LSB Temperature register in this case
 */
#define	RTC_SECONDS_ADDRESS		0x00	
#define	RTC_MINUTES_ADDRESS		0x01
#define	RTC_HOURS_ADDRESS		0x02
#define	RTC_DAYS_ADDRESS		0x03
#define	RTC_DATE_ADDRESS		0x04
#define	RTC_MONTH_ADDRESS		0x05
#define	RTC_YEAR_ADDRESS		0x06
#define	RTC_ALM1_START_ADDRESS		0x07
#define	RTC_ALM1_SECONDS_ADDRESS	0x07
#define	RTC_ALM1_MINUTES_ADDRESS	0x08
#define	RTC_ALM1_HOURS_ADDRESS		0x09
#define	RTC_ALM1_DAYS_ADDRESS		0x0A
#define	RTC_ALM2_START_ADDRESS		0x0B
#define	RTC_ALM2_MINUTES_ADDRESS	0x0B
#define	RTC_ALM2_HOURS_ADDRESS		0x0C
#define	RTC_ALM2_DAYS_ADDRESS		0x0D
#define	RTC_CONTROL_ADDRESS		0x0E
#define	RTC_STATUS_ADDRESS		0x0F
#define	RTC_MSB_TEMP_ADDRESS		0x11
#define	RTC_LSB_TEMP_ADDRESS		0x12


/*! \def RTC_START_ADDRESS
    \brief RTC Addresses constants. Start address
 */
/*! \def RTC_ADDRESS
    \brief RTC Addresses constants. I2C RTC Address
 */
/*! \def RTC_DATA_SIZE
    \brief RTC Addresses constants. RTC Data size
 */
#define	RTC_START_ADDRESS		0x00
#define RTC_ADDRESS 			0x68
#define RTC_DATA_SIZE 			0x12


/*! \def RTC_DATE_ADDRESS_2
    \brief RTC Addresses constants to use in function 'readRTC'. Use when only time and date wants to be read from the RTC
 */
/*! \def RTC_ALARM1_ADDRESS
    \brief RTC Addresses constants to use in function 'readRTC'. Use when time, date and alarm1 wants to be read from the RTC
 */
/*! \def RTC_ALARM2_ADDRESS
    \brief RTC Addresses constants to use in function 'readRTC'. Use when time, date, alarm1 and alarm2 wants to be read from the RTC
 */
#define	RTC_DATE_ADDRESS_2		6	
#define	RTC_ALARM1_ADDRESS		10	
#define	RTC_ALARM2_ADDRESS		13	

/*! \def RTC_ALM1_MODE1
    \brief RTC Alarm Modes. Day of the week,hours,minutes and seconds match
 */
/*! \def RTC_ALM1_MODE2
    \brief RTC Alarm Modes. Date,hours,minutes and seconds match
 */
/*! \def RTC_ALM1_MODE3
    \brief RTC Alarm Modes. Hours,minutes and seconds match
 */
/*! \def RTC_ALM1_MODE4
    \brief RTC Alarm Modes. Minutes and seconds match
 */
/*! \def RTC_ALM1_MODE5
    \brief RTC Alarm Modes. Seconds match
 */
/*! \def RTC_ALM1_MODE6
    \brief RTC Alarm Modes. Once per second 
 */
/*! \def RTC_ALM2_MODE1
    \brief RTC Alarm Modes. Day of the week,hours and minutes match
 */
/*! \def RTC_ALM2_MODE2
    \brief RTC Alarm Modes. Date,hours and minutes match
 */
/*! \def RTC_ALM2_MODE3
    \brief RTC Alarm Modes. Hours and minutes
 */
/*! \def RTC_ALM2_MODE4
    \brief RTC Alarm Modes. Minutes match
 */
/*! \def RTC_ALM2_MODE5
    \brief RTC Alarm Modes. Once per minute
 */
#define	RTC_ALM1_MODE1			0	
#define	RTC_ALM1_MODE2			1	
#define	RTC_ALM1_MODE3			2	
#define	RTC_ALM1_MODE4			3	
#define	RTC_ALM1_MODE5			4	
#define	RTC_ALM1_MODE6			5	

#define	RTC_ALM2_MODE1			0	
#define	RTC_ALM2_MODE2			1	
#define	RTC_ALM2_MODE3			2	
#define	RTC_ALM2_MODE4			3	
#define	RTC_ALM2_MODE5			4	


/*! \def RTC_OFFSET
    \brief RTC Alarm Values. This option adds the time specified by the user to the actual time
 */
/*! \def RTC_ABSOLUTE
    \brief RTC Alarm Values. This option establishes the time specified by the user as the alarm time
 */
/*! \def RTC_ALM1
    \brief RTC Alarm Values. Specifies Alarm1
 */
/*! \def RTC_ALM2
    \brief RTC Alarm Values. Specifies Alarm2
 */
#define	RTC_OFFSET			0	
#define	RTC_ABSOLUTE			1	
#define	RTC_ALM1			1	
#define	RTC_ALM2			2


/*! \def DAY_1
    \brief Days of the Week. Sunday in this case
 */
/*! \def DAY_2
    \brief Days of the Week. Monday in this case
 */
/*! \def DAY_3
    \brief Days of the Week. Tuesday in this case
 */
/*! \def DAY_4
    \brief Days of the Week. Wednesday in this case
 */
/*! \def DAY_5
    \brief Days of the Week. Thursday in this case
 */
/*! \def DAY_6
    \brief Days of the Week. Friday in this case
 */
/*! \def DAY_7
    \brief Days of the Week. Saturday in this case
 */

/*
#define DAY_1 "Sunday"
#define DAY_2 "Monday"
#define DAY_3 "Tuesday"
#define DAY_4 "Wednesday"
#define DAY_5 "Thursday"
#define DAY_6 "Friday"
#define DAY_7 "Saturday"
*/
#define DAY_1 "Sun"
#define DAY_2 "Mon"
#define DAY_3 "Tue"
#define DAY_4 "Wed"
#define DAY_5 "Thu"
#define DAY_6 "Fri"
#define DAY_7 "Sat"





/*! \def RTC_ON
    \brief RTC Power Modes. ON in this case
 */
/*! \def RTC_OFF
    \brief RTC Power Modes. OFF in this case
 */
#define	RTC_ON	1
#define	RTC_OFF	2

/*! \def RTC_I2C_MODE
    \brief Used to set RTC.isON value
 */
/*! \def RTC_NORMAL_MODE
    \brief Used to set RTC.isON value
 */
#define	RTC_I2C_MODE	1
#define	RTC_NORMAL_MODE	0

//From WASPACC.H

/*! \def cbi
    \brief Function definition to set a register bit to '0'
    
    'sfr' is the register. 'bit' is the register bit to change
 */
 /*
#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif	 */


/*! \def sbi
    \brief Function definition to set a register bit to '1'
    
    'sfr' is the register. 'bit' is the register bit to change
 */
 /*
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif */


#ifndef cbi
//#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))	//看起来_SFR_BYTE( 为AVR特有
#define cbi(sfr, bit) 
#endif

#ifndef sbi
//#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit)) //
#define sbi(sfr, bit) 
#endif





//From WaspSensorgas.H


/*! \def SENS_3V3
    \brief Sensor Board Types. 3V3 switch in this case
    
 */
/*! \def SENS_5V
    \brief Sensor Board Types. 5V switch in this case
    
 */






/******************************************************************************
 * Class
 ******************************************************************************/

//! WaspRTC Class
/*!
	WaspRTC Class defines all the variables and functions used for managing the RTC
 */
class WaspRTC
{
  private:
  public:
	//! class constructor
    	/*!
	  It does nothing
	  \param void
	  \return void
	*/ 
    	WaspRTC();
	
	//! Variable : It stores if the RTC is ON(1) or OFF(0)
    	/*!    
	 */
	//uint8_t isON;

		//uint8_t millisecond;	 这个RTC好像没有ms级别的数据
	
	//! Variable : It stores the value of the seconds
    	/*!    
	 */
    	uint8_t second;
	
	//! Variable : It stores the value of the minutes
    	/*!    
	 */
    	uint8_t minute;	
	//! Variable : It stores the value of the hours
    	/*!    
	 */
    	uint8_t hour;
	
	//! Variable : It stores the value of the day of the week
    	/*!    
	 */
    	uint8_t day;

	
	//! Variable : It stores the value of the date(day of the week)
    	/*!    
	 */
    	uint8_t date;
	
	//! Variable : It stores the value of the month
    	/*!    
	 */
    	uint8_t month;
	
	//! Variable : It stores the value of the year 
    	/*!    
	 */
    	uint8_t year;
	
	//! Variable : It stores the value of the seconds for Alarm1
    	/*!    
	 */
	uint8_t second_alarm1;
	
	//! Variable : It stores the value of the minutes for Alarm1
    	/*!    
	 */
	uint8_t minute_alarm1;
	
	//! Variable : It stores the value of the hours for Alarm1
    	/*!    
	 */
	uint8_t hour_alarm1;
	
	//! Variable : It stores the value of the day of the week/date for Alarm1
    	/*!    
	 */
	uint8_t day_alarm1;

	uint8_t date_alarm1;
	
	//! Variable : It stores the value of the minutes for Alarm2
    	/*!    
	 */
	uint8_t minute_alarm2;
	
	//! Variable : It stores the value of the hours for Alarm2
    	/*!    
	 */
	uint8_t hour_alarm2;
	
	//! Variable : It stores the value of the day of the week/date for Alarm2
    	/*!    
	 */
	uint8_t day_alarm2;

	uint8_t date_alarm2;

	uint8_t control;
	uint8_t status;
	uint8_t agingoffset;
	uint8_t msbtemp;
	uint8_t lsbtemp;

	//! Variable : It stores the timeStamp in "dow, YY/MM/DD - HH:MM.SS"  format
    	/*!    
	 */
	char timeStamp[60];
	

	// RTC Internal Functions
	//! It resets the variables used through the library
    	/*!
	\param void
	\return void
	 */
    	void resetVars();
	
//	//! It gets 'registerRTC' variable
//    	/*!
//	\param void
//	\return 'registerRTC' variable
//	 */
//   	char* getRTCarray();
	
//	//! It gets date and time
//    	/*!
//	\param void
//	\return a string containing date and time. These values are got from the library variables
//	 */
//    	char* getTimestamp();


	
	//! It writes the date and time set in the corresponding variables to the RTC
    	/*!
	\param void
	\return void
	\sa readRTC(uint8_t endAddress)
	 */
    	void writeRTC();
	
	//! It reads from the RTC the date,time and optionally alarm1 and alarm2, setting the corresponding variables
    	/*!
	\param uint8_t endAddress : specifies the last RTC register we want to read
	\return void
	\sa writeRTC()
	 */
	void readRTC();
	

	






	
	//! It detaches the interruption from the defined pin
    	/*!
	\param void
	\return void
	\sa attachInt()
	 */

	
	// RTC User Functions
	
	//! It opens I2C bus and powers the RTC
    	/*!
	\param void
	\return void
	\sa close(), begin()
     	*/ 
	void ON();
	
	//! It closes I2C and powers off the RTC module
    	/*!
	\param void
	\return void
	\sa close(), begin()
     	*/ 
	void OFF();
	
	//! It inits the I2C bus and the variables reading them from the RTC
    	/*!
	\param void
	\return void
	 */
	void begin();
	
	//! It closes I2C bus
    	/*!
	\param void
	\return void
	 */
	void close();
	

	

    	/*!
	\param const char* time : the time and date to set in the RTC. It looks like "YY:MM:DD:dow:hh:mm:ss"
	\return void
	\sa setTime(uint8_t year, uint8_t month, uint8_t date, uint8_t day_week, uint8_t hour, uint8_t minute, uint8_t second), getTime()
	 */
	void setTime(const char* timestr);
	
	//! It sets in the RTC the specified date and time
    	/*!
	\param uint8_t year : the year to set in the RTC
	\param uint8_t month : the month to set in the RTC
	\param uint8_t date : the date to set in the RTC
	\param uint8_t day_week : the day of the week to set in the RTC
	\param uint8_t hour : the hours to set in the RTC
	\param uint8_t minute : the minutes to set in the RTC
	\param uint8_t second : the seconds to set in the RTC
	\return void
	\sa setTime(const char* time), getTime()
	 */
	void setTime(uint8_t year, uint8_t month, uint8_t date, uint8_t day_week, uint8_t hour, uint8_t minute, uint8_t second);
	
	//! It gets from the RTC the date and time, storing them in the corresponding variables
    	/*!
	\param void
	\return a string containing the date and the time
	\sa setTime(const char* time), setTime(uint8_t year, uint8_t month, uint8_t date, uint8_t day_week, uint8_t hour, uint8_t minute, uint8_t second)
	 */
	char* getTime();
	
	//! It sets time and date from the GPS to the RTC. GPS has to be initialized first and got the time/date
    	/*!
	\param void
	\return void
	\sa setTime(const char* time), setTime(uint8_t year, uint8_t month, uint8_t date, uint8_t day_week, uint8_t hour, uint8_t minute, uint8_t second)
	 */
	void setTimeFromGPS();
uint8_t getTemperature();
int setAlarm1andOn(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type);
int clearAlarm1(void);
int offAlarm1(void);
int setAlarm1asAwake(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type);
};

extern WaspRTC RTCbianliang;

#endif

