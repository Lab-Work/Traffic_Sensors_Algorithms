/*
 *  Copyright (C) 2009 Libelium Comunicaciones Distribuidas S.L.
 *  http://www.libelium.com
 *
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
 *
 *  Version:		1.1
 *  Design:		David Gascón
 *  Implementation:	Alberto Bielsa, David Cuartielles, Marcos Yarza
 */
 

#ifndef __WPROGRAM_H__
  #include "WaspClasses.h"
#include "dingke_delay.h"
 #include "dingke_i2c.h"
#include <ctype.h>
#endif




union RTCUNION RTCUn;



// Constructors ////////////////////////////////////////////////////////////////


WaspRTC::WaspRTC()
{
  // nothing to do when constructing
}

// Public Methods //////////////////////////////////////////////////////////////

/*
 * ON (void) - It opens I2C bus and powers the RTC
 *
 *  It opens I2C bus and powers the RTC
 */
void WaspRTC::ON(void)
{
	i2cOn();
}


/*
 * OFF (void) - It closes I2C bus and powers off the RTC
 *
 *  It closes I2C bus and powers off the RTC
 */
void WaspRTC::OFF(void)
{
	i2cOff();
}


/* begin() - inits I2C bus and used pins
 *
 * It enables internal pull-up resistor for the RTC interrupt pin, so as this pin is set to HIGH when init
 * It inits I2C bus for communicating with RTC
 * It reads from RTC time,date and alarms, setting the corresponding variables
 *
 * Returns nothing
 */ 
void WaspRTC::begin()
{
 	i2cInit();
}


/* close() - closes I2C bus and used pins
 *
 * It enables internal pull-up resistor for the RTC interrupt pin, so as this pin is set to HIGH when init
 * It inits I2C bus for communicating with RTC
 * It reads from RTC time,date and alarms, setting the corresponding variables
 *
 * Returns nothing
 */
void WaspRTC::close()
{

}





/* resetVars() - resets variables to zero
 *
 * It resets all the used variables to default value
 */
//seconds 0-59
//minutes 0-59
//hour  1-12 am/pm   0-23
//day 1-7
//date 1-31
//month 1-12
//year 0-99 
  
void WaspRTC::resetVars()
{
	second = 0;
	minute = 0;
	hour = 0;
	day = 1;
	date = 1;
	month = 1;
	year = 0;
	second_alarm1 = 0;
	minute_alarm1 = 0;
	hour_alarm1 = 0;
	day_alarm1 = 1;
	date_alarm1 = 1;
	minute_alarm2 = 0;
	hour_alarm2 = 0;
	day_alarm2 = 1;
	date_alarm2 = 1;
}




/* getTimestamp() - returns a string containing variables related with time and date
 *
 * It returns a string containing variables related with time and date. These values are the last taken from RTC
 */
//char* WaspRTC::getTimestamp() 
//{
////  free(timeStamp);
////  timeStamp=NULL;
////  timeStamp=(char*)calloc(31,sizeof(char)); 
////  sprintf (timeStamp, "%s, %02d/%02d/%02d - %02d:%02d:%02d", DAY_s[day-1], year, month, date, hour, minute, second);
//	if(day<=1)day=1;
//	else if(day>=7)	day=7;
//	timeStamp[50]=0x00;timeStamp[59]=0x00;
//	sprintf (timeStamp, "%s, %d/%02d/%02d - %02d:%02d:%02d", DAY_s[day-1], year, month, date, hour, minute, second);
//
//  return timeStamp;
//}





/* readRTC(endAddress) - reads from RTC the specified addresses
  get all data in ds3231
*/
void WaspRTC::readRTC() 
{
//	second = bcdtobyte(readi2cRTC(0x00));
//	minute = bcdtobyte(readi2cRTC(0x01));
//	hour = bcdtobyte(readi2cRTC(0x02));
//	day = bcdtobyte(readi2cRTC(0x03));
//	date = bcdtobyte(readi2cRTC(0x04));
//	month = bcdtobyte(readi2cRTC(0x05));
//	year = bcdtobyte(readi2cRTC(0x06));
//	second_alarm1 = bcdtobyte(readi2cRTC(0x07));
//	minute_alarm1 = bcdtobyte(readi2cRTC(0x08));
//	hour_alarm1 = bcdtobyte(readi2cRTC(0x09));
//	//day_alarm1 = readi2cRTC(0x0a);
//	date_alarm1 = bcdtobyte(readi2cRTC(0x0a));
//	minute_alarm2 = bcdtobyte(readi2cRTC(0x0b));
//	hour_alarm2 = bcdtobyte(readi2cRTC(0x0c));
//	//day_alarm2 = readi2cRTC(0x0d);
//	date_alarm2 = bcdtobyte(readi2cRTC(0x0d));
//
//	control=readi2cRTC(0x0e);
//	status=readi2cRTC(0x0f);
//	agingoffset=readi2cRTC(0x10);
//	msbtemp=readi2cRTC(0x11);
//	lsbtemp=readi2cRTC(0x12);


	second = bcdtobyte(readExternalRTC(0x00));
	minute = bcdtobyte(readExternalRTC(0x01));
	hour = bcdtobyte(readExternalRTC(0x02));
	day = bcdtobyte(readExternalRTC(0x03));
	date = bcdtobyte(readExternalRTC(0x04));
	month = bcdtobyte(readExternalRTC(0x05));
	year = bcdtobyte(readExternalRTC(0x06));
	second_alarm1 = bcdtobyte(readExternalRTC(0x07));
	minute_alarm1 = bcdtobyte(readExternalRTC(0x08));
	hour_alarm1 = bcdtobyte(readExternalRTC(0x09));
	//day_alarm1 = readi2cRTC(0x0a);
	date_alarm1 = bcdtobyte(readExternalRTC(0x0a));
	minute_alarm2 = bcdtobyte(readExternalRTC(0x0b));
	hour_alarm2 = bcdtobyte(readExternalRTC(0x0c));
	//day_alarm2 = readi2cRTC(0x0d);
	date_alarm2 = bcdtobyte(readExternalRTC(0x0d));

	control=readExternalRTC(0x0e);
	status=readExternalRTC(0x0f);
	agingoffset=readExternalRTC(0x10);
	msbtemp=readExternalRTC(0x11);
	lsbtemp=readExternalRTC(0x12);
}


/* writeRTC() - writes the stored variables to the RTC
 *

 */
void WaspRTC::writeRTC() 
{
//	writei2cRTC(0x00,bytetobcd(second) );
//	writei2cRTC(0x01,bytetobcd(minute));
//	writei2cRTC(0x02,bytetobcd(hour));
//	writei2cRTC(0x03,bytetobcd(day));
//	writei2cRTC(0x04,bytetobcd(date));
//	writei2cRTC(0x05,bytetobcd(month));
//	writei2cRTC(0x06,bytetobcd(year));
//	writei2cRTC(0x07,bytetobcd(second_alarm1));
//	writei2cRTC(0x08,bytetobcd(minute_alarm1));
//	writei2cRTC(0x09,bytetobcd(hour_alarm1));
//	writei2cRTC(0x0a,bytetobcd(date_alarm1));
//	writei2cRTC(0x0b,bytetobcd(minute_alarm2));
//	writei2cRTC(0x0c,bytetobcd(hour_alarm2));
//	writei2cRTC(0x0d,bytetobcd(date_alarm2));


	writeExternalRTC(0x00,bytetobcd(second) );
	writeExternalRTC(0x01,bytetobcd(minute));
	writeExternalRTC(0x02,bytetobcd(hour));
	writeExternalRTC(0x03,bytetobcd(day));
	writeExternalRTC(0x04,bytetobcd(date));
	writeExternalRTC(0x05,bytetobcd(month));
	writeExternalRTC(0x06,bytetobcd(year));
	writeExternalRTC(0x07,bytetobcd(second_alarm1));
	writeExternalRTC(0x08,bytetobcd(minute_alarm1));
	writeExternalRTC(0x09,bytetobcd(hour_alarm1));
	writeExternalRTC(0x0a,bytetobcd(date_alarm1));
	writeExternalRTC(0x0b,bytetobcd(minute_alarm2));
	writeExternalRTC(0x0c,bytetobcd(hour_alarm2));
	writeExternalRTC(0x0d,bytetobcd(date_alarm2));

}







/*
void obtain(const char* timestr)
{
	uint8_t i;
	uint8_t str[10],strcnt=0;
	uint8_t len=strlen(timestr);
//	uint8_t len=strlen("Sun,.45678, 12/08/04 - 05:03:07");
	uint8_t flag=0;

	RTCUn.structv.month =len;//str[1];
	RTCUn.structv.mday  =strcnt;//str[2];

	for(i=0;i<len;i++)
	{
		if(flag==0)
		{
			if(isdigit(timestr[i]))
			{
				str[strcnt]=(uint8_t)atoi(&timestr[i]);
				strcnt++;
				flag=1;
			}
		}
		else if(flag==1)
		{
			if(isalpha(timestr[i]))
			{
				flag=0;
			}
		}
	}
	RTCUn.structv.year  =str[0];


	RTCUn.structv.hour  =str[3];
	RTCUn.structv.minute=str[4];
	RTCUn.structv.second=str[5];
}*/

/* setTime(time) - sets time and date in the RTC
 *
 * It sets time and date in the RTC.
 *
 * After setting the variables, function 'writeRTC' is called to write to RTC the values
 *
 * 'time' must be set in a specify format: 
 */
void WaspRTC::setTime(const char* timestr)
{
//   	uint8_t i;
//	//obtain(timestr);
//	
//	RTCUn.structv.year  =(timestr[5] - 48)*10+(timestr[6] - 48);
//	RTCUn.structv.month =(timestr[8] - 48)*10+(timestr[9] - 48);
//	RTCUn.structv.mday  =(timestr[11] - 48)*10+(timestr[12] - 48);
//
//	RTCUn.structv.hour  =(timestr[16] - 48)*10+(timestr[17] - 48);
//	RTCUn.structv.minute=(timestr[19] - 48)*10+(timestr[20] - 48);
//	RTCUn.structv.second=(timestr[22] - 48)*10+(timestr[23] - 48);/**/
//
//	for(i=0;i<7;i++)
//	{
//		if(strncmp(timestr,DAY_s[i],3)==0)
//		{
//			RTCUn.structv.wday   =i+1;
//			break;	
//		}
//	}	
//	
//	//if()	
//
//	for(i=0;i<7;i++)
//	{
//		writei2cRTC(i,bytetobcd(RTCUn.unionstr[i]));
//	}
	setExternalTimeStr(timestr);
}

//day_week;1-7  1 sunday,2 monday,...7 stauday 
//year: example 12    (2012)
//month: 1-12
//date: 1-31
//
/* setTime(_year,_month,_date,day_week,_hour,_minute,_second) - sets time and date in the RTC
 *
 * It sets time and date in the RTC.
 *
 * After setting the variables, function 'writeRTC' is called to write to RTC the values
 *
 * Each input corresponds to the relayed part of time and date.
 */
void WaspRTC::setTime(uint8_t _year, uint8_t _month, uint8_t _date, uint8_t day_week, uint8_t _hour, uint8_t _minute, uint8_t _second)
{
//	uint8_t i;
//	RTCUn.structv.year  =(uint8_t)(_year);
//	RTCUn.structv.month =_month;
//	RTCUn.structv.mday  =_date;
//	RTCUn.structv.wday   =day_week;
//	RTCUn.structv.hour  =_hour;
//	RTCUn.structv.minute=_minute;
//	RTCUn.structv.second=_second;
//
//	for(i=0;i<7;i++)
//	{
//		writei2cRTC(i,bytetobcd(RTCUn.unionstr[i]));
//	}
	setExternalTime(_year, _month, _date, day_week, _hour, _minute, _second);


}


/* getTime() - gets time and date
 *
 * It gets time and date, storing them in 'registersRTC' array.
 *
 * It returns a string containing time and data in the following format: 
 */
char* WaspRTC::getTime()
{
//	uint8_t i;
//	for(i=0;i<7;i++)
//	{
//		RTCUn.unionstr[i]=	bcdtobyte(readi2cRTC(i));		
//	}
//
//	year  =RTCUn.structv.year;
//	month =RTCUn.structv.month;
//	date  =RTCUn.structv.mday;
//	day   =RTCUn.structv.wday;
//	hour  =RTCUn.structv.hour;
//	minute=RTCUn.structv.minute;
//	second=RTCUn.structv.second;
//
//	return getTimestamp();

	getExternalTime();	
	year  =RTCExternalUn.structv.year;
	month =RTCExternalUn.structv.month;
	date  =RTCExternalUn.structv.date;
	day   =RTCExternalUn.structv.day;
	hour  =RTCExternalUn.structv.hour;
	minute=RTCExternalUn.structv.minute;
	second=RTCExternalUn.structv.second;
	sprintf (timeStamp, "%s", RTCExternalStamp);
	return timeStamp;
}

uint8_t WaspRTC::getTemperature()
{
	//return readi2cRTC(0x11);
	return readExternalRTC (0x11);
}

int WaspRTC::setAlarm1andOn(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type)
{
	if(setExternalAlarm1(dateorday, hour, minute, second, type)==I2CFALSE)return I2CFALSE;
	return onExternalAlarm1();
}

//只是把当前的响起来的闹钟关掉，并不是把当前设置的闹钟干掉
int WaspRTC::clearAlarm1(void)
{
	return clearExternalAlarm1();
}

//设置没有闹钟
int WaspRTC::offAlarm1(void)
{
	return offExternalAlarm1();
}



int WaspRTC::setAlarm1asAwake(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type)
{
//	GPIO_InitTypeDef  GPIO_InitStructured;
//  	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);	*/
//  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
//  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_12;
//  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
//  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
//  	GPIO_Init(GPIOE, &GPIO_InitStructured);
//
//	EXTI_InitTypeDef EXTI_InitStruct;
//
//	RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);
//
//	SYSCFG_EXTILineConfig(EXTI_PortSourceGPIOE ,EXTI_PinSource12 );	
//	EXTI_InitStruct.EXTI_Line = EXTI_Line12;
//	EXTI_InitStruct.EXTI_Mode = EXTI_Mode_Interrupt;
//	EXTI_InitStruct.EXTI_Trigger = EXTI_Trigger_Falling;
//	EXTI_InitStruct.EXTI_LineCmd = ENABLE;	
//	EXTI_Init(&EXTI_InitStruct);
//
//	NVIC_InitTypeDef NVIC_InitStruct;
//	NVIC_InitStruct.NVIC_IRQChannel=EXTI15_10_IRQn;
//	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 12;
//	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 12;
//	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
//	NVIC_Init(&NVIC_InitStruct);
//	Timer2_Init(200,1000);
//	if(setExternalAlarm1(dateorday, hour, minute, second, type)==I2CFALSE)return I2CFALSE;
//	return onExternalAlarm1();
	return setExternalAlarm1AsAwake(dateorday, hour, minute, second, type);
}

//int WaspRTC::setAlarm1asAwake(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type)
//{
//	if(setExternalAlarm1(dateorday, hour, minute, second, type)==I2CFALSE)return I2CFALSE;
//	return onExternalAlarm1();
//}









/*******************************************************************************
 * HANDLING HARDWARE INTERRUPTS
 *******************************************************************************/






// Private Methods /////////////////////////////////////////////////////////////

// Preinstantiate Objects //////////////////////////////////////////////////////

WaspRTC RTCbianliang = WaspRTC();
