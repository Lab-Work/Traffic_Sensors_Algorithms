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
 *  Implementation:	Alberto Bielsa, David Cuartielles
 */
 
#ifndef __WPROGRAM_H__
	#include "WaspClasses.h"
#endif

#include <inttypes.h>
extern "C"{
#include "stm32f4xx_iwdg.h"
}
extern "C"{
#include "stm32f4xx.h"
}
/*
extern "C"{
#include "24C02.h"
}  

extern "C"{

#include "usart.h"
}  */
//#include "wasprtc.h"

extern "C"{
#include "dingke_delay.h"
#include "dingke_pwr.h"
}
//#define ADCSRA	0 //看起来是AVR特有的，暂时这样
//#define SLEEP_MODE_PWR_DOWN	0//看起来是AVR特有
//#define	set_sleep_mode(x)	//看起来是AVR特有
//#define		sleep_enable()	//看起来是AVR特有
//#define		sleep_mode()	//看起来是AVR特有
//#define		sleep_disable()	//看起来是AVR特有

//#ifndef cbi
////#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))	//看起来_SFR_BYTE( 为AVR特有
//#define cbi(sfr, bit) 
//#endif
//
//#ifndef sbi
////#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit)) //
//#define sbi(sfr, bit) 
//#endif
//
//// Variables ////////////////////////////////////////////////////////////////
//
//// these two variables have been moved to the general system
////uint8_t intIPRA = 0;
////uint8_t intIPRB = 0;

// Constructors ////////////////////////////////////////////////////////////////

WaspPWR::WaspPWR()
{
  // nothing to do when constructing
}





// Private Methods /////////////////////////////////////////////////////////////

/* setIPF ( peripheral )
 * - sets a certain internal peripheral on 
 * - to control the pwr on the different internal peripherals it is
 *   convenient to read MCU's manual on pgs. 56/57
 * FIXME: missing all the Timers and UART to reduce consumption
 */
//void WaspPWR::setIPF(uint8_t peripheral) 
//{
//	setIPF_(peripheral);
//	intIPRA = IPRA;
//}

/* resetIPR ( peripheral )
 * - resets a certain internal peripheral to off 
 * - to control the pwr on the different internal peripherals it is
 *   convenient to read MCU's manual on pgs. 56/57
 * FIXME: missing all the Timers and UART to reduce consumption
 */
//void WaspPWR::resetIPF(uint8_t peripheral) 
//{
//	resetIPF_(peripheral);
//	intIPRA = IPRA;
//}

/* uint8_t getIPR ( )
 * - answers with the whole IPR 
 */
//uint8_t WaspPWR::getIPF() 
//{
//	return intIPRA;
//}


// Public Methods //////////////////////////////////////////////////////////////


//激活DS2745在指定的地址中将MODE位置0
//SDA 或者SCL置高
void WaspPWR::activeDS2745()
{
  writeDS2745(0x01,0x90);
	SDA_OUT();    //sda线输出
	GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA  	  
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
}

//DS2745进入睡眠模式
//将DS2745的SDA SCL 置低大于2S 同时将MODE置高
void WaspPWR::sleepDS2745()
{
    writeDS2745(0x01,0xB0);
	//SDA_OUT();    //sda线输出
 	GPIO_ResetBits(GPIOB,GPIO_Pin_9); //SDA  //START:when CLK is high,DATA change form high to low 
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL //钳住I2C总线，准备发送或接收数据 
	delay_ms(3000);
}
void 	WaspPWR::initPower(uint32_t choose)
{
	initPwr(choose);
}
void 	WaspPWR::initAllPower(void)
{
	initAllPwr();	
}



/* setSensorPower( type, mode) - set ON/OFF 3V3 or 5V switches
 *
 * It sets ON/OFF 3V3 or 5V switches
 */
//关闭传感器电源3.3V  传感器1 5V 传感器2 5V 传感器3 5V 
void 	WaspPWR::setSensorPower(uint32_t type, uint8_t mode)
{
	setPwr(type&0x0f, mode);
//	switch( type )
//	{
//		case SENS_3V3: 	
//			if(mode==SENS_ON) GPIO_SetBits(GPIOC, GPIO_Pin_5);
//			else if(mode==SENS_OFF) GPIO_ResetBits(GPIOC, GPIO_Pin_5);
//			break;
//
//		case SENS1_5V:	
//			if(mode==SENS_ON) GPIO_SetBits(GPIOE, GPIO_Pin_7);
//			else if(mode==SENS_OFF) GPIO_ResetBits(GPIOE, GPIO_Pin_7);
//			break;
//
//		case SENS2_5V:	
//			if(mode==SENS_ON)  GPIO_SetBits(GPIOA, GPIO_Pin_4);
//			else if(mode==SENS_OFF) GPIO_ResetBits(GPIOA, GPIO_Pin_4);
//			break;
//
//		case SENS3_5V:	
//			if(mode==SENS_ON) GPIO_SetBits(GPIOD, GPIO_Pin_6);
//			else if(mode==SENS_OFF) GPIO_ResetBits(GPIOD, GPIO_Pin_6);
//			break;
//	}
}


/* setWatchdog( mode, timer) - enables or disables watchdog interruption
 *
 * It enables or disables watchdog interruption.
 *
 * 'mode' --> if mode=WTD_ON, it enables watchdog interruption. If mode=WTD_OFF, it disables watchdog interruption.
 * 'timer' --> it specifies the time before the watchdog activates the interruption. Possible values are:
 * 	WTD_16MS	0
 *	WTD_32MS	1
 *	WTD_64MS	2
 *	WTD_128MS	3
 *	WTD_250MS	4
 *	WTD_500MS	5
 *	WTD_1S		6
 *	WTD_2S		7
 *	WTD_4S		8
 *	WTD_8S		9
 * 
 * It returns nothing
 */
void 	WaspPWR::setWatchdog(uint8_t mode, uint8_t timer)
{
	if(mode==WTD_ON) //看门狗使能 设置看门狗时间
	{
		
		IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable);
		
		IWDG_SetPrescaler(timer);
		
		IWDG_Enable();
	}
	
	if(mode==WTD_OFF) //看门狗关
	{
		
		IWDG_WriteAccessCmd(IWDG_WriteAccess_Disable);
		
		IWDG_ReloadCounter();
		//off_watchdog
	}
}


/* switchesOFF() - switches off the Waspmote switches specified
 *
 * It switches off all Waspmote switches
 */
  void	WaspPWR::switchesOFF(uint32_t option) //电源关 
{
	setPwr(option, PWR_OFF);	
//	if( option & SENS_OFF ) //SENS_OFF与 传感器3.3V 5V 有关
//	{
//		GPIO_ResetBits(GPIOC, GPIO_Pin_5);
//		GPIO_ResetBits(GPIOE, GPIO_Pin_7);
//		GPIO_ResetBits(GPIOA, GPIO_Pin_4);
//		GPIO_ResetBits(GPIOD, GPIO_Pin_6);
//	}
//	
//	if( option & MUX_OFF )  //MUX_OFF 与串口的多路复用有关系
//	{
//
//		Mux_poweroff();
//	}
//	
//	if( option & SD_OFF )  // SD_OFF 与 SD电源有关
//	{
//		SD_poweroff();
//	}
//	
//	if( option & XBEE_OFF )  // XBEE_OFF 与 ZIGBEE电源有关
//	{
//		Xbee_poweroff();
//	}
//	
//	if( option & RTC_OFF )  // PTC_PW 与 RTC I2C FLASH 电源有关
//	{
//  		i2cOff();
//	}
//	
//	if( option & BAT_OFF ) //镍钴电池充电开关。当PA8 pd5 全为低为 DISABLE MODE 全为高时为CHARGE MODE
//	{
////		pinMode(MAX1501_ACOK,OUTPUT);
////		digitalWrite(MAX1501_ACOK,LOW);
////		pinMode(TS12A_IN,OUTPUT);
////		digitalWrite(TS12A_IN,LOW);
//
//	}	
}

/* switchesON() - switches on all Waspmote switches
 *
 * It switches on all Waspmote switches
 */
void	WaspPWR::switchesON(uint16_t option)
{
 	setPwr(option, PWR_ON);		
//	if( option & SENS_OFF ) //开3.3V 5V 传感器电源 
//	{
//		GPIO_SetBits(GPIOC, GPIO_Pin_5);
//		GPIO_SetBits(GPIOE, GPIO_Pin_7);
//		GPIO_SetBits(GPIOA, GPIO_Pin_4);
//		GPIO_SetBits(GPIOD, GPIO_Pin_6);
//	}
//	
//	if( option & MUX_OFF ) //串口多路复用开关
//	{
//		Mux_poweron();
//	}
//	
//if( option & SD_OFF )  // SD_OFF 与 SD电源有关
//	{
//		SD_poweron();
//	}
//	
//	if( option & XBEE_OFF )  // XBEE_OFF 与 ZIGBEE电源有关
//	{
//			Xbee_poweron();
//	}
//	
//	if( option & RTC_OFF )  // PTC_PW 与 RTC I2C FLASH 电源有关
//	{
//		i2cOn();
//	}
//	
//	if( option & BAT_OFF ) //镍钴电池充电开关。当PA8 pd5 全为低为 DISABLE MODE 全为高时为CHARGE MODE
//	{
////		pinMode(MAX1501_ACOK,OUTPUT);
////		digitalWrite(MAX1501_ACOK,HIGH);
////		pinMode(TS12A_IN,OUTPUT);
////		digitalWrite(TS12A_IN,HIGH);
//	}	
}

void WaspPWR::initBattery(void) 
{
	i2cInit();
}



/* clearInts() - clears all captured interrupts to allow new interrupts on that modules
 *
 * It clears all captured interrupts to allow new interrupts on that modules
 */
/*!void 	WaspPWR::clearInts()
{
	if( intFlag & ACC_INT )
	{
		ACC.setFF();
	}
	if( intFlag & BAT_INT )
	{
	}
	if( intFlag & RTC_INT ) // hay que mirar si está inicializado el I2C
	{
		RTC3231.clearAlarmFlag();
	}
	if( intFlag & UART1_INT )
	{
		enableInterrupts(UART1_INT);
	}
	if( intFlag & WTD_INT )
	{
	}
	if( intFlag & SENS_INT )
	{
	}
}*/

/* sleep() - sets the microcontroller to the lowest consumption sleep mode
 *
 * It sets the microcontroller to the lowest consumption sleep mode. Before setting this state, some interruption
 * should be enabled to be able to wake up the microcontroller from this state.
 *
 * It switches off all the switches on the Waspmote board.
 *
 * It returns nothing.
 */
//void	WaspPWR::sleep(uint8_t option)
//{
//	switchesOFF(option);
//	set_sleep_mode(SLEEP_MODE_PWR_DOWN);
//	sleep_enable();
//	sleep_mode();
//	sleep_disable();
//	switchesON(option);
//}


/* sleep(timer) - sets the microcontroller to the lowest consumption sleep mode
 *
 * It sets the microcontroller to the lowest consumption sleep mode. It enables watchdog interruption to be able to
 * wake up the microcontroller after 'timer' time.
 *
 * 'timer' --> it specifies the time before the watchdog activates the interruption. Possible values are:
 * 	WTD_16MS	0
 *	WTD_32MS	1
 *	WTD_64MS	2
 *	WTD_128MS	3
 *	WTD_250MS	4
 *	WTD_500MS	5
 *	WTD_1S		6
 *	WTD_2S		7
 *	WTD_4S		8
 *	WTD_8S		9
 * 
 * It switches off all the switches on the Waspmote board.
 *
 * It returns nothing.
 */
//void	WaspPWR::sleep(uint8_t	timer, uint8_t option)
//{
//	switchesOFF(option);
//	set_sleep_mode(SLEEP_MODE_PWR_DOWN);
//	sleep_enable();
//	
//	setWatchdog(WTD_ON,timer);
//	sleep_mode();
//	sleep_disable();
//	switchesON(option);
//	
//}


/* deepSleep(time2wake, offset, mode) - sets the microcontroller to the lowest consumption sleep mode
 *
 * It sets the microcontroller to the lowest consumption sleep mode. It enables RTC interruption to be able to
 * wake up the microcontroller when the RTC alarm is launched.
 *
 * 'time2wake' --> it specifies the time at which the RTC alarm will activate. It must follow the next format:
 *	"DD:HH:MM:SS"
 * 'offset' --> it specifies if 'time2wake' is added to the actual time or if this time is set as the alarm
 * 'mode' --> it specifies the mode for RTC alarm
 *
 * It uses Alarm1 on the RTC due to this Alarm has more precision than Alarm2
 * 
 * It switches off all the switches on the Waspmote board.
 *
 * It returns nothing.
 */
// void	WaspPWR::deepSleep(const char*	time2wake, uint8_t offset, uint8_t mode, uint8_t option)
// {
// 	// Set RTC alarme to wake up from Sleep Power Down Mode
// 	RTC3231.setAlarm1(time2wake,offset,mode);
// 	RTC3231.close();
// 	switchesOFF(option);
// 	set_sleep_mode(SLEEP_MODE_PWR_DOWN);
// 	sleep_enable();
// 	sleep_mode();
// 	sleep_disable();
// 	switchesON(option);
// 	RTC3231.ON();
// 	RTC3231.clearAlarmFlag();
// 	if( option & RTC_OFF ) RTC3231.OFF();
// }










/* getBatteryVolts() - gets Volts of remaining battery
 *
 * This function gets the Volts of remaining battery.
 *
 * Returns the voltage of battery. 
 */
 //获取电池电压 电压以MV为单位
int16_t WaspPWR::getBatteryVolts()
{
  uint8_t voltage[2];
	int16_t temp;
	int16_t volbase=488;
	int16_t volta;
	voltage[0]=readDS2745(0X0D);
	voltage[1]=readDS2745(0X0C);
//   for(uint8_t i=0;i<2;i++)
//	printf("voltage=%d",voltage[i]);
	if(voltage[1]&0x80)
	{
//		voltage[1]=voltage[1]&0x7f;		
	  temp=(uint16_t)(255-voltage[1])*256+(255-voltage[0]);
	  temp=temp/32;
	  temp=((uint32_t)temp*volbase)/100;
//		//temp=temp/1000;
	  volta=-temp;
	  //volta=0;
	}
	else
	{
		temp=(uint16_t)voltage[1]*256+voltage[0];
	  temp=temp/32;
	  temp=((uint32_t)temp*volbase)/100;
		volta=temp;//1000;
	}
	return(volta);	
} 

//获取电池温度 温度以.C为单位
int16_t  WaspPWR::getBatterytemperature(void) 
{
  uint8_t temperature[2];
	int16_t temp;
	int16_t tembase=125;
	int16_t tempera;
	temperature[0]=readDS2745(0X0B);
	temperature[1]=readDS2745(0X0A);
//	for(uint8_t i=0;i<2;i++)
//	printf("temperature=%d",temperature[i]);//sendstrusart3(temperature,2);
	if(temperature[1]&0x80)
	{
//		temperature[1]=temperature[1]&0x7f;		
	 temp=(uint16_t)(255-temperature[1])*256+(255-temperature[0]);
	  temp=temp/32;
//	  temp=((uint32_t)temp*tembase)/1000;
	  tempera=-temp;
	  //tempera=0;
	}
	else
	{
		temp=(uint16_t)temperature[1]*256+temperature[0];
	  temp=temp/32;
	  temp=((uint32_t)temp*tembase)/1000;
		tempera=temp;
	}
	return(tempera);
}

//获取电池电流 电流以MA为单位。
int16_t  WaspPWR::getBatteryCurrent() 
{
  int16_t curbase=1042;
	uint8_t Current[2];
	int16_t temp;
	int16_t currt;
	Current[0]=readDS2745(0X0F);
	Current[1]=readDS2745(0X0E);
//	for(uint8_t i=0;i<2;i++)
//	printf("Cu=%d",Current[i]);
//sendstrusart3(Current,2);
	if(Current[1]&0x80)
	{
		//Current[1]=Current[1]&0x7f;	
    temp=(uint16_t)(255-Current[1])*256+(255-Current[0]);
		temp=(temp)/52;
		currt=temp;
  }
	else
	{
    temp=(uint16_t)Current[1]*256+Current[0];
//	printf("<temp=%x>",temp);
		temp=((uint32_t)temp*curbase)/30000;
		currt=temp;
  }
	return(currt);
}
//获取电池累加电流 电流以MAH为单位。
uint16_t  WaspPWR::addBatteryCurrent() 
{
  uint16_t curbase=2731;
	uint16_t cbase=625;
	uint8_t Current[2];
	uint32_t temp;
	uint16_t currt;
	Current[0]=readDS2745(0X11);
	Current[1]=readDS2745(0X10);
	//sendstrusart3(Current,2);
	temp=(uint16_t)Current[1]*256+Current[0];
	temp=temp*cbase;
	temp=temp/curbase;
	currt=(uint16_t)temp;
	return(currt);
}


/* closeI2C() - closes I2C, setting SDA and SCL to '0'
 *
 * This function closes I2C, setting SDA and SCL to '0'
 *
 * Returns nothing
 */
void	WaspPWR::closeI2C()
{
//	pinMode(I2C_SDA,INPUT);
//	pinMode(I2C_SCL,INPUT);
/*	Wire.close();*/
}



// reboots waspmote
void    WaspPWR::reboot()
{
//    __asm__("jmp 0x1E000");
;
}


// Private Methods /////////////////////////////////////////////////////////////

// Preinstantiate Objects //////////////////////////////////////////////////////

WaspPWR PWR2745 = WaspPWR();
