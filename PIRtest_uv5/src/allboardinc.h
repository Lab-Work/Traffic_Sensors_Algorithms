#ifndef __ALLBOARDINC_H
#define __ALLBOARDINC_H 			   

//////////////////////////////////////////////////////////////////////////////////	 
#include <stm32f4xx.h>
#include<string.h> //strccmp 之类
#include<stdlib.h> //atoi之类
#include "dingke_delay.h"
#include "dingke_spi.h"
#include "dingke_i2c.h"
#include "dingke_timer.h"
#include "dingke_uart.h"
#include "dingke_pwr.h"
#include "dingke_exti.h"
#include "dingke_sysclkchange.h" //变主频用的

//#include "wiring.h"	主要是uart
#include "sdio_sd.h"  //sdio
#include "ff.h"		  //fat32(16) 目前应用于sd卡
  #include "mysdfat.h"	
#include "matrix.h"


#include "WaspEEPROM.h"	//I2C
#include "Wasprtc.h"	  //I2C
#include "WaspFLASH.h"	  //SPI

#include "WaspSD.h"	  //sdio
#include "WaspUtils.h"//led什么的

#include "dingke_wdg.h"


#include "wasppwr.h"
//#include "exti.h"     //PA0作为外部中断 比如当进入睡眠模式时，按PA0键或者任何其他中断都可以唤醒



#include "WaspXBee.h"
#include "WaspXBeeCore.h"
#include "WaspXBee802.h"


void setup(void);
void loop(void);
#endif


