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

    Design:		David Gasc髇

    Implementation:	David Cuartielles, Alberto Bielsa

 */
 
 
/*! \def XBee_h
    \brief The library flag
    
 */
#ifndef Flash_h
#define Flash_h
#include "dingke_spi.h"
/******************************************************************************
 * Includes
 ******************************************************************************/
 
#include <inttypes.h>

#define FLASHBUSY -1
#define FLASHNOTBUSY -2
#define FLASHCANWRITE -3
#define FLASHWRITEDONE -4
#define FLASHWRITEERR -5
#define FLASHCANTWRITE -6

#define FLASHWRITEENABLEDONE -7
#define FLASHWRITEDISABLEDONE -8

#define FLASHTOOBUSY -9 //尝试了10次还是busy

#define FLASHPAGEDONE -10
#define FLASHPAGEERR -11

#define FLASHSECTOEADDERR -12

#define FLASHSECTOEREADOK 0
#define FLASHSECTOEREADERR -13

#define FLASHSECTOEWRITEERR -14
#define FLASHSECTOEWRITEOK 0

#define FLASHLENERR -15

#define FLASHSTATUSDONE -16

#define FLASHWPROTECTIONON 1
#define FLASHWPROTECTIONOFF 0


#define Dummy 					0xA5	//任意8位数都可以

#define AT_CS_LOW()     	GPIO_ResetBits(GPIOB, GPIO_Pin_12)
#define AT_CS_HIGH()   		GPIO_SetBits(GPIOB, GPIO_Pin_12)

#define FLASH_PROON()     	GPIO_ResetBits(GPIOE, GPIO_Pin_3)
#define FLASH_PROOFF()   		GPIO_SetBits(GPIOE, GPIO_Pin_3)

#define SECTORBYTE 4096



/******************************************************************************
 * Class
 ******************************************************************************/
 
  //! WaspXBee Class
/*!
	WaspXBee Class defines all the variables and functions used to manage the UART related with the XBee
 */
class WaspFLASH
{
  private:
	//uint8_t _uart;

	//void printNumber(unsigned long n, uint8_t base);
	//uint8_t SlaveAdress;

	unsigned char Flagaddbit; //24 //32 //刚上电时不知道这个片子里面是24位地址还是32位地址，先检测
unsigned char TxBuffer[256];
unsigned char  RxBuffer[256];
unsigned long MAXCNT;

  int flashpagebyte4add(long add,unsigned char ch);
int flashpagestr4add(long add,unsigned char *str,unsigned int len);
  void flashsectorerase4add(long add);

  public:
  
	unsigned char Flashbuf[SECTORBYTE];//4KB 一个扇区  	  
	//! class constructor
	WaspFLASH();
	void setADP(void);
	unsigned char checkaddbit(void);
void SPI_Flash_Init(void);
int checkbusy(void);
int writeEnable(void);
int writeDisable(void);
  uint8_t readFlash1byte(uint32_t add);
  uint16_t readFlash2byte(uint32_t add);
  uint32_t readFlash4byte(uint32_t add);
  void sectorerase(uint32_t add);

  int readFlashsector(uint32_t add);
  int flashreaddata4add(long add);
  int writeFlash(uint32_t add,uint8_t ch);
int writeFlash(uint32_t add,uint8_t *str,uint32_t len);
int writeFlash(uint32_t add,uint16_t data);
int writeFlash(uint32_t add,uint32_t data);

uint8_t readStatusRegister1(void);
uint8_t readStatusRegister2(void);
uint8_t readStatusRegister3(void) ;
int writestatusregister1(uint8_t status);
int writestatusregister2(uint8_t status);
int writestatusregister3(uint8_t status);
void reset(void);
 void writeprotection(uint8_t sta);

};

extern WaspFLASH Flash;

#endif

