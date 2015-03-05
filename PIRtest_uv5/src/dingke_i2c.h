#ifndef __DINGKE_I2C_H
#define __DINGKE_I2C_H
#include <stm32f4xx.h>
//////////////////////////////////////////////////////////////////////////////////	 

////////////////////////////////////////////////////////////////////////////////// 	  
#include "dingke_smbus.h"


#define I2CTRUE 0
#define I2CFALSE -1

#define I2COK 	I2CTRUE
#define I2CERROR 	I2CFALSE


   	   		   
//IO方向设置
#define SDA_IN()  {GPIOB->MODER&=0XFFF3FFFF;GPIOB->MODER|=0X00000000;}
#define SDA_OUT() {GPIOB->MODER&=0XFFF3FFFF;GPIOB->MODER|=0X00040000;}

//IO操作函数	 
#define IIC_SCL    PBout(8) //SCL
#define IIC_SDA    PBout(9) //SDA	 
#define als        PDout(7)
#define READ_SDA   GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9) //输入SDA 

//
 #define IICPROTECTON 1
 #define IICPROTECTOFF 0

#ifdef __cplusplus
extern "C"{
#endif


//IIC所有操作函数
			 

void i2cAck(void);					//IIC发送ACK信号
void i2cNAck(void);				//IIC不发送ACK信号

void i2cWrite_One_Byte(u8 daddr,u8 addr,u8 data);
u8 i2cRead_One_Byte(u8 daddr,u8 addr);	
void i2cOn(void);
void i2cOff(void);
void i2cInit(void);                //初始化IIC的IO口	
void i2cStart(void);				//发送IIC开始信号
void i2cStop(void);	  			//发送IIC停止信号
void i2cSend_Byte(uint8_t txd);			//IIC发送一个字节
int IICSendByteAck(uint8_t txd);
uint8_t i2cRead_Byte(unsigned char ack);//IIC读取一个字节
int i2cWait_Ack(void); 				//IIC等待ACK信号
void writeprotection(int flag);
void softwarereset (void);


// 外部RTC
#define SLAVEADDRTC 0xd0
struct RTCSTRUCT
{
    	uint8_t second;
    	uint8_t minute;
    	uint8_t hour;
    	uint8_t day; //周的天
    	uint8_t date; //月的天
		uint8_t month;
    	uint8_t year;
}; 

union RTCUNION
{
	uint8_t unionstr[sizeof(struct RTCSTRUCT)/sizeof(uint8_t)];
	struct RTCSTRUCT structv;
};

extern union RTCUNION RTCExternalUn;
extern char RTCExternalStamp[60];
extern const char DAY_s[7][4];

extern int readExternalRTC (unsigned char address);
extern int writeExternalRTC (unsigned char address, uint8_t value);
extern uint8_t bcdtobyte(uint8_t bcd);

extern uint8_t bytetobcd(uint8_t abyte);
extern int getExternalTime(void);
extern int setExternalTime(uint8_t _year, uint8_t _month, uint8_t _date, uint8_t day_week, uint8_t _hour, uint8_t _minute, uint8_t _second);
extern int setExternalTimeStr(const char* timestr);

int setExternalAlarm1(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type);
int clearExternalAlarm1(void);
int offExternalAlarm1(void);
int onExternalAlarm1(void);
int setExternalAlarm1AsAwake(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type);


//EEPROM
int readEeprom (uint8_t slaveadress,int address);
int writeEeprom (uint8_t slaveadress,int address, uint8_t value);


//电池芯片读写
uint8_t readDS2745(uint16_t ReadAddr);							//指定地址读取一个字节
void writeDS2745(uint16_t WriteAddr,uint8_t DataToWrite);		//指定地址写入一个字节


	
#ifdef __cplusplus
} // extern "C"
#endif 
 
  
#endif
















