
 

#ifndef __WPROGRAM_H__
  #include "WaspClasses.h"
#include "dingke_delay.h"
 #include "dingke_i2c.h"
#endif

// Constructors ////////////////////////////////////////////////////////////////

WaspSensor::WaspSensor()
{
 //  SlaveAdress=0x00;
}

// Public Methods //////////////////////////////////////////////////////////////
//
void WaspSensor::init()
{
 	i2cInit();
}

//对地址为8位的器件
int WaspSensor::readSensor (uint8_t slaveadress,uint8_t address)
{
	uint8_t temp=0;

	i2cStart(); 
	if(IICSendByteAck(slaveadress)==I2CFALSE)return I2CFALSE;
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址
	i2cStart(); 

	if(IICSendByteAck(slaveadress+1)==I2CFALSE)return I2CFALSE;//发送字节	
		 										  		    		    	   
    temp=i2cRead_Byte(0);		   
    i2cStop();//产生一个停止条件	   
	return temp; 
}
//对地址是16位的器件
int WaspSensor::readSensor (uint8_t slaveadress,int address)
{
	uint8_t temp=0;

	i2cStart(); 
	if(IICSendByteAck(slaveadress)==I2CFALSE)return I2CFALSE;
	if(IICSendByteAck(address>>8)==I2CFALSE)return I2CFALSE;//发送高地址
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址
	i2cStart(); 

	if(IICSendByteAck(slaveadress+1)==I2CFALSE)return I2CFALSE;//发送字节	
		 										  		    		    	   
    temp=i2cRead_Byte(0);		   
    i2cStop();//产生一个停止条件	   
	return temp; 
}
//对地址为8位的器件
int WaspSensor::writeSensor (uint8_t slaveadress,uint8_t address, uint8_t value)
{

    i2cStart();  
	if(IICSendByteAck(slaveadress)==I2CFALSE)return I2CFALSE;			 	
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址	
	if(IICSendByteAck(value))return I2CFALSE; //发送字节		  		    	   
    i2cStop();//产生一个停止条件 
	delay_ms(10);	 
	return I2CTRUE;	
} 

//对地址是16位的器件
int WaspSensor::writeSensor (uint8_t slaveadress,int address, uint8_t value)
{

    i2cStart();  
	if(IICSendByteAck(slaveadress)==I2CFALSE)return I2CFALSE;		
	if(IICSendByteAck(address>>8)==I2CFALSE)return I2CFALSE;//发送高地址	 	
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址	
	if(IICSendByteAck(value))return I2CFALSE; //发送字节		  		    	   
    i2cStop();//产生一个停止条件 
	delay_ms(10);	 
	return I2CTRUE;	
} 


WaspSensor Sensor = WaspSensor();

