#ifndef __WPROGRAM_H__
  #include "WaspClasses.h"
#include "dingke_delay.h"
 #include "dingke_i2c.h"
//#include "sysclkchange.h"
#endif
// Constructors ////////////////////////////////////////////////////////////////
WaspEEPROM::WaspEEPROM()
{
   SlaveAdress=0x00;
}
// Public Methods //////////////////////////////////////////////////////////////
//

void WaspEEPROM::ON()
{
	i2cOn();
}
void WaspEEPROM::OFF()
{
	i2cOff();
}
void WaspEEPROM::begin()
{
 	i2cInit();
}
//void WaspEEPROM::close()
//{
//}
void WaspEEPROM::WriteProtection(int flag)
{
	writeprotection(flag);
}

void WaspEEPROM::start()
{
	i2cStart(); 
}
void WaspEEPROM::SoftwareReset (void)
{
	softwarereset ();
}
/*
int WaspEEPROM::IICSendByteAck(uint8_t txd)
{
	IIC_Send_Byte(txd);	   //发送写命令
	if(IIC_Wait_Ack()==I2CTRUE)return I2CTRUE;
	else return I2CFALSE;
}*/
//客户提供的函数式样
int WaspEEPROM::setSlaveAddress(uint8_t slaveadress)
{
	SlaveAdress=slaveadress;
	return IICSendByteAck(SlaveAdress);	
}
//客户提供的函数式样
int WaspEEPROM::writeEEPROM (int address, uint8_t value)
{	
	if(IICSendByteAck((uint8_t)(address>>8))==I2CFALSE)return I2CFALSE;//发送高地址
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址
	if(IICSendByteAck(value))return I2CFALSE; //发送字节		 										  		    		    	   
    i2cStop();//产生一个停止条件 
	delay_ms(10);	 
	return I2CTRUE;	
} 
//客户提供的函数式样
int WaspEEPROM::readEEPROM (int address)
{
	uint8_t temp=0;
	
	if(IICSendByteAck((uint8_t)(address>>8))==I2CFALSE)return I2CFALSE;//发送高地址
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址
	i2cStart(); 

	if(IICSendByteAck(SlaveAdress+1))return I2CFALSE; //发送字节	
		 										  		    		    	   
    temp=i2cRead_Byte(0);		   
    i2cStop();//产生一个停止条件	   
	return temp; 
}
//我们希望的写一个字节函数的式样
//电路图决定了板子上的EEPROM地址为0xa0 ,如果电路图改变了，只需要把下面的0xa0改成对应的地址就可以了
int WaspEEPROM::writeEEPROM (uint8_t slaveadress,int address, uint8_t value)
{
	return writeEeprom (0xa0,address, value);
}


//我们希望的读一个字节函数的式样
//电路图决定了板子上的EEPROM地址为0xa0 ,如果电路图改变了，只需要把下面的0xa0改成对应的地址就可以了
int WaspEEPROM::readEEPROM (uint8_t slaveadress,int address)
{ 
    return readEeprom (0xa0,address);
}

//我们希望的写字符串函数的式样
int WaspEEPROM::writeEEPROMStr(uint8_t slaveadress,uint16_t writeaddr,uint8_t *pbuffer,uint16_t strlen)
{
	while(strlen--)
	{
		if(writeEeprom(slaveadress,writeaddr,*pbuffer)==I2CFALSE)return I2CFALSE;
		writeaddr++;
		pbuffer++;
	}
	return I2CTRUE;
}
//我们希望的读字符串函数的式样
int WaspEEPROM::readEEPROMStr(uint8_t slaveadress,uint16_t readaddr,uint8_t *pbuffer,uint16_t strlen)
{
	int flag;
	while(strlen)
	{
		flag=readEeprom (slaveadress,readaddr++);
		if(flag==I2CFALSE)return I2CFALSE;
		else *pbuffer++=flag;	
		strlen--;
	}
	return I2CTRUE;
} 
 



// Preinstantiate Objects //////////////////////////////////////////////////////

WaspEEPROM Eeprom = WaspEEPROM();

