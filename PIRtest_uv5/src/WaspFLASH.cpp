/*
 
 */
 

#ifndef __WPROGRAM_H__
  #include "WaspClasses.h"
#include "dingke_delay.h"
#include "dingke_spi.h"
#endif

#define MAXCNT 500

// Constructors ////////////////////////////////////////////////////////////////

WaspFLASH::WaspFLASH()
{
   //SlaveAdress=0x00;
	  Flagaddbit=0;
	  //MaxCnt=500;
}



void WaspFLASH::setADP(void)
{
		RxBuffer[0]=0;
		AT_CS_LOW();
		SPI2_RWByte(0x15);
		RxBuffer[0]=SPI2_RWByte(Dummy);		
		AT_CS_HIGH();

		if(RxBuffer[0]&0x02){;}
		else
		{ 
			AT_CS_LOW();
			SPI2_RWByte(0xb7);		
			AT_CS_HIGH();
		}
}

unsigned char WaspFLASH::checkaddbit(void)
{
	RxBuffer[0]=0;
	AT_CS_LOW();
	SPI2_RWByte(0x15);
	RxBuffer[0]=SPI2_RWByte(Dummy);		
	AT_CS_HIGH();

	if(RxBuffer[0]&0x02){Flagaddbit=32;}
	else
	{
		Flagaddbit=24; 
	}
	return Flagaddbit;	
}

void WaspFLASH::SPI_Flash_Init(void)
{
	unsigned char i;
	SPI2_Init();
	CSPin_init();
	FLASH_SPI_Init();
	for(i=0;i<10;i++)
	{
		setADP();
		if(checkaddbit()==32)break;
	}
	if(i<10);//addbit=32 OK
		
} 
//如果忙延时1ms返回	FLASHBUSY，不忙直接返回FLASHNOTBUSY
int WaspFLASH::checkbusy(void)
{
	RxBuffer[0]=0;
	AT_CS_LOW();
	SPI2_RWByte(0x05);
	RxBuffer[0]=SPI2_RWByte(Dummy);		
	AT_CS_HIGH();
	//displaychar(TxBuffer[0]);
	if(RxBuffer[0]&0x01)//busy
	{   
		delay_ms(1);
		return FLASHBUSY; 
	}
	else //	not busy
	{ 
		//delay_ms(10);
		return FLASHNOTBUSY;
	}
}
//在写使能之前先判定是否忙，若超出 MAXCNT次数忙则返回FLASHTOOBUSY
//不忙则写使能，返回 FLASHWRITEENABLEDONE
int WaspFLASH::writeEnable(void)
{
  unsigned long cntbusy;
	for(cntbusy=0;cntbusy<MAXCNT;cntbusy++)
	{
		if(checkbusy()!=FLASHBUSY)
		{
			AT_CS_LOW();
			SPI2_RWByte(0x06);//写使能
			AT_CS_HIGH();	
			return FLASHWRITEENABLEDONE;													
		}		
	}
	return FLASHTOOBUSY;
}

//在写使能之前先判定是否忙，若超出 MAXCNT次数忙则返回FLASHTOOBUSY
//不忙则写使能，返回 FLASHWRITEDISABLEDONE
int WaspFLASH::writeDisable(void)
{
 unsigned long cntbusy;
	for(cntbusy=0;cntbusy<MAXCNT;cntbusy++)
	{
		if(checkbusy()!=FLASHBUSY)
		{
			AT_CS_LOW();
			SPI2_RWByte(0x04);//写disable
			AT_CS_HIGH();	
			return FLASHWRITEDISABLEDONE;													
		}		
	}
	return FLASHTOOBUSY;
}


//能读到对应地址的数据返回读的数据
//如果忙，次数超过MAXCNT 返回FLASHBUSY，不忙则读出一个字节数据
//正常读
int WaspFLASH::flashreaddata4add(long add)
{
 unsigned long cntbusy;
	for(cntbusy=0;cntbusy<MAXCNT;cntbusy++)
	{
		if(checkbusy()!=FLASHBUSY)//测试的时候遇到过写数据忙，如果刚执行了写命令就开始读这个时候也是忙的，所以读写之前都判断下是否忙
		{
			TxBuffer[0]=0;
			AT_CS_LOW();
			SPI2_RWByte(0x13); //read data
			SPI2_RWByte((u8)(add>>24));
			SPI2_RWByte((u8)(add>>16));
			SPI2_RWByte((u8)(add>>8));
			SPI2_RWByte((u8)add);
			RxBuffer[0]=SPI2_RWByte(Dummy);
			AT_CS_HIGH();
			return RxBuffer[0];			
		}
	}
	return FLASHBUSY;		
}
//以下三个函数按照客户提供式样，但是这样的式样没有返回FLAHS忙不忙的信息，所以忙的时候只有简单的返回0处理
//当读忙时就返回0 ,不忙返回读出的数据，
uint8_t WaspFLASH::readFlash1byte(uint32_t add)
{
	int result;
	result=flashreaddata4add(add);
	if(result<0)result=0;
	return (uint8_t)result;	
}

//当读忙时就返回0 ,不忙返回读出的数据，
uint16_t WaspFLASH::readFlash2byte(uint32_t add)
{
	int result[2];
	result[0]=flashreaddata4add(add);
	result[1]=flashreaddata4add(add+1);
	if(result[0]<0)result[0]=0;
	if(result[1]<0)result[1]=0;
	return (((uint16_t)result[1])<<8)|((uint16_t)result[0]);
}
//当读忙时就返回0 ,不忙返回读出的数据，
uint32_t WaspFLASH::readFlash4byte(uint32_t add)
{
	int result[4];
	result[0]=flashreaddata4add(add);
	result[1]=flashreaddata4add(add+1);
	result[2]=flashreaddata4add(add+2);
	result[3]=flashreaddata4add(add+3);

	if(result[0]<0)result[0]=0;
	if(result[1]<0)result[1]=0;
	if(result[2]<0)result[2]=0;
	if(result[3]<0)result[3]=0;
	return (((uint16_t)result[3])<<24)|(((uint16_t)result[2])<<16)|(((uint16_t)result[1])<<8)|((uint16_t)result[0]);
}
//把一个扇区的数据读给Flashbuf这个数组，忙的次数超过MAXCNT返回FLASHBUSY
int WaspFLASH::readFlashsector(uint32_t add)
{
	uint32_t i;
	//uint8_t  cntrbusy=0,maxrbusy=100;
	int result;
	unsigned long cntbusy;
	cntbusy=0;
	if(add%SECTORBYTE!=0)return FLASHSECTOEADDERR;
	for(i=0;i<SECTORBYTE;i++)
	//for(i=0;i<1500;i++)
	{
		while(1)
		{
			result=flashreaddata4add(add+i);
			if(result<0)//读忙
			{
				cntbusy++;
				if(cntbusy>MAXCNT)
				{
					return FLASHBUSY;
				}
			}
			else break;
		}
		Flashbuf[i]=result;			
	}
	return FLASHSECTOEREADOK;	
}

//page一个字节，当忙的时候返回忙，不忙的话page且返回 FLASHPAGEDONE
//改变page数据之前一定要扇区檫除一下
int WaspFLASH::flashpagebyte4add(long add,unsigned char ch)
{
	if(writeEnable()==FLASHWRITEENABLEDONE)
	{
	  	AT_CS_LOW();
		SPI2_RWByte(0x02); //prage program	
		SPI2_RWByte((unsigned char )(add>>24));		
		SPI2_RWByte((unsigned char )(add>>16));
		SPI2_RWByte((unsigned char )(add>>8));
		SPI2_RWByte((unsigned char )add);
		SPI2_RWByte(ch);
		AT_CS_HIGH();
		return FLASHPAGEDONE;		
	}
	else return FLASHBUSY;
}
//page几个字节，当忙的时候返回忙，不忙的话page且返回 FLASHPAGEDONE
//改变page数据之前一定要扇区檫除一下
int WaspFLASH::flashpagestr4add(long add,unsigned char *str,unsigned int len)
{
	unsigned int ilen;
	if(writeEnable()==FLASHWRITEENABLEDONE)
	{
	  	AT_CS_LOW();
		SPI2_RWByte(0x02); //prage program	
		SPI2_RWByte((unsigned char )(add>>24));					
		SPI2_RWByte((unsigned char )(add>>16));
		SPI2_RWByte((unsigned char )(add>>8));
		SPI2_RWByte((unsigned char )add);
		for(ilen=0;ilen<len;ilen++)
		SPI2_RWByte(str[ilen]);
		AT_CS_HIGH();

		return FLASHPAGEDONE;					
	}
	else return FLASHBUSY;
}
//扇区檫除
//一个扇区为4KB,一页为256字节，则，一个扇区为16页
void WaspFLASH::flashsectorerase4add(long add)
{
	AT_CS_LOW();
	SPI2_RWByte(0x06);//写使能
	AT_CS_HIGH();

  	AT_CS_LOW();
	SPI2_RWByte(0x20); 			
	SPI2_RWByte((unsigned char )(add>>24));					
	SPI2_RWByte((unsigned char )(add>>16));
	SPI2_RWByte((unsigned char )(add>>8));
	SPI2_RWByte((unsigned char )add);
	AT_CS_HIGH();
}

//扇区檫除
void WaspFLASH::sectorerase(uint32_t add)
{
	flashsectorerase4add(add);
}

//在一个地址上指定写一个数据，过程：读这个扇区，檫除扇区，page扇区
//一个扇区为4KB,一页为256字节，则，一个扇区为16页
int WaspFLASH::writeFlash(uint32_t add,uint8_t ch)
{
	uint32_t addhead;
	uint32_t addleave;
	uint32_t ipage;
	//uint8_t  cntwbusy=0,maxwbusy=100;
	

	addleave=add%SECTORBYTE;
	addhead=add-addleave;

//先读当前扇区内容,内容放在Flashbuf里面
	if(readFlashsector(addhead)==FLASHSECTOEREADOK);
	else return FLASHSECTOEREADERR;

//把当前扇区内容对应写的地方改变
	Flashbuf[addleave]=	ch;

//扇区檫除
	sectorerase(addhead);
//把整个扇区写一遍

	for(ipage=0;ipage<16;ipage++)
	{
		//if(flashpagestr4add(addhead+256*ipage,&Flashbuf[256*ipage],256)==FLASHPAGEDONE);
		if(flashpagestr4add(addhead+256*ipage,&Flashbuf[256*ipage],256)==FLASHPAGEDONE);
		//if(flashpagebyte4add(add,Flashbuf[addleave])==FLASHPAGEDONE);
		else return FLASHSECTOEWRITEERR;			
	}
return FLASHSECTOEWRITEOK;
}


//在一个地址上指定写几个数据，过程：读这个扇区，檫除扇区，page扇区
//一个扇区为4KB,一页为256字节，则，一个扇区为16页
int WaspFLASH::writeFlash(uint32_t add,uint8_t *str,uint32_t len)
{
	uint32_t addhead;
	uint32_t addleave;
	uint32_t ipage;
	uint32_t ilen;
	//uint8_t  cntwbusy=0,maxwbusy=100;
	

	addleave=add%SECTORBYTE;
	addhead=add-addleave;

	if((addleave+len)>SECTORBYTE)return FLASHLENERR;

//先读当前扇区内容,内容放在Flashbuf里面
	if(readFlashsector(addhead)==FLASHSECTOEREADOK);
	else return FLASHSECTOEREADERR;

//把当前扇区内容对应写的地方改变
	for(ilen=0;ilen<len;ilen++)
	Flashbuf[addleave+ilen]=str[ilen];

//扇区檫除
	sectorerase(addhead);
//把整个扇区写一遍

	for(ipage=0;ipage<16;ipage++)
	{
		//if(flashpagestr4add(addhead+256*ipage,&Flashbuf[256*ipage],256)==FLASHPAGEDONE);
		if(flashpagestr4add(addhead+256*ipage,&Flashbuf[256*ipage],256)==FLASHPAGEDONE);
		//if(flashpagebyte4add(add,Flashbuf[addleave])==FLASHPAGEDONE);
		else return FLASHSECTOEWRITEERR;			
	}
return FLASHSECTOEWRITEOK;
}


//在一个地址上指定写数据，过程：读这个扇区，檫除扇区，page扇区
//一个扇区为4KB,一页为256字节，则，一个扇区为16页

int WaspFLASH::writeFlash(uint32_t add,uint16_t data)
{
	uint8_t  str[2]={(uint8_t)data,(uint8_t)(data>>8)};
	return writeFlash(add,str,2);
}

int WaspFLASH::writeFlash(uint32_t add,uint32_t data)
{
	uint8_t  str[4]={(uint8_t)data,(uint8_t)(data>>8),(uint8_t)(data>>16),(uint8_t)(data>>24)};
	return writeFlash(add,str,4);
}


//以下3个为读状态寄存器，读这个可以在任何时候
//状态寄存器1
uint8_t WaspFLASH::readStatusRegister1(void)
{
	AT_CS_LOW();
	SPI2_RWByte(0x05); 
	RxBuffer[0]=SPI2_RWByte(Dummy);
	AT_CS_HIGH();
	return RxBuffer[0];			
}

//状态寄存器2
uint8_t WaspFLASH::readStatusRegister2(void)
{
	AT_CS_LOW();
	SPI2_RWByte(0x35); 
	RxBuffer[0]=SPI2_RWByte(Dummy);
	AT_CS_HIGH();
	return RxBuffer[0];			
}

//状态寄存器3
uint8_t WaspFLASH::readStatusRegister3(void)
{
	AT_CS_LOW();
	SPI2_RWByte(0x15); 
	RxBuffer[0]=SPI2_RWByte(Dummy);
	AT_CS_HIGH();
	return RxBuffer[0];			
}


int WaspFLASH::writestatusregister1(uint8_t status)
{
	if(writeEnable()==FLASHWRITEENABLEDONE)
	{
	  	AT_CS_LOW();
		SPI2_RWByte(0x01); 	
		SPI2_RWByte(status);
		AT_CS_HIGH();
		return FLASHSTATUSDONE;		
	}
	else return FLASHBUSY;
}

int WaspFLASH::writestatusregister2(uint8_t status)
{
	if(writeEnable()==FLASHWRITEENABLEDONE)
	{
	  	AT_CS_LOW();
		SPI2_RWByte(0x31); 	
		SPI2_RWByte(status);
		AT_CS_HIGH();
		return FLASHSTATUSDONE;		
	}
	else return FLASHBUSY;
}

int WaspFLASH::writestatusregister3(uint8_t status)
{
	if(writeEnable()==FLASHWRITEENABLEDONE)
	{
	  	AT_CS_LOW();
		SPI2_RWByte(0x11); 	
		SPI2_RWByte(status);
		AT_CS_HIGH();
		return FLASHSTATUSDONE;		
	}
	else return FLASHBUSY;
}

void WaspFLASH::reset(void)
{
	  	AT_CS_LOW();
		SPI2_RWByte(0x66); 	
		AT_CS_HIGH();
	  	AT_CS_LOW();
		SPI2_RWByte(0x99); 	
		AT_CS_HIGH();

}

void WaspFLASH::writeprotection(uint8_t sta)
{
	if(sta==FLASHWPROTECTIONON)
	{
		FLASH_PROON(); 	
	}
	else
	{
		FLASH_PROOFF(); 
	}

}





// Preinstantiate Objects //////////////////////////////////////////////////////

WaspFLASH Flash = WaspFLASH();

