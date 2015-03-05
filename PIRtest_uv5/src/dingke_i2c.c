#include "dingke_i2c.h"
#include "dingke_delay.h"
#include "string.h"
#include "dingke_uart.h"
//#include "sysclkchange.h"
//////////////////////////////////////////////////////////////////////////////////	 
// 外部RTC  EEPROM  电池管理芯片 公用一个i2c总线	iic开头函数
// MLX90614(SMBUS)另一个I2C总线						   smbus开头函数
////////////////////////////////////////////////////////////////////////////////// 	  

//初始化IIC
GPIO_InitTypeDef GPIO_InitStructure;

void i2cOn(void)//PE10
{
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE); 
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);
	
	GPIO_SetBits(GPIOE,GPIO_Pin_10);
}

void i2cOff(void)//PE15
{
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE); 
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);
	
	GPIO_ResetBits(GPIOE,GPIO_Pin_10);
}

void i2cInit(void)
{					     
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
    //I2C_SCL PB.8   I2C_SDA PB.9 
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 |GPIO_Pin_9 ;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	  GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

   // I2C PE5 PE6 PE10 
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);

  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOE, &GPIO_InitStructure);
			
	  GPIO_SetBits(GPIOB,GPIO_Pin_8);//SCL  
	  GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA

}

//产生IIC起始信号
void i2cStart(void)
{
	SDA_OUT();    //sda线输出
	GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA  	  
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
	delay_us(4);
 	GPIO_ResetBits(GPIOB,GPIO_Pin_9); //SDA  //START:when CLK is high,DATA change form high to low 
	delay_us(4);
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL //钳住I2C总线，准备发送或接收数据 
}	  
//产生IIC停止信号
void i2cStop(void)
{
	SDA_OUT();//sda线输出
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 
	GPIO_ResetBits(GPIOB,GPIO_Pin_9); //SDA //STOP:when CLK is high DATA change form low to high
 	delay_us(4);
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
 	delay_us(4);	
	GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA  //发送I2C总线结束信号
	delay_us(4);							   	
}
//等待应答信号到来
//返回值：false，接收应答失败
//        true，接收应答成功
int i2cWait_Ack(void)
{
	uint8_t  ucErrTime=0;

	SDA_OUT(); 
	GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA  
	delay_us(2);
	SDA_IN();     //SDA设置为输入  		   
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
	delay_us(2);	 
	while(READ_SDA)
	{
		ucErrTime++;
		if(ucErrTime>250)
		{
			i2cStop();
			return I2CFALSE;
		}
	}
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL //时钟输出0 	   
	return I2CTRUE;  
} 
//产生ACK应答
void i2cAck(void)
{
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 
	SDA_OUT();
	GPIO_ResetBits(GPIOB,GPIO_Pin_9); //SDA 
	delay_us(2);
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
	delay_us(2);
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 
}
//不产生ACK应答		    
void i2cNAck(void)
{
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 
	SDA_OUT();
	GPIO_SetBits(GPIOB,GPIO_Pin_9);//SDA  
	delay_us(2); 
	GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
	delay_us(2); 
	GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 
}					 				     
//IIC发送一个字节
//返回从机有无应答
//1，有应答
//0，无应答			  
void i2cSend_Byte(uint8_t  txd)
{                        
    uint8_t  t;   
	 SDA_OUT(); 	    
    GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL //拉低时钟开始数据传输
    for(t=0;t<8;t++)
    {   
     if(txd&0x80) 
      GPIO_SetBits(GPIOB,GPIO_Pin_9);		
      else
	  GPIO_ResetBits(GPIOB,GPIO_Pin_9);		
       // IIC_SDA=(txd&0x80)>>7;
        txd<<=1; 	  
		delay_us(2);   //对TEA5767这三个延时都是必须的
		GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
		delay_us(2); 
		GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL 	
		delay_us(2); 
    }	 
} 	  

int IICSendByteAck(uint8_t txd)
{
	i2cSend_Byte(txd);	   //发送写命令
	if(i2cWait_Ack()==I2CTRUE)return I2CTRUE;
	else return I2CFALSE;
}


  
//读1个字节，ack=1时，发送ACK，ack=0，发送nACK   
uint8_t  i2cRead_Byte(unsigned char ack)
{
	unsigned char i,receive=0;
	SDA_IN();//SDA设置为输入
    for(i=0;i<8;i++ )
	{
        GPIO_ResetBits(GPIOB,GPIO_Pin_8); //SCL  
        delay_us(2);
		GPIO_SetBits(GPIOB,GPIO_Pin_8); //SCL 
        receive<<=1;
        if(READ_SDA)receive++;   
		delay_us(1); 
    }					 
    if (!ack)
        i2cNAck();//发送nACK
    else
        i2cAck(); //发送ACK   
    return receive;
}


void writeprotection(int flag)
{
	if(flag==IICPROTECTON)GPIO_SetBits(GPIOE,GPIO_Pin_11);
	else GPIO_ResetBits(GPIOE,GPIO_Pin_11);
}

void softwarereset (void)
{
	uint8_t i;
	for(i=0;i<9;i++)
	{
		i2cStart();	
	}
}












///////////////////////////////////////////////////////////////////////////////////////////////////
// 外部RTC
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

//data write -slave receiver mode
//		  slavesddress(r)               word address      data n            data n+1       ...      data n+x 
//host         host           device    host   device     host   device     host   device        host  device       host 
//start        0xD0           ack       xxxxxxxx ack      xxxxxxxx ack      xxxxxxxx ack  	     xxxxxxxx ack       stop

//data read -slave transmitter mode
//		  slavesddress(w)               data n           data n+1    ...      data n+x 
//host         host           device    device   host    device   host 		  device   host      host 
//start        0xD1           ack       xxxxxxxx ack     xxxxxxxx ack    	  xxxxxxxx Nack  	 stop

//data write/read(write pointer, then read) -slave receiver and transmit
//		  slavesddress(w)               word address                   slave address(r)         data n            data n+1       ...     data n+x 
//host         host           device    host   device     host         host           device    device   host     device   host          device   host 	  host   
//start        0xD0           ack       xxxxxxxx ack      start        0xD1           ack       xxxxxxxx ack      xxxxxxxx ack  	     xxxxxxxx Nack       stop


//范围
//seconds 0-59
//minutes 0-59
//hour  1-12 am/pm   0-23	//这里仅采用24小时制的
//day 1-7
//date 1-31
//month 1-12
//year 0-99
static const uint8_t MaxRtcValue[7]={59,59,23, 7,31,12,99};
static const uint8_t MinRtcValue[7]={ 0, 0, 0, 1, 1, 1, 0};
union RTCUNION RTCExternalUn;
char RTCExternalStamp[60];
 const char DAY_s[7][4]={"Sun","Mon","Tue" ,"Wed","Thu","Fri","Sat"};
int readExternalRTC (unsigned char address)
{	
	uint8_t temp=0;
	
	i2cStart(); //先发送启动
	if(IICSendByteAck(SLAVEADDRTC)==I2CFALSE)return I2CFALSE; //发送设备地址(0) 含等待设备应答
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送RTC里面具体的地址，是问时，还是问分还是问秒这些 //含等待设备应答
	i2cStart(); //重新发送启动
	
	if(IICSendByteAck(SLAVEADDRTC+1)==I2CFALSE)return I2CFALSE;//发送设备地址(1) 含等待设备应答	
		 										  		    		    	   
    temp=i2cRead_Byte(0);//接收设备过来的一个字节，含应带设备	,因为只要接收一个，所以主机对设备应带为1 	   
    i2cStop();//发送停止	  
	return temp;   
}
int writeExternalRTC (unsigned char address, uint8_t value)
{	
    i2cStart();//发送启动 
	if(IICSendByteAck(SLAVEADDRTC)==I2CFALSE)return I2CFALSE;//发送设备地址(0) 含等待设备应答			
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送RTC里面具体的地址，//含等待设备应答	
	if(IICSendByteAck(value))return I2CFALSE; //发送具体要改的值，//含等待设备应答		  		    	   
    i2cStop();//发送停止
	delay_ms(10);	 
	return I2CTRUE;	
} 
uint8_t bcdtobyte(uint8_t bcd)
{
	return ((bcd>>4)*10+ (bcd&0x0f));
}

uint8_t bytetobcd(uint8_t abyte)
{
	return 	((((abyte/10)%10)<<4)|((abyte%10)&0x0f));
}

void getExternalTimestamp() 
{
//  free(timeStamp);
//  timeStamp=NULL;
//  timeStamp=(char*)calloc(31,sizeof(char)); 
//  sprintf (timeStamp, "%s, %02d/%02d/%02d - %02d:%02d:%02d", DAY_s[day-1], year, month, date, hour, minute, second);


//  return RTCExternalStamp;
}

int getExternalTime(void)
{
	uint8_t i;
	int zhi;
	for(i=0;i<7;i++)
	{
		zhi = readExternalRTC(i);
		if(zhi==I2CFALSE) return I2CFALSE;
		RTCExternalUn.unionstr[i]=	bcdtobyte(zhi);		
	}

//	if(RTCExternalUn.structv.wday==0)RTCExternalUn.structv.wday=1;
//	if(RTCExternalUn.structv.wday>7)RTCExternalUn.structv.wday=7;
	//RTCExternalStamp[20]=0x00;RTCExternalStamp[21]=0x00;RTCExternalStamp[22]=0x00;RTCExternalStamp[23]=0x00;
	sprintf (RTCExternalStamp, "%s, %d/%02d/%02d - %02d:%02d:%02d", DAY_s[RTCExternalUn.structv.day-1]
	, RTCExternalUn.structv.year
	, RTCExternalUn.structv.month
	, RTCExternalUn.structv.date
	, RTCExternalUn.structv.hour
	, RTCExternalUn.structv.minute
	, RTCExternalUn.structv.second);

	return I2CTRUE;

}

int setExternalTime(uint8_t _year, uint8_t _month, uint8_t _date, uint8_t day_week, uint8_t _hour, uint8_t _minute, uint8_t _second)
{
	uint8_t i;
	RTCExternalUn.structv.year  =(uint8_t)(_year);
	RTCExternalUn.structv.month =_month;
	RTCExternalUn.structv.date =_date;
	RTCExternalUn.structv.day   =day_week;
	RTCExternalUn.structv.hour  =_hour;
	RTCExternalUn.structv.minute=_minute;
	RTCExternalUn.structv.second=_second;

	for(i=0;i<7;i++)
	{
		if(RTCExternalUn.unionstr[i]>MaxRtcValue[i])RTCExternalUn.unionstr[i]=MaxRtcValue[i];
		else if(RTCExternalUn.unionstr[i]<MinRtcValue[i])RTCExternalUn.unionstr[i]=MinRtcValue[i];
	}
	for(i=0;i<7;i++)
	{
		
		if(writeExternalRTC(i,bytetobcd(RTCExternalUn.unionstr[i]))==I2CFALSE)return I2CFALSE;
	}
	return I2CTRUE;
}
int setExternalTimeStr(const char* timestr)
{
   	uint8_t i;
	
	RTCExternalUn.structv.year  =(timestr[5] - 48)*10+(timestr[6] - 48);
	RTCExternalUn.structv.month =(timestr[8] - 48)*10+(timestr[9] - 48);
	RTCExternalUn.structv.date  =(timestr[11] - 48)*10+(timestr[12] - 48);

	RTCExternalUn.structv.hour  =(timestr[16] - 48)*10+(timestr[17] - 48);
	RTCExternalUn.structv.minute=(timestr[19] - 48)*10+(timestr[20] - 48);
	RTCExternalUn.structv.second=(timestr[22] - 48)*10+(timestr[23] - 48);/**/


	for(i=0;i<7;i++)
	{
		if(strncmp(timestr,DAY_s[i],3)==0)
		{
			RTCExternalUn.structv.day   =i+1;
			break;	
		}
	}
	for(i=0;i<7;i++)
	{
		if(RTCExternalUn.unionstr[i]>MaxRtcValue[i])RTCExternalUn.unionstr[i]=MaxRtcValue[i];
		else if(RTCExternalUn.unionstr[i]<MinRtcValue[i])RTCExternalUn.unionstr[i]=MinRtcValue[i];
	}		
	
	for(i=0;i<7;i++)
	{
		
		if(writeExternalRTC(i,bytetobcd(RTCExternalUn.unionstr[i]))==I2CFALSE)return I2CFALSE;
	}
	return I2CTRUE;
}

//设置闹钟，这里仅仅设置闹钟，开启闹钟要再调用int onExternalAlarm1(void)
//type = 0 	 每秒一个闹钟
//type = 1 	 对应的秒数匹配一个闹钟，即每分钟一个闹钟
//type = 2 	 对应的分钟和秒数匹配一个闹钟，即每小时一个闹钟
//type = 3 	 对应的时分秒匹配一个闹钟，即每天一个闹钟
//type = 4 	 对应的日期、时、分、秒匹配一个闹钟，即差不多每月一个闹钟
//type = 5 	 对应的星期、时、分、秒匹配一个闹钟，即每个星期一个闹钟
// dateorday ,当type = 4和5时，dateorday对应日期和星期
//小时需要是24小时制的
int setExternalAlarm1(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type)
{
	uint8_t temp;
	int flagerr;
	if(type>5)type=5;
	if(second>59)second =59;
	if(minute>59)minute=59;
	if(hour>23)hour=23;
	if((type==4)&&(dateorday>31))dateorday=31;
	else if(type==5)
	{	
		if(dateorday>7)dateorday=7;
		else if(dateorday==0)dateorday=1;

	}

	second =  bytetobcd(second);
	minute =  bytetobcd(minute);
	hour =  bytetobcd(hour);
	dateorday =  bytetobcd(dateorday);
	
	switch(type)
	{
		case 0: second |=0x80;  minute |=0x80;  hour |=0x80;  dateorday |=0x80;  break;
		case 1: second &=~0x80; minute |=0x80;  hour |=0x80;  dateorday |=0x80; break;
		case 2: second &=~0x80; minute &=~0x80; hour |=0x80;  dateorday |=0x80; break;
		case 3: second &=~0x80; minute &=~0x80; hour &=~0x80; dateorday |=0x80; break;
		case 4: second &=~0x80; minute &=~0x80; hour &=~0x80;  dateorday &=~0xc0; break;
		case 5: second &=~0x80; minute &=~0x80; hour &=~0x80; dateorday |=0x40;  dateorday &=~0x80; break;
		default: break;
	}

	if(writeExternalRTC(0x07,second)==I2CFALSE)return I2CFALSE;
	if(writeExternalRTC(0x08,minute)==I2CFALSE)return I2CFALSE;
	if(writeExternalRTC(0x09,hour)==I2CFALSE)return I2CFALSE;
	if(writeExternalRTC(0x0a,dateorday)==I2CFALSE)return I2CFALSE;

	flagerr = readExternalRTC(0x0e);
	if(flagerr==I2CFALSE)return I2CFALSE;
	temp = flagerr; 
	temp = temp|0x05;
	if(writeExternalRTC(0x0e,temp)==I2CFALSE)return I2CFALSE;

	return I2CTRUE;
}


//只是把当前的响起来的闹钟关掉，并不是把当前设置的闹钟干掉
int clearExternalAlarm1(void)
{
	uint8_t temp;
	int flagerr;

	flagerr = readExternalRTC(0x0f);
	if(flagerr==I2CFALSE)return I2CFALSE;
	temp = flagerr; 
	temp = temp & 0xfe;
	if(writeExternalRTC(0x0f,temp)==I2CFALSE)return I2CFALSE;

	return I2CTRUE;
}

//设置没有闹钟
int offExternalAlarm1(void)
{
	uint8_t temp;
	int flagerr;
	flagerr = readExternalRTC(0x0e);
	if(flagerr==I2CFALSE)return I2CFALSE;
	temp = flagerr; 
	temp &= ~0x01;
	if(writeExternalRTC(0x0e,temp)==I2CFALSE)return I2CFALSE;

	return I2CTRUE;
}
//开启闹钟1中断
int onExternalAlarm1(void)
{
	uint8_t temp;
	int flagerr;

	flagerr = readExternalRTC(0x0e);
	if(flagerr==I2CFALSE)return I2CFALSE;
	temp = flagerr; 
	temp = temp|0x05;
	if(writeExternalRTC(0x0e,temp)==I2CFALSE)return I2CFALSE;

	return I2CTRUE;
}

int setExternalAlarm1AsAwake(uint8_t dateorday, uint8_t hour, uint8_t minute, uint8_t second, uint8_t type)
{
//	GPIO_InitTypeDef  GPIO_InitStructured;
//	EXTI_InitTypeDef EXTI_InitStruct;
//	NVIC_InitTypeDef NVIC_InitStruct;
//
//  	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);	*/
//  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
//  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_12;
//  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
//  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
//  	GPIO_Init(GPIOE, &GPIO_InitStructured);
//
//
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
//
//	NVIC_InitStruct.NVIC_IRQChannel=EXTI15_10_IRQn;
//	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 12;
//	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 12;
//	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
//	NVIC_Init(&NVIC_InitStruct);
//	Timer2_Init(200,1000);
//	if(setExternalAlarm1(dateorday, hour, minute, second, type)==I2CFALSE)return I2CFALSE;
	return onExternalAlarm1();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//EEPROM
//////////////////////////////////////////////////////////////////////////////////////////////////////
//我们这里的slaveadress 就是0xa0，所以下面两个函数是不是考虑下直接去掉slaveaddress 这个参数？
int readEeprom (uint8_t slaveadress,int address)
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
int writeEeprom (uint8_t slaveadress,int address, uint8_t value)
{	
    i2cStart();  
	if(IICSendByteAck(slaveadress)==I2CFALSE)return I2CFALSE;		
	if(IICSendByteAck((uint8_t)(address>>8))==I2CFALSE)return I2CFALSE;//发送高地址	 	
	if(IICSendByteAck((uint8_t)(address&0x00ff))==I2CFALSE)return I2CFALSE;//发送低地址	
	if(IICSendByteAck(value))return I2CFALSE; //发送字节		  		    	   
    i2cStop();//产生一个停止条件 
	delay_ms(10);
		 
	return I2CTRUE;	
} 



/////////////////////////////////////////////////////////
//电池芯片读写
// 芯片的名字为 DS2745
//////////////////////////////////////////////////////////


u8 readDS2745(u16 ReadAddr)
{				  
	u8 temp=0;		  	    																 
	i2cStart();  
	
	i2cSend_Byte(0X90);	   //发送写命令
	i2cWait_Ack();
	i2cSend_Byte(ReadAddr);//发送高地址>>8
	i2cWait_Ack();		 
	//IIC_Send_Byte(ReadAddr%256);   //发送低地址
	//IIC_Wait_Ack();	    
	i2cStart();  	 	   
	i2cSend_Byte(0X91);           //进入接收模式			   
	i2cWait_Ack();	 
	temp=i2cRead_Byte(0);		   
	i2cStop();//产生一个停止条件	    
	return temp;
}
//在AT24CXX指定地址写入一个数据
//WriteAddr  :写入数据的目的地址    
//DataToWrite:要写入的数据
void writeDS2745(u16 WriteAddr,u8 DataToWrite)
{				   	  	    																 
	i2cStart();  
	
	i2cSend_Byte(0X90);	    //发送写命令
	i2cWait_Ack();
	i2cSend_Byte(WriteAddr);//发送高地址>>8
	i2cWait_Ack();		 
	
	//IIC_Send_Byte(WriteAddr%256);   //发送低地址
	//IIC_Wait_Ack(); 	 										  		   
	i2cSend_Byte(DataToWrite);     //发送字节							   
	i2cWait_Ack();  		    	   
	i2cStop();//产生一个停止条件 
	delay_ms(10);	 
}


