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
 *  Version:		0.13
 *  Design:		David Gascón
 *  Implementation:	Alberto Bielsa
 */
 

#ifndef __WPROGRAM_H__
#include "WaspClasses.h"
#endif
//打印debug数据
//1:打印sendhandleresponse一些数据
//2:
//3:
#define ZPPDEBUGXBEEPRINT 0



//extern unsigned long TimeusLinshi ;

//extern unsigned char FlagTimeLinshi;

extern uint32_t Flag1ms;

//unsigned char FlagTimeLinshi = 0;
//uint32_t TimemsArray[10];

//uint32_t TMsArray2[30];


extern volatile unsigned long timer0_overflow_count;

#define BYTEZPP 0

    uint8_t MemoryArray[MAX_PARSE];
    uint8_t ByteINArray[120];// = (uint8_t*) calloc(120,sizeof(uint8_t));
	uint8_t TXArray[120];

char RuleStr[200];//检验收到的一帧frame数据比如具备什么条件，每个条件用<>分开，比如第1个位置的数据必须是3则写成<X:01:03>
uint8_t LenRuleStr;
char RecAPIFrameStr[300];//收到的一帧数据存放的地方，在处理收到的一帧数据时候在那个子函数里就有了这一帧数据的大小了，所以这里暂时没有定义一帧大小这个全局变量


uint8_t SendAPIFrameStr[300];//准备发送的一帧数据，
uint16_t LenSendAPIFrameStr;

static char StrRx80[300];//和上面的RecAPIFrameStr不同，RecAPIFrameStr处理所有接受到的xbee对单片机的回复，StrRx80只处理xbee从别的xbee接受到的数据
static uint16_t LenRx80=0;

uint8_t  NDMacLength=0;
uint64_t NDMacAddress[10];
uint16_t NDMyAddress[10];

extern	int rx_buffer_head3;
extern	int rx_buffer_tail3;

//是十六进制数据则返回1 不是返回0// 0-9  a-f  A-F
static uint8_t isxdigitzpp(char zhi)
{
	if((zhi>='0')&&(zhi<='9'));
	else if((zhi>='a')&&(zhi<='f'));
	else if((zhi>='A')&&(zhi<='F'));
	else return 0;

	return 1;
}


//功能：
//		对当前的字符串分析出十六进制数据(目前不带0x前缀的，支持数据前面是空格的)
//		比如：(" 106",3,&v) 则返回2, v值为0x10即 16 		
//value:是真正返回的数据
//return:
//		当str从开头没有正确的十六进制数据时，返回0
//		当str有十六进制的数据，只不过开头是空格，那没有关系。
//		return的值是到最终不再是十六进制的偏移量
//		当返回0代表没有合适的十六进制	
//len:    
//		对这个str顶多分析len个字节
static uint8_t atohexpart(char * str, uint8_t len, unsigned long * value)
{
	uint8_t i;
	uint8_t offset=0;
	uint8_t k;
	uint8_t zhi;
	*value =0;
	for(i=0;i<len;i++)
	{
		if(offset==0)
		{
			if((*(str+i))==0x20)continue;
			if(isxdigitzpp(*(str+i)))offset++;
			else return 0;
		}
		else
		{
			if(isxdigitzpp(*(str+i)))offset++;
			else 
			{
				//结算一下现在获得的十六进制数据
				break;
			}
		}
	}
	if(offset==0)return 0;
	*value =0;
	for(k=0;k<offset;k++)
	{	
		*value = (*value)<<4;
		zhi=str[i-offset+k];
		//printf("k=%2x zhi=%2x ",k,zhi);
		if((zhi>='0')&&(zhi<='9'))zhi=zhi-'0';
		else if((zhi>='a')&&(zhi<='f'))zhi=zhi-'a'+10;
		else if((zhi>='A')&&(zhi<='F'))zhi=zhi-'A'+10;
		else return 0;
		*value = (*value)|zhi;
		//printf("v=%4x ",*(value));
				
	}
	return offset;	
}


void get_check(char * str, unsigned char len)
{
	unsigned char sum=0;
	unsigned char i;
	char * pstr = str+3;
	for(i=3;i<(len-1);i++)
	{
		sum += *(pstr);
		pstr++;
	}
	sum = 0xff - sum;
	*pstr = sum;
}




/*
Function: Initializes all the global variables that will be used later
Returns: Nothing
Parameters:
  protocol_used: Protocol the XBee is using
  frequency: Frequency the XBee is running on
  model_used: Model of XBee used
*/
//void	WaspXBeeCore::init(uint8_t protocol_used, uint8_t frequency, uint8_t model_used)
//{
//    protocol=protocol_used;
//    freq=frequency;
//    model=model_used;
//
//    totalFragmentsReceived=0;
//    pendingPackets=0;
//    pos=0;
//    discoveryOptions=0x00;
//    if(protocol==XBEE_802_15_4)
//    {
//        awakeTime[0]=AWAKE_TIME_802_15_4_H;
//        awakeTime[1]=AWAKE_TIME_802_15_4_L;
//        sleepTime[0]=SLEEP_TIME_802_15_4_H;
//        sleepTime[1]=SLEEP_TIME_802_15_4_L;
//        scanTime[0]=SCAN_TIME_802_15_4;
//        scanChannels[0]=SCAN_CHANNELS_802_15_4_H;
//        scanChannels[1]=SCAN_CHANNELS_802_15_4_L;
//        encryptMode=ENCRYPT_MODE_802_15_4;
//        powerLevel=POWER_LEVEL_802_15_4;
//        timeRSSI=TIME_RSSI_802_15_4;
//        sleepOptions=SLEEP_OPTIONS_802_15_4;
//    }
//    if(protocol==ZIGBEE)
//    {
//        awakeTime[0]=AWAKE_TIME_ZIGBEE_H;
//        awakeTime[1]=AWAKE_TIME_ZIGBEE_L;
//        sleepTime[0]=SLEEP_TIME_ZIGBEE_H;
//        sleepTime[1]=SLEEP_TIME_ZIGBEE_L;
//        scanTime[0]=SCAN_TIME_ZIGBEE;
//        scanChannels[0]=SCAN_CHANNELS_ZIGBEE_H;
//        scanChannels[1]=SCAN_CHANNELS_ZIGBEE_L;
//        timeEnergyChannel=TIME_ENERGY_CHANNEL_ZIGBEE;
//        encryptMode=ENCRYPT_MODE_ZIGBEE;
//        powerLevel=POWER_LEVEL_ZIGBEE;
//        timeRSSI=TIME_RSSI_ZIGBEE;
//        sleepOptions=SLEEP_OPTIONS_ZIGBEE;
//    }
//    if(protocol==DIGIMESH)
//    {
//        awakeTime[0]=AWAKE_TIME_DIGIMESH_H;
//        awakeTime[1]=AWAKE_TIME_DIGIMESH_M;
//        awakeTime[2]=AWAKE_TIME_DIGIMESH_L;
//        sleepTime[0]=SLEEP_TIME_DIGIMESH_H;
//        sleepTime[1]=SLEEP_TIME_DIGIMESH_M;
//        sleepTime[2]=SLEEP_TIME_DIGIMESH_L;
//        scanTime[0]=SCAN_TIME_DIGIMESH_H;
//        scanTime[1]=SCAN_TIME_DIGIMESH_L;
//        encryptMode=ENCRYPT_MODE_DIGIMESH;
//        powerLevel=POWER_LEVEL_DIGIMESH;
//        timeRSSI=TIME_RSSI_DIGIMESH;
//        sleepOptions=SLEEP_OPTIONS_DIGIMESH;
//    }
//	
//    data_length=0;
//    it=0;
//    start=0;
//    finish=0;
//    add_type=0;
//    mode=0;
//    frag_length=0;
//    TIME1=0;
//    nextIndex1=0;
//    frameNext=0;
//    replacementPolicy=XBEE_OUT;
//    indexNotModified=1;
//    error_AT=2;
//    error_RX=2;
//    error_TX=2;
//    clearFinishArray();
//    clearCommand();
//    apsEncryption=0;
//}


/*
 Function: Get the 32 lower bits of my MAC address
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: When it is executed stores the returned value by SL command in the global 
         "sourceMacLow[4]" variable
*/
//unsigned char comparein()


//static unsigned char RecXbeeBuffer[300];
//static unsigned char RecXbeeBufferLOK[300];
//static uint8_t RecLenWant=290;
//static int  RecLenFact;

//功能:	准备要发送的一帧数据，//这个frame是已经准备好的，比如问SL，问MY，问CH这些的frame是可以事先准备好的
//		比如#define	get_own_mac_low		"7E00040852534C06" , 
//			代入的参数为 get_own_mac_low ,生成SendAPIFrameStr[]={0x7e,0x00,0x04,0x04,0x08,0x52,0x53,0x4c,0x06};
//return: 0 OK   1 error 
//		代入的str符合标准，比如上面例子样子，那能够生成sendframe	，不符合，比如"0x7E0004"前面有了莫需要的0x了，这样生成的sendframe就不对了
uint8_t prepareaexistframe(const char* str)
{
	unsigned long value=0;
	uint8_t len;
	uint8_t i;
	len=strlen(str);
	LenSendAPIFrameStr=0;
	for(i=0;i<len;i=i+2)
	{
		if(atohexpart((char *)str+i, 2, &value)>0)
		{
			SendAPIFrameStr[LenSendAPIFrameStr]=(uint8_t)value;
			LenSendAPIFrameStr++;
		}
	}
	if(LenSendAPIFrameStr==(len/2))return 0;
	else return 1;
		
}


uint8_t WaspXBeeCore::getOwnMacLow()
{
    int8_t error=2;
     
    error_AT=2;
    gen_data(get_own_mac_low);
    error=gen_send(get_own_mac_low);
    
    if(error==0)
    {
        for(it=0;it<4;it++)
        {
            sourceMacLow[it]=data[it];
        }
    }
    return error; 

//	prepareaexistframe(get_own_mac_low);
//	return sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);



}

/*
 Function: Get the 32 higher bits of my MAC address
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: When it is executed stores the returned value by SH in the global 
         "sourceMacHigh[4]" variable
*/
uint8_t WaspXBeeCore::getOwnMacHigh()
{
    int8_t error=2;
     
    error_AT=2;
    gen_data(get_own_mac_high);
    error=gen_send(get_own_mac_high);
    
    if(error==0)
    {
        for(it=0;it<4;it++)
        {
            sourceMacHigh[it]=data[it];
        }
    }
    return error;

//	prepareaexistframe(get_own_mac_high);
//	return sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);
}

/*
 Function: Get the 64 bits of my MAC address
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes functions getOwnMacLow() and getOwnMacHigh()
*/
 uint8_t WaspXBeeCore::getOwnMac()
{
    int8_t error=2;
    error=getOwnMacLow();
    if(error==0)
    {
        error=getOwnMacHigh();  
    }
    return error;
}

/*
 Function: Set the 16b network address
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command in this protocol
 Parameters: 
   NA_H : Higher byte of Network Address (0x00-0xFF)
   NA_L : Lower byte of Network Address (0x00-0xFF)
   Values: Stores in global "sourceNA[2]" variable the 16b address set by the user
 */
uint8_t WaspXBeeCore::setOwnNetAddress(uint8_t NA_H, uint8_t NA_L)
{
    int8_t error=2;

    if(protocol==XBEE_802_15_4)
    {
        error_AT=2;
        gen_data(set_own_net_address,NA_H,NA_L);
        gen_checksum(set_own_net_address);
        error=gen_send(set_own_net_address);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
    
    if(!error)
    {
        sourceNA[0]=NA_H;
        sourceNA[1]=NA_L;
    }
    return error;

//	uint8_t flagerr;
//	uint8_t i;
//
//	SendAPIFrameStr[0] = 0x7e;
//	SendAPIFrameStr[1] = 0x00;
//	SendAPIFrameStr[2] = 0x06;
//	SendAPIFrameStr[3] = 0x08;
//	SendAPIFrameStr[4] = 0x52;
//	SendAPIFrameStr[5] = 'M';
//	SendAPIFrameStr[6] = 'Y';
//	SendAPIFrameStr[7] = NA_H;
//	SendAPIFrameStr[8] = NA_L;
//
//	LenSendAPIFrameStr =  SendAPIFrameStr[2]+4;
//
//	get_check((char *)SendAPIFrameStr, LenSendAPIFrameStr);
////	printf("len=%d ",LenSendAPIFrameStr);
////	for(i=0;i<LenSendAPIFrameStr;i++)
////		printf(" %2x,",SendAPIFrameStr[i]);
////	printf("send set net");
//
//	flagerr=sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);
//	if(flagerr==0)
//	{
//        sourceNA[0]=NA_H;
//        sourceNA[1]=NA_L;	
//	}
//	return flagerr;
}

/*
 Function: Get the 16b network address
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command in this protocol  
 Values: Stores in global "sourceNA[2]" variable the returned 16b network address
 by MY command
*/
uint8_t WaspXBeeCore::getOwnNetAddress()
{
    int8_t error=2;
     

    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) ) 
    {
        error_AT=2;
		//serialWrite('e', 2);serialWrite('2', 2);serialWrite(' ', 2);
        gen_data(get_own_net_address);
		//serialWrite('e', 2);serialWrite('3', 2);serialWrite(' ', 2);

        error=gen_send(get_own_net_address);

		//serialWrite('e', 2);serialWrite('4', 2);

//		if((error_AT>=0)&&(error_AT<=9)) serialWrite(error_AT+'0', 2);
//		else  serialWrite('e', 2);
//		 serialWrite(' ', 2);  serialWrite(' ', 2);
    } 
    else
    {
        error_AT=-1;
        error=-1;
    }
  
    if(!error)
    {
        sourceNA[0]=data[0];
        sourceNA[1]=data[1];
    }
    return error;


//	prepareaexistframe(get_own_net_address);
//	return sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);


}

/*
 Function: Set Baudrate to use
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command in this protocol
 Parameters: 
   baud_rate: integer that contains the baudrate
   Values: Stores in global "baudrate" variable the baudrate
 */
//uint8_t WaspXBeeCore::setBaudrate(uint8_t baud_rate)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(set_baudrate,baud_rate);
//    gen_checksum(set_baudrate);
//    error=gen_send(set_baudrate);
//    
//    if(!error)
//    {
//        baudrate=baud_rate;
//    }
//    return error;
//}

/*
 Function: Set API values
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command in this protocol
 Parameters: 
   api_value: integer that contains the api value
   Values: Stores in global "apiValue" variable the baudrate
 */
//uint8_t WaspXBeeCore::setAPI(uint8_t api_value)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(set_api_mode,api_value);
//    gen_checksum(set_api_mode);
//    error=gen_send(set_api_mode);
//    
//    if(!error)
//    {
//        apiValue=api_value;
//    }
//    return error;
//}

/*
 Function: Set API options. Enable ZIgBee Application Layer Addressing
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command in this protocol
 Parameters: 
   api_options: integer that contains the baudrate
 */
//uint8_t WaspXBeeCore::setAPIoptions(uint8_t api_options)
//{
//    int8_t error=2;
//        
//    if( (protocol!=XBEE_802_15_4) )
//    {
//        error_AT=2;
//        gen_data(set_api_options,api_options);
//        gen_checksum(set_api_options);
//        error=gen_send(set_api_options);
//    }
//    else
//    {
//        error_AT=-1;
//        error=-1;
//    }
//    return error;
//}

/*
 Function: Set the network identifier
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Parameters: 
   PANID: Array of integers than contains the 16b or 64b PAN ID
 Values: Stores in global "PAN_ID" variable the recent set PAN ID value
 */
uint8_t WaspXBeeCore::setPAN(uint8_t* PANID)
{
    int8_t error=2;
        
    if( (protocol==XBEE_802_15_4) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) ) 
    {
        error_AT=2;
        gen_data(set_pan,PANID);
        gen_checksum(set_pan);
        error=gen_send(set_pan);
    }
    
    if(protocol==ZIGBEE) 
    {	
        error_AT=2;
        gen_data(set_pan_zb,PANID);
        gen_checksum(set_pan_zb);
        error=gen_send(set_pan_zb);
    }

    if(!error)
    {
        if( (protocol==XBEE_802_15_4) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) ) 
        {
            for(it=0;it<2;it++)
            {
                PAN_ID[it]=PANID[it];
            }
        }
        if(protocol==ZIGBEE) 
        {
            for(it=0;it<8;it++)
            {
                PAN_ID[it]=PANID[it];
            }
        }
    } 
    return error;
}

/*
 Function: Get Network ID
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "error" variable any error happened while execution
	 Stores in global "PAN_ID" variable the 16b or 64b network PAN ID
*/
//uint8_t WaspXBeeCore::getPAN()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_pan);
//    if( protocol==ZIGBEE ) error=gen_send(get_pan);
//    else error=gen_send(get_pan);
//    
//    if(!error)
//    {
//        if( (protocol==XBEE_802_15_4) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) ) 
//        {
//            for(it=0;it<2;it++)
//            {
//                PAN_ID[it]=data[it];
//                delay(20);
//            }
//        }
//        if(protocol==ZIGBEE) 
//        {
//            for(it=0;it<8;it++)
//            {
//                PAN_ID[it]=data[it];
//                delay(20);
//            }
//        }
//    } 
//    return error;
//}

/*
 Function: Set the module to the sleep mode specified.
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores the returned value by SM command in the global "sleepMode" variable
 Parameters:
   sleep: Defines the sleep mode to use by the XBee (0-5)
*/
uint8_t WaspXBeeCore::setSleepMode(uint8_t sleep)
{
    int8_t error=2;
     
    error_AT=2;
    gen_data(set_sleep_mode_xbee,sleep);
    gen_checksum(set_sleep_mode_xbee);
    error=gen_send(set_sleep_mode_xbee);
    
    if(!error)
    {
        sleepMode=sleep;
    }
    return error;
}

/*
 Function: Get the XBee mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores the XBee mode in the global "sleepMode" variable
*/
//uint8_t WaspXBeeCore::getSleepMode()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_sleep_mode_xbee);
//    error=gen_send(get_sleep_mode_xbee);
//    
//    if(error==0)
//    {
//        sleepMode=data[0];
//    } 
//    return error;
//}

/*
 Function: Set the time the module has to be idle before start sleeping
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the ST parameter in XBee module
	 Stores in global "awakeTime" the value of this time
 Parameters: 
   awake: Array of integers that specifies the time to be awake before sleep
 */
//uint8_t WaspXBeeCore::setAwakeTime(uint8_t* awake)
//{
//    int8_t error=2;
//        
//    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_868) )
//    {
//        error_AT=2;
//        gen_data(set_awake_time,awake);
//        gen_checksum(set_awake_time);
//        error=gen_send(set_awake_time);
//    }
//    
//    if( (protocol==DIGIMESH) || (protocol==XBEE_900) )
//    {
//        error_AT=2;
//        gen_data(set_awake_time_DM,awake);
//        gen_checksum(set_awake_time_DM);
//        error=gen_send(set_awake_time_DM);
//    }
//    
//    if(!error)
//    {
//        if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_868) )
//        {
//            awakeTime[0]=awake[0];
//            awakeTime[1]=awake[1];
//        }
//        if( (protocol==DIGIMESH) || (protocol==XBEE_900) )
//        {
//            awakeTime[0]=awake[0];
//            awakeTime[1]=awake[1];
//            awakeTime[2]=awake[2];
//        }
//    }
//    return error;
//}

/*
 Function: Set the cyclic sleeping time of the node
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the SP parameter in the XBee module
	 Stores in global "sleepTime" the value of this time
 Parameters:
   sleep: Array of Integers that specifies the amount of time the module spends sleeping
*/
//uint8_t WaspXBeeCore::setSleepTime(uint8_t* sleep)
//{
//    int8_t error=2;
//    
//    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_868) )
//    {
//        error_AT=2;
//        gen_data(set_sleep_time,sleep);
//        gen_checksum(set_sleep_time);
//        error=gen_send(set_sleep_time);
//    }
//    
//    if( (protocol==DIGIMESH) || (protocol==XBEE_900) )
//    {
//        error_AT=2;
//        gen_data(set_sleep_time_DM,sleep);
//        gen_checksum(set_sleep_time_DM);
//        error=gen_send(set_sleep_time_DM);
//    }
//    
//    if(!error)
//    {
//        if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_868) )
//        {
//            sleepTime[0]=sleep[0];
//            sleepTime[1]=sleep[1];
//        }
//        if( (protocol==DIGIMESH) || (protocol==XBEE_900) )
//        {
//            sleepTime[0]=sleep[0];
//            sleepTime[1]=sleep[1];
//            sleepTime[2]=sleep[2];
//        }
//    }
//    return error;
//}

/*
 Function: Set the channel frequency where the module is going to work 
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Stores the selected channel in the global "channel" variable
 Parameters:
   _channel: Channel used to transmit (0x0B-0x1A)
*/
uint8_t WaspXBeeCore::setChannel(uint8_t _channel)
{
    int8_t error=2;
     
    if( (protocol==XBEE_802_15_4) || (protocol==DIGIMESH) || (protocol==XBEE_900) )
    {
        error_AT=2;
        gen_data(set_channel,_channel);
        gen_checksum(set_channel);
        error=gen_send(set_channel);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
    if(!error)
    {
        channel=_channel;
    }

    return error;
}

/*
 Function: Get the actual frequency channel 
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores the frequency channel in the global "channel" variable
*/
uint8_t WaspXBeeCore::getChannel()
{
    int8_t error=2;
     
    error_AT=2;
    gen_data(get_channel);
    error=gen_send(get_channel);
    
    if(!error)
    {
        channel=data[0];
    }
    return error;


//	prepareaexistframe(get_channel);
//	return sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);

}

/*
 Function: Set the Node Indentifier
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the NI to the selected in the function
         The NI must be a 20 character max string
         Stores the given NI in the global "nodeID" variable
 Parameters: 
   node: string that specifies the node indentifier
*/
//uint8_t WaspXBeeCore::setNodeIdentifier(const char* node)
//{
//    uint8_t* NI = (uint8_t*) calloc(30,sizeof(uint8_t)); //{0x7E, 0x00, 0x00, 0x08, 0x52, 0x4E, 0x49, 0x02};
//    if( NI==NULL ) return 2;
//    NI[0]=0x7E;
//    NI[1]=0x00;
//    NI[3]=0x08;
//    NI[4]=0x52;
//    NI[5]=0x4E;
//    NI[6]=0x49;
//    int8_t error=2;
//    uint8_t* ByteIN = (uint8_t*) calloc(20,sizeof(uint8_t));
//    if( ByteIN==NULL ) return 2;
//    
//    uint8_t counter=0;
//    uint8_t checksum=0; 
//
//
//    it=0;
//    error_AT=2;
//    while( (node[it]!='\0') )
//    {
//        NI[it+7]=uint8_t(node[it]);
//        it++;
//    }
//    NI[2]=4+it;
//    for(it=3;it<(7+(NI[2]-4));it++)
//    {
//        checksum=checksum+NI[it];
//    }
//    while( (checksum>255))
//    {
//        checksum=checksum-256;
//    }
//    checksum=255-checksum;
//    NI[7+NI[2]-4]=checksum;
//    while(counter<(8+NI[2]-4))
//    {
//        XBee.print(NI[counter], BYTE); 
//        counter++;
//    }
//    counter=0;
//    clearCommand();
//    command[5]=0x4E;
//    command[6]=0x49;
//    error=parse_message(command);
//
//    if(error==0)
//    {
//        for(it=0;it<NI[2]-4;it++)
//        {
//            nodeID[it]=node[it];
//        }
//    }
//    free(NI);
//    free(ByteIN);
//    NI=NULL;
//    ByteIN=NULL;
//    return error;
//}

/*
 Function: Get the Node Identifier
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores the NI in the global "nodeID" variable
*/
//uint8_t WaspXBeeCore::getNodeIdentifier()
//{
//    int8_t error=2; 
//
//    error_AT=2;
//    gen_data(get_NI);
//    error=gen_send(get_NI);
//    
//    if(!error)
//    {
//        for(it=0;it<data_length;it++)
//        {
//            nodeID[it]=char(data[it]);
//        }
//    }
//    return error;
//}

/*
 Function: Scans for brothers in the same channel and same PAN ID
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores given info (SH,SL,MY,RSSI,NI) in global array "scannedBrothers" variable
         Stores in global "totalScannedBrothers" the number of found brothers
*/
uint8_t WaspXBeeCore::scanNetwork()
{
    uint8_t error=2;
	
    error_AT=2;
    totalScannedBrothers=0;
    gen_data(scan_network);
    error=gen_send(scan_network);

    return error;

//	prepareaexistframe(scan_network);
//	return sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);

}

/*
 Function: Scans for brothers in the same channel and same PAN ID
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores given info (SH,SL,MY,RSSI,NI) in global array "scannedBrothers" variable
         Stores in global "totalScannedBrothers" the number of founded brothers
 Parameters:
    node: 20-byte max string containing NI of the node to search
*/
//uint8_t WaspXBeeCore::scanNetwork(const char* node)
//{
//    uint8_t* ND = (uint8_t*) calloc(30,sizeof(uint8_t)); //{0x7E, 0x00, 0x04, 0x08, 0x52, 0x4E, 0x44, 0x13};
//    if( ND==NULL ) return 2;
//    ND[0]=0x7E;
//    ND[1]=0x00;
//    ND[3]=0x08;
//    ND[4]=0x52;
//    ND[5]=0x4E;
//    ND[6]=0x44;
//    int8_t error=2;
//    uint8_t* ByteIN = (uint8_t*) calloc(20,sizeof(uint8_t));
//    if( ByteIN==NULL ) return 2;
//    
//    uint8_t counter=0;
//    uint16_t checksum=0;
//    uint16_t interval=WAIT_TIME2;
//    
//    error_AT=2;
//    totalScannedBrothers=0;
//    if( (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) )
//    {
//        interval=14000;
//    }
//    it=0;
//    while( (node[it]!='\0') )
//    {
//        ND[it+7]=uint8_t(node[it]);
//        it++;
//    }
//    ND[2]=4+it;
//    for(it=3;it<(7+(ND[2]-4));it++)
//    {
//        checksum=checksum+ND[it];
//    }
//    while( (checksum>255))
//    {
//        checksum=checksum-256;
//    }
//    checksum=255-checksum;
//    ND[7+ND[2]-4]=checksum;
//    while(counter<(8+ND[2]-4))
//    {
//        XBee.print(ND[counter], BYTE); 
//        counter++;
//    }
//    counter=0;
//    clearCommand();
//    command[5]=ND[5];
//    command[6]=ND[6];
//    error=parse_message(command);
//    
//    free(ND);
//    free(ByteIN);
//    ND=NULL;
//    ByteIN=NULL;
//    return error;
//}


/*
 Function: Defines the amount of time the scanNetwork() function is scanning
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Changes the NT command
	 Stores in global "scanTime" variable the recent set time
 Parameters:
   time: amount of time ND is scanning for brothers (0x01-0xFC)
*/
uint8_t WaspXBeeCore::setScanningTime(uint8_t* time)
{
    int8_t error=2;
     
    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_900) )
    {
        error_AT=2;
        gen_data(set_scanning_time,time);
        gen_checksum(set_scanning_time);
        error=gen_send(set_scanning_time);
    }
    
    if( (protocol==DIGIMESH) || (protocol==XBEE_868) )
    {
        error_AT=2;
        gen_data(set_scanning_time_DM,time);
        gen_checksum(set_scanning_time_DM);
        error=gen_send(set_scanning_time_DM);
    }
        
    
    if(!error)
    {
        if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) || (protocol==XBEE_900) )
        {
            scanTime[0]=time[0];
        }
        if( (protocol==DIGIMESH) || (protocol==XBEE_868) )
        {
            scanTime[0]=time[0];
            scanTime[1]=time[1];
        }
    }
    return error;
}

/*
 Function: Get the Scanning Time
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "error" variable any error happened while execution
	 Stores in global "scanTime" the value of scanning time
*/
//uint8_t WaspXBeeCore::getScanningTime()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_scanning_time);
//    if( (protocol==DIGIMESH) || (protocol==XBEE_868) || (protocol==ZIGBEE) || (protocol==XBEE_900) ) error=gen_send(get_scanning_time);
//    else error=gen_send(get_scanning_time);
//    
//    if(!error)
//    {
//        if( (protocol==XBEE_802_15_4) )
//        {
//            scanTime[0]=data[0];
//        }
//        if( (protocol==ZIGBEE) || (protocol==XBEE_900) )
//        {
//            scanTime[0]=data[1]; 
//        }
//        if( (protocol==DIGIMESH) || (protocol==XBEE_868) )
//        {
//            scanTime[0]=data[0];
//            scanTime[1]=data[1];
//        }
//    }
//    return error;
//}

/*
 Function: Set the options value for the network discovery command
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the NO command
 Parameters:
   options: chosen option (0x00-0x03)
*/
//uint8_t WaspXBeeCore::setDiscoveryOptions(uint8_t options)
//{
//    int8_t error=2;
//    
//    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
//    {
//        error_AT=2;
//        gen_data(set_discov_options,options);
//        gen_checksum(set_discov_options);
//        error=gen_send(set_discov_options);
//    }
//    else
//    {
//        error_AT=-1;
//        error=-1;
//    }
//    if(!error)
//    {
//        discoveryOptions=options;
//    }
//    return error;
//}

/*
 Function: Get the options value for the network discovery command
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes the NO command. Stores in global "discoveryOptions" variable the options
*/
//uint8_t WaspXBeeCore::getDiscoveryOptions()
//{
//    int8_t error=2;
//    
//    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
//    {
//        error_AT=2;
//        gen_data(get_discov_options);
//        error=gen_send(get_discov_options);
//    }
//    else
//    {
//        error_AT=-1;
//        error=-1;
//    }
//    if(error==0)
//    {
//        discoveryOptions=data[0];
//    }
//    return error;
//}

/*
 Function: Performs a quick search. 
	   802.15.4 : It keeps in DL the MY of the looked up NI brother
	   ZIGBEE : Stores in global "paquete" naD,macDH,macDL from the searched device
	   DIGIMESH: Stores in global "paquete" macDH,macDL from the searched device
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes DN command. 
 Parameters: 
   node: string that specifies the NI that identifies the searched brother
   length: length of that NI (0-20)
*/
//uint8_t WaspXBeeCore::nodeSearch(const char* node, struct packetXBee* paq)
//{
//    uint8_t* DN = (uint8_t*) calloc(30,sizeof(uint8_t)); //{0x7E, 0x00, 0x00, 0x08, 0x52, 0x44, 0x4E, 0xE3};
//    if( DN==NULL ) return 2;
//    DN[0]=0x7E;
//    DN[1]=0x00;
//    DN[3]=0x08;
//    DN[4]=0x52;
//    DN[5]=0x44;
//    DN[6]=0x4E;
//    uint8_t num_DN;
//    int8_t error=2;
//    uint8_t* ByteIN = (uint8_t*) calloc(25,sizeof(uint8_t));
//    if( ByteIN==NULL ) return 2;
//    
//    uint8_t counter=0;
//    uint8_t checksum=0; 
//    uint16_t interval=2000;
//
//
//    error_AT=2;
//    if(protocol==DIGIMESH)
//    {
//        interval=14000;
//    }    
//    it=0;
//    while( (node[it]!='\0') )
//    {
//        DN[it+7]=uint8_t(node[it]);
//        it++;
//    }
//    DN[2]=4+it;
//    for(it=3;it<(7+(DN[2]-4));it++)
//    {
//        checksum=checksum+DN[it];
//    }
//    while( (checksum>255))
//    {
//        checksum=checksum-256;
//    }
//    checksum=255-checksum;
//    DN[7+DN[2]-4]=checksum;
//    while(counter<(8+DN[2]-4))
//    {
//        XBee.print(DN[counter], BYTE); 
//        counter++;
//    }
//    
//    counter=0;
//    clearCommand();
//    command[5]=0x44;
//    command[6]=0x4E;
//    error=parse_message(command);
//    
//    if(error==0)
//    {
//        if( (protocol==ZIGBEE) || (protocol==XBEE_900) || (protocol==XBEE_868) )
//        {
//            for(it=0;it<2;it++)
//            {
//                paq->naD[it]=data[it];
//            }
//            for(it=0;it<4;it++)
//            {
//                paq->macDH[it]=data[it+2];
//            }
//            for(it=0;it<4;it++)
//            {
//                paq->macDL[it]=data[it+6];
//            }
//        }
//        if(protocol==DIGIMESH)
//        {
//            for(it=0;it<4;it++)
//            {
//                paq->macDH[it]=data[it];
//            }
//            for(it=0;it<4;it++)
//            {
//                paq->macDL[it]=data[it+4];
//            }
//        }
//    }
//    free(DN);
//    free(ByteIN);
//    DN=NULL;
//    ByteIN=NULL;
//    return error;
//}

/*
 Function: Write the current parameters to a non volatil memory
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes the WR command
*/
uint8_t WaspXBeeCore::writeValues()
{
    int8_t error=2;
    
    error_AT=2;
    gen_data(write_values);
    error=gen_send(write_values);
    
    return error;
}

/*
 Function: Specifies the list of channels to scan when performing an energy scan 
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Change the SC command. Stores in global "scanChannels" variable the list of channels
 Parameters: 
   channel_H: higher byte of list of channels (0x00-0xFF)
   channel_L: lower byte of list of channels (0x00-0xFF
*/
uint8_t WaspXBeeCore::setScanningChannels(uint8_t channel_H, uint8_t channel_L)
{
    int8_t error=2;
    
    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
    {
        error_AT=2;
        gen_data(set_scanning_channel,channel_H,channel_L);
        gen_checksum(set_scanning_channel);
        error=gen_send(set_scanning_channel);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
    if(error==0)
    {
        scanChannels[0]=channel_H;
        scanChannels[1]=channel_L;
    }
    return error;
}

/*
 Function: Get the list of channels to scan when performing an energy scan
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Stores in global "scanChannels" variable the scanning channel list
*/
uint8_t WaspXBeeCore::getScanningChannels()
{
    int8_t error=2;
    
    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
    {
        error_AT=2;
        gen_data(get_scanning_channel);
        error=gen_send(get_scanning_channel);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
    if(error==0)
    {
        for(it=0;it<2;it++)
        {
            scanChannels[it]=data[it];
        }
    }
    return error;
}

/*
 Function: It sets the time the energy scan will be performed
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Change the ED command. Stores in global "energyChannel" variable the energy in each channel
 Parameters:
   duration: amount of time that the energy scan will be performed (0-6)
*/
uint8_t WaspXBeeCore::setDurationEnergyChannels(uint8_t duration)
{
    int8_t error=2;
    
    if( (protocol==XBEE_802_15_4) )
    {
        error_AT=2;
        gen_data(set_duration_energy,duration);
        gen_checksum(set_duration_energy);
        error=gen_send(set_duration_energy);
    }
    else if( (protocol==ZIGBEE) )
    {
        error_AT=2;
        gen_data(set_duration_energy_ZB,duration);
        gen_checksum(set_duration_energy_ZB);
        error=gen_send(set_duration_energy_ZB);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
	
    if(error==0)
    {
        if(protocol==XBEE_802_15_4)
        {
            for(it=0;it<data_length;it++)
            {
                energyChannel[it]=data[it];
            }
        }
        if(protocol==ZIGBEE)
        {
            timeEnergyChannel=data[0];
        }
    }
    return error;
}

/*
 Function: It gets the time the energy scan will be performed
 Returns: Integer that determines if there has been any error 
   error=2  --> The command has not been executed
   error=1  --> There has been an error while executing the command
   error=0  --> The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Change the SD command. Stores in global "timeEnergyChannel" variable the time the energy 
	 scan will be performed
*/
uint8_t WaspXBeeCore::getDurationEnergyChannels()
{
    int8_t error=2;
    
    if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
    {
        error_AT=2;
        gen_data(get_duration_energy);
        error=gen_send(get_duration_energy);
    }
    else
    {
        error_AT=-1;
        error=-1;
    }
    if(!error)
    {
        timeEnergyChannel=data[0];
    }
    return error;
}

/*
 Function: Sets the encryption key to be used in the 128b AES algorithm
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the KY command. Stores in global "linkKey" variable the key has been set
 Parameters:
   key: 16 byte array of chars that specifies the 128b AES key
*/
uint8_t WaspXBeeCore::setLinkKey(const char* key)
{
    int8_t error=2;
    
    error_AT=2;
    gen_data(set_link_key,key);
    gen_checksum(set_link_key);
    error=gen_send(set_link_key);
    
    if(!error)
    {
        for(it=0;it<16;it++)
        {
            linkKey[it]=char(key[it]);
        }
    }
    return error;
}

//uint8_t WaspXBeeCore::getLinkKey(void)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_link_key);
//    error=gen_send(get_link_key);
//    
//    if(!error)
//    {
//        for(it=0;it<16;it++)
//        {
//            linkKey[it]=char(data[it]);
//        }
//    }
//    return error;
//}
//


/*
 Function: Sets the encryption mode on/off
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the EE command. Stores in global "encryptMode" variable the encryption mode
 Parameters:
   mode: on/off the encryption mode (1/0)
*/
uint8_t WaspXBeeCore::encryptionMode(uint8_t mode)
{
    int8_t error=2;
    
    error_AT=2;
    gen_data(set_encryption,mode);
    gen_checksum(set_encryption);
    error=gen_send(set_encryption);
    if(!error)
    {
        encryptMode=mode;
    }
    return error;
}

//uint8_t WaspXBeeCore::getencryptionMode(void)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_encryption);
//    error=gen_send(get_encryption);
//    
//    if(!error)
//    {
//        encryptMode=data[0];
//    }
//    return error;
//}



/*
 Function: Select the power level at which the RF module transmits
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the PL command. Stores in global "powerLevel" the power level at which RF tx
 Parameters:
   value: power level of transmission (0-4)
*/
//uint8_t WaspXBeeCore::setPowerLevel(uint8_t value)
//{
//    int8_t error=2;
//    
//    if(protocol!=XBEE_900)
//    {
//        error_AT=2;
//        gen_data(set_power_level,value);
//        gen_checksum(set_power_level);
//        error=gen_send(set_power_level);
//    }
//    if(!error)
//    {
//        powerLevel=value;
//    }
//    return error;
//}

/*
 Function: Get the Received Signal Strength Indicator of the last received packet
 Returns: Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "valueRSSI" variable the RSSI value of last received packet
*/
//uint8_t WaspXBeeCore::getRSSI()
//{
//    int8_t error=2;
//    uint8_t* ByteIN = (uint8_t*) calloc(40,sizeof(uint8_t));
//    if( ByteIN==NULL ) return 2;
//    uint8_t i=0;
//
//    if( (protocol == XBEE_802_15_4 ) || (protocol==ZIGBEE) )
//    {
//        error_AT=2;
//        gen_data(get_RSSI);
//        error=gen_send(get_RSSI);
//    }
//    else if( (protocol== DIGIMESH) || (protocol==XBEE_868) || (protocol==XBEE_900) )
//    {
//        delay(2000);        
//        XBee.print("+++");
//        delay(2000);
//        XBee.flush();
//        XBee.println("atdb");
//        delay(1000);
//        error_AT=2;
//        while(XBee.available()>0)
//        {
//            ByteIN[i]=XBee.read();
//            error=0;
//            i++;
//            error_AT=0;
//        }
//        i=0;
//        XBee.println("atcn");
//        delay(1000);
//        valueRSSI[0]=Utils.str2hex(ByteIN);
//    }
//    if(error==0)
//    {
//        if( (protocol==XBEE_802_15_4) || (protocol==ZIGBEE) )
//        {
//            valueRSSI[0]=data[0];
//        }
//    }
//    free(ByteIN);
//    ByteIN=NULL;
//    return error;
//}

/*
 Function: Get the Harware Version
 Returns: Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "hardVersion" variable the Hardware Version
*/
//uint8_t WaspXBeeCore::getHardVersion()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(get_hard_version);
//    error=gen_send(get_hard_version);
//    if(!error)
//    {
//        hardVersion[0]=data[0];
//        hardVersion[1]=data[1];
//    } 
//    return error;
//}

/*
 Function: Get the version of the firmware
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "softVersion" variable the firmware version
*/
//uint8_t WaspXBeeCore::getSoftVersion()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(get_soft_version);
//    error=gen_send(get_soft_version);
//    if(error==0)
//    {
//        softVersion[0]=data[0];
//        softVersion[1]=data[1];
//    } 
//    return error;
//}


/*
 Function: Set the RSSI time
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the RP command. Stores in global "timeRSSI" variable the RSSI time
 Parameters:
   time: amount of time to do the pwm (0x00-0xFF)
*/
//uint8_t WaspXBeeCore::setRSSItime(uint8_t time)
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(set_RSSI_time,time);
//    gen_checksum(set_RSSI_time);
//    error=gen_send(set_RSSI_time);
//    if(!error)
//    {
//        timeRSSI=time;
//    }
//    return error;
//}

/*
 Function: Get the RSSI time
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "timeRSSI" variable the RSSI time
*/
//uint8_t WaspXBeeCore::getRSSItime()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(get_RSSI_time);
//    error=gen_send(get_RSSI_time);
//    if(!error)
//    {
//        timeRSSI=data[0];
//    } 
//    return error;
//}

/*
 Function:  Immediately applies new settings without exiting command mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes the AC command
*/
//uint8_t WaspXBeeCore::applyChanges()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(apply_changes);
//    error=gen_send(apply_changes);
//    return error;
//}


/*
 Function: Reset the XBee Firmware
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes the FR command  
*/
//uint8_t WaspXBeeCore::reset()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(reset_xbee);
//    error=gen_send(reset_xbee);
//    return error;
//}

/*
 Function: Set the parameteres to the factory defaults
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Executes the RE command  
*/
//uint8_t WaspXBeeCore::resetDefaults()
//{
//    int8_t error=2;
//    
//    error_AT=2;
//    gen_data(reset_defaults_xbee);
//    error=gen_send(reset_defaults_xbee);
//    return error;
//}

/*
 Function: Configure options for sleep
 Returns: Integer that determines if there has been any error 
   error=2 -->  The command has not been executed
   error=1 -->  There has been an error while executing the command
   error=0 -->  The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Change the SO command. Stores in global "sleepOptions" variable the options
 Parameters:
   soption: options for sleep (0x00-0xFF)
*/
//uint8_t WaspXBeeCore::setSleepOptions(uint8_t soption)
//{
//    int8_t error=2;
//        
//    if( (protocol==ZIGBEE) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) )
//    {
//        error_AT=2;
//        gen_data(set_sleep_options_xbee,soption);
//        gen_checksum(set_sleep_options_xbee);
//        error=gen_send(set_sleep_options_xbee);
//    }
//    else
//    {
//        error_AT=-1;
//        error=-1;
//    }
//    if(!error)
//    {
//        sleepOptions=soption;
//    }
//    return error;
//}

/*
 Function: Reads the options for sleep
 Returns: Integer that determines if there has been any error 
   error=2 -->  The command has not been executed
   error=1 -->  There has been an error while executing the command
   error=0 -->  The command has been executed with no errors
   error=-1 --> Forbidden command for this protocol
 Values: Executes the SO command. Stores in global "sleepOptions" variable the options
*/
//uint8_t WaspXBeeCore::getSleepOptions()
//{
//    int8_t error=2;
//        
//    if( (protocol==ZIGBEE) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) )
//    {
//        error_AT=2;
//        gen_data(get_sleep_options_xbee);
//        error=gen_send(get_sleep_options_xbee);
//    }
//    else
//    {
//        error_AT=-1;
//        error=-1;
//    }
//    if(!error)
//    {
//        sleepOptions=data[0]; 
//    }
//    return error;
//}

/*
 Function: Transparent function. The user introduces an AT command within a string and the function executes it without knowing its meaning
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Parameters:
  atcommand : String to specify the AT command to execute
*/
//uint8_t WaspXBeeCore::sendCommandAT(const char* atcommand)
//{
//    uint8_t* AT = (uint8_t*) calloc(30,sizeof(uint8_t));// {0x7E, 0x00, 0x00, 0x08, 0x52, 0x00, 0x00, 0x00};
//    if( AT==NULL ) return 2;
//    AT[0]=0x7E;
//    AT[1]=0x00;
//    AT[3]=0x08;
//    AT[4]=0x52;
//    int8_t error=2;
//    uint8_t it2=0;
//    
//    uint8_t* ByteIN = (uint8_t*) calloc(120,sizeof(uint8_t));
//    if( ByteIN==NULL ) return 2;
//    uint8_t counter=0;
//    uint8_t checksum=0; 
//    uint16_t length=0;
//    
//    it=0;
//    error_AT=2;
//    while( atcommand[it2]!='#' )
//    {
//        if( it>=2 )
//        {
//            if( atcommand[it2+1]!='#' )
//            {
//                AT[it+5]=Utils.converter(atcommand[2*(it-1)],atcommand[2*(it-1)+1]);
//                it2+=2;
//            }
//            else
//            {
//                switch( atcommand[it2] )
//                {
//                    case '0':	AT[it+5]=0;
//                    break;
//                    case '1':	AT[it+5]=1;
//                    break;
//                    case '2':	AT[it+5]=2;
//                    break;
//                    case '3':	AT[it+5]=3;
//                    break;
//                    case '4':	AT[it+5]=4;
//                    break;
//                    case '5':	AT[it+5]=5;
//                    break;
//                    case '6':	AT[it+5]=6;
//                    break;
//                    case '7':	AT[it+5]=7;
//                    break;
//                    case '8':	AT[it+5]=8;
//                    break;
//                    case '9':	AT[it+5]=9;
//                    break;
//                    case 'A':	AT[it+5]='A';
//                    break;
//                    case 'B':	AT[it+5]='B';
//                    break;
//                    case 'C':	AT[it+5]='C';
//                    break;
//                    case 'D':	AT[it+5]='D';
//                    break;
//                    case 'E':	AT[it+5]='E';
//                    break;
//                    case 'F':	AT[it+5]='F';
//                    break;
//                }
//                it2++;
//            }
//        }
//        else
//        {
//            AT[it+5]=atcommand[it];
//            it2++;
//        }
//        it++;
//    } 
//    length=it;
//    
//    AT[2]=2+length;
//    for(it=3;it<(5+length);it++)
//    {
//        checksum=checksum+AT[it];
//    }
//    while( (checksum>255))
//    {
//        checksum=checksum-256;
//    }
//    checksum=255-checksum;
//    AT[5+length]=checksum;
//    while(counter<(6+length))
//    {
//        XBee.print(AT[counter], BYTE); 
//        counter++;
//    }
//    counter=0;
//    clearCommand();
//    command[5]=AT[5];
//    command[6]=AT[6];
//    data_length=0;
//    error=parse_message(command);
//    if(error==0)
//    {
//        if(data_length>0)
//        {
//            for(it=0;it<data_length;it++)
//            {
//                commandAT[it]=data[it];
//                delay(20);
//            }
//        }
//        else
//        {
//            commandAT[0]=0x4F;
//            commandAT[1]=0x4B;
//        }
//    }
//    
//    free(AT);
//    AT=NULL;
//    free(ByteIN);
//    ByteIN=NULL;
//    return error;
//}

/*
 Function: Connect XBee, activating switch in Waspmote
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::ON()
{
    uint8_t error=2;
    //XBee.begin();
	beginxbee(XBEE_RATE, XBEE_UART);
    //XBee.setMode(XBEE_ON);
    if( protocol== ZIGBEE || protocol==XBEE_868 ) delay(500);
    else delay(50);
    error=0;
    XBee_ON=1;
    return error;
}


/*
 Function: disconnects XBee, switching it off and closing the UART
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::OFF()
{
    uint8_t error=2;
    XBee.close();
    XBee.setMode(XBEE_OFF);
    error=0;
    XBee_ON=0;
    return error;
}






/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Parameters: 
   packet :	A struct of packetXBee type
*/
uint8_t WaspXBeeCore::sendXBeePriv(packetXBee* packet)
{

    //if( TX==NULL ) return 2;
    uint8_t counter=0;
    uint8_t checksum=0; 
    long previous=0;
    uint16_t aux=0;
    uint8_t protegido=0;
    uint8_t tipo=0;
    uint8_t estado=1;
    int8_t error=2;
    uint8_t ByteIN[20];
    uint8_t old_netAddress[2];
    uint8_t net_Address_changed = 0;
    uint8_t a=5;

    clearCommand();

    error_TX=2;
    for(it=0;it<120;it++)
    {
        TXArray[it]=0;
    }
    TXArray[0]=0x7E;
    TXArray[1]=0x00;
    TXArray[4]=packet->packetID; // frame ID
    it=0;
    error_AT=2;
    if(protocol==XBEE_802_15_4)
    {
        if(packet->mode==BROADCAST)
        {

            tipo=15;
            TXArray[3]=0x00;
	    
			//FlagTimeLinshi = 1;TMsArray2[FlagTimeLinshi] = Flag1ms;

            previous=millis();
            error_AT=2;
			    //printf("zpp0123 before ");serialWrite('e', 2);serialWrite('1', 2);serialWrite(' ', 2);	delay_ms(500);
//            while( ((error_AT==1) || (error_AT==2)) )//&& (millis()-previous<5000) 
//            {
//                estado=getOwnNetAddress();
//				if( millis()<previous  ) previous=millis(); //avoid millis overflow problem
//            } /**/
//			   // printf("zpp0123 ");	delay_ms(500);
//			
//            old_netAddress[0]=sourceNA[0];
//            old_netAddress[1]=sourceNA[1];
	    	
//			FlagTimeLinshi = 12;TMsArray2[FlagTimeLinshi] = Flag1ms;

//            previous=millis();
//            error_AT=2;
//            while( ((error_AT==1) || (error_AT==2))  )//&& (millis()-previous<5000)
//            {
//                estado=setOwnNetAddress(0xFF,0xFF);
//                net_Address_changed = 1;
//		if( (long)millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//            } 
//
//		   FlagTimeLinshi = 13;TMsArray2[FlagTimeLinshi] = Flag1ms;

            error=2;
            while(a<13) // destination address
            {
                for(it=0;it<4;it++)
                {
                    TXArray[a]=0x00;
                    a++; 
                }
                for(it=0;it<2;it++)
                {
                    TXArray[a]=0x00;
                    a++;
                }
                for(it=0;it<2;it++)
                {
                    TXArray[a]=0xFF;
                    a++;
                }
            }
            TXArray[13]=0x00;
            TXArray[14]=packet->packetID;
            TXArray[15]=packet->numFragment;
            it=0;
            gen_frame(packet,TXArray,16);
			
            TXArray[2]=11+packet->frag_length; // fragment length

            for(it=3;it<(TXArray[2]+3);it++) // calculating checksum
            {
                checksum=checksum+TXArray[it];
            }
            while((checksum>255))
            {
                checksum=checksum-256;
            }
            checksum=255-checksum;
            TXArray[packet->frag_length+14]=checksum; // setting checksum
        }   
        if(packet->mode==UNICAST)
        {

            if(packet->address_type==_64B)
            {

					//printf("_64B"); 
                tipo=15;
                TXArray[3]=0x00;
                while(a<13) // destination address
                {
                    for(it=0;it<4;it++)
                    {
                        TXArray[a]=packet->macDH[it];
						//printf("%2x ",TXArray[a]);
                        a++; 
                    }
                    for(it=0;it<4;it++)
                    {
                        TXArray[a]=packet->macDL[it];
						//printf("%2x ",TXArray[a]);
                        a++;
                    }
                }
				
                previous=millis();
                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<5000) )
//                {
//                    estado=getOwnNetAddress();
//		    if( (long)millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//                }/**/
//			
//                old_netAddress[0]=sourceNA[0];
//                old_netAddress[1]=sourceNA[1];
//				
//                previous=millis();
//                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<5000) )
//                {
//                    estado=setOwnNetAddress(0xFF,0xFF);
//                    net_Address_changed = 1;
//		            //if( millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//		            if( millis()<previous ) previous=millis(); //avoid millis overflow problem
//                } /**/
                error=2;
                TXArray[13]=0x00;
                TXArray[14]=packet->packetID;
                TXArray[15]=packet->numFragment;
                it=0;
                gen_frame(packet,TXArray,16);
//				printf("after gen_frame:");
//		    for(int linshii=0;linshii<25;linshii++)
//			{
//				printf("%2x ",TXArray[linshii]);		
//			}

                TXArray[2]=11+packet->frag_length; // fragment length
                for(it=3;it<(TXArray[2]+3);it++) // calculating checksum
                {
                    checksum=checksum+TXArray[it];
                }
                while((checksum>255))
                {
                    checksum=checksum-256;
                }
                checksum=255-checksum;
                TXArray[packet->frag_length+14]=checksum; // setting checksum
            }
            if(packet->address_type==_16B)
            {

                tipo=9;
                TXArray[3]=0x01;
                TXArray[5]=packet->naD[0];
                TXArray[6]=packet->naD[1];
                TXArray[7]=0x00;
                TXArray[8]=packet->packetID;
                TXArray[9]=packet->numFragment;
                it=0;
                gen_frame(packet,TXArray,10);
                TXArray[2]=5+packet->frag_length; // fragment length
                for(it=3;it<(TXArray[2]+3);it++) // calculating checksum
                {
                    checksum=checksum+TXArray[it];
                }
                while((checksum>255))
                {
                    checksum=checksum-256;
                }
                checksum=255-checksum;
                TXArray[packet->frag_length+8]=checksum; // setting checksum
            }
        }
//        if(packet->mode==SYNC)
//        {
//            tipo=15;
//            TX[3]=0x00;
//            if(packet->opt==1) // Broadcast
//            {
//                previous=millis();
//                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<500) )
//                {
//                    estado=getOwnNetAddress();
//		    if( millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//                }/**/
			
//                old_netAddress[0]=sourceNA[0];
//                old_netAddress[1]=sourceNA[1];
		
//                previous=millis();
//                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<500) )
//                {
//                    estado=setOwnNetAddress(0xFF,0xFF);
//                    if( (error_AT==1) || (error_AT==2) )
//                    {
//                        TIME1=millis();
//                        delay(100);
//                    }
//                    net_Address_changed = 1;
//		    if(millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//                }/**/
//                error=2;
//                while(a<13) // destination address
//                {
//                    for(it=0;it<4;it++)
//                    {
//                        TX[a]=0x00;
//                        a++; 
//                    }
//                    for(it=0;it<2;it++)
//                    {
//                        TX[a]=0x00;
//                        a++;
//                    }
//                    for(it=0;it<2;it++)
//                    {
//                        TX[a]=0xFF;
//                        a++;
//                    }
//                }
//            }
//            else if(packet->opt==0) // UNICAST 64B
//            {
//                while(a<13) // destination address
//                {
//                    for(it=0;it<4;it++)
//                    {
//                        TX[a]=packet->macDH[it];
//                        a++; 
//                    }
//                    for(it=0;it<4;it++)
//                    {
//                        TX[a]=packet->macDL[it];
//                        a++;
//                    }
//                }
	    
//                previous=millis();
//                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<
//500) )
//                    estado=getOwnNetAddress();
//		    if( millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//                }/**/
			
//                old_netAddress[0]=sourceNA[0];
//                old_netAddress[1]=sourceNA[1];
	    
//                previous=millis();
//                error_AT=2;
//                while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<
//500) )
//                    estado=setOwnNetAddress(0xFF,0xFF);
//                    net_Address_changed = 1;
//		    if( millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
//                } /**/
//            }
//            TX[13]=0x00;
//            TX[14]=packet->packetID;
//            TX[15]=packet->numFragment;
//            it=0;
//            gen_frame(packet,TX,16);
//            TX[2]=11+packet->frag_length; // fragment length
//            for(it=3;it<(TX[2]+3);it++) // calculating checksum
//            {
//                checksum=checksum+TX[it];
//            }
//            while((checksum>255))
//            {
//                checksum=checksum-256;
//            }
//            checksum=255-checksum;
//            TX[packet->frag_length+14]=checksum; // setting checksum

    // AP = 2
    	//printf("printf qian 20");
//    	for(int linshii=0;linshii<20;linshii++)
//	{
//		printf("%2x ",TXArray[linshii]);		
//	}
//	printf("end");
        gen_frame_ap2(packet,TXArray,protegido,tipo);

		
    // Frame OK
//        printf("Tx:");
//		counter=0;
//        while(counter<(packet->frag_length+tipo+protegido))
//        {
//			printf("%2x ",TXArray[counter]);	
//            counter++;
//        }

        counter=0;
        while(counter<(packet->frag_length+tipo+protegido))
        {
            //XBee.print((char)TXArray[counter]); 
			printByte((char)TXArray[counter],  XBEE_UART);
            counter++;
        }


        counter=0;
    
//        command[0]=0xFF;
//        error=parse_message(command);
		error_TX = txStatusResponse();
//		printf("H%dL%d ",rx_buffer_head3,rx_buffer_tail3);
	    error = error_TX; 
        packet->deliv_status=delivery_status;
//    	FlagTimeLinshi = 14;TMsArray2[FlagTimeLinshi] = Flag1ms;
//        if( net_Address_changed )
//        {
//            error_AT=2;
//            previous=millis();
//            while( ((error_AT==1) || (error_AT==2)) && (millis()-previous<500
//) )
//                estado=setOwnNetAddress(old_netAddress[0],old_netAddress[1]);
//								if( (millis()-previous) < 0 ) previous=millis(); //add by otisyf;avoid millis overflow problem
//            }/**/
//        }
//		FlagTimeLinshi = 15;TMsArray2[FlagTimeLinshi] = Flag1ms;
    
    }
    

    //free(TX);
//    free(ByteIN);
    //TX=NULL;
//    ByteIN=NULL;
    return error;
}






uint8_t WaspXBeeCore:: analyserecframe(char * sendstr, uint8_t len)
{
#if ZPPDEBUGXBEEPRINT==1
	printf("get a right frame ");
#endif
	//这个处理也可以弄个单独的函数来做
	//TX _64   TX _16   要求xbee对外发送数据，第三位为0x00表示目标地址是64位的， 第三位为0x01表示目标地址是16位的，
	//当xbee知道了单片机要他对外发送数据，则xbee发给单片机的回复是 0x7e 0x00 0x03 0x89 帧号 是否OK 校验位 其中0x89是对TX的回答
	//[5] =0 success  1 No ACK   2 CCA  failure   3 Purged	
	if((sendstr[3]==0x00)||(sendstr[3]==0x01))//TX _64   TX _16
	{
		if(RecAPIFrameStr[5]==0x00)//说明xbee 知道要发送数据了，回复一个OK
			return 0;
		else                    //情况很多，暂时用返回1
			return 1;
	}
	//第3位为0x08 表示ATcmd	  问话
	//回答中[7] = 0 OK  1 ERROR  2 Invalid Command  3 Invalid Parameter 
	else if(sendstr[3]==0x08)
	{
#if ZPPDEBUGXBEEPRINT==1
		printf(" ATcmd ");
#endif
		if(RecAPIFrameStr[7]!=0x00)//对方不OK
		{
			return 1;
		}
		else
		{
#if ZPPDEBUGXBEEPRINT==1
			printf(" atok ");
#endif
			if((sendstr[1]==0x00)	&&(sendstr[2]==0x04)	)//表示询问xbee的参数，这个肯定有返回值，不同AT不同处理
			{
				if(strncmp(sendstr+5,"SL",2)==0)
				{
#if ZPPDEBUGXBEEPRINT==1
					printf("SL");
#endif
				        for(uint8_t it=0;it<4;it++)
				        {
				            sourceMacLow[it]=RecAPIFrameStr[it+8];
				        }					
				}
				else if(strncmp(sendstr+5,"SH",2)==0)
				{
#if ZPPDEBUGXBEEPRINT==1
					printf("SH");
#endif
					for(uint8_t it=0;it<4;it++)
					{
					    sourceMacHigh[it]=RecAPIFrameStr[it+8];
					}				
				}
				else if(strncmp(sendstr+5,"MY",2)==0)
				{
#if ZPPDEBUGXBEEPRINT==1
					printf("MY");
#endif
					for(uint8_t it=0;it<2;it++)
					{
					    sourceNA[it]=RecAPIFrameStr[it+8];
					}				
				}
				else if(strncmp(sendstr+5,"CH",2)==0)
				{
#if ZPPDEBUGXBEEPRINT==1
					printf("CH");
#endif
					channel=RecAPIFrameStr[8];				
				}
				else if(strncmp(sendstr+5,"ND",2)==0)
				{
#if ZPPDEBUGXBEEPRINT==1
					printf("ND");
#endif
					NDMacAddress[NDMacLength]=0;
					for(uint8_t it=0;it<8;it++)
					{
					    NDMacAddress[NDMacLength]=(NDMacAddress[NDMacLength]<<8)|RecAPIFrameStr[it+10];
					}
					NDMyAddress[NDMacLength]=0;
					for(uint8_t it=0;it<2;it++)
					{
					    NDMyAddress[NDMacLength]=(NDMyAddress[NDMacLength]<<8)|RecAPIFrameStr[it+8];
					}
					NDMacLength++;
									
				}



				else
				{
#if ZPPDEBUGXBEEPRINT==1
					printf(" else ");
#endif
				}					
			}
			else//其他为设置xbee参数，这里回复只能一种
			{
				if((RecAPIFrameStr[1]==0x00)	&&(RecAPIFrameStr[2]==0x05)	)
				{
					return 0;
				}
				else //这个麻烦了，把别的回复弄进来了//暂时return 0
				{
					return 0;
				}
			}
		}
		
	}
	return 0;
}


//发送sendstr ，这个一般为单片机对xbee的问话，发完之后，单片机根据情况看要不要查询xbee的回复
//关于是否要查询回复，回复是什么样的要求，可以根据sendstr的内容来定.
//假如需要有回复，因为这里是基于API=1的情况，所以，先看回复的一帧数据是不是API形式的
//如果是AT命令的，假如要xbee对外发送东西，那么看下xbee对单片机有没有应该OK就行了
//                  如果是问xbee的一些参数，比如mac adress  比如net address 这些，那要看回复的内容是不是针对这些的
//                  如果是设置xbe的一些参数，比如设置mac adress ，net address 的值，只要看xbee有没有应该OK就行了
//return:      0  OK   else WRONG or ELSE  or no more
//处理流程:   串口发送数据给xbee模块 
//             如果需要xbee回复，则查看 xbee回应，在规定时间内，有正确回应则return 0，没有回应1 或者其他数据
//             查看回应数据规则:  由发送的包数据确定回应规则。
//                                  规则一: 这里必须是API的格式，回应的帧应该和发送的帧是一样的 
//									规则二: 不同的AT命令，回应应该有对应的AT命令，比如AT08 回应该是88写成这样方式<X:03:88>即第3位（首位是第0位）为88
//
//
//
uint8_t WaspXBeeCore:: sendhandleresponse(char * sendstr, uint8_t len)
{
	char rulestrchild[20];
	unsigned long timei=0;
	unsigned int recnum=0;
	uint8_t ch;
	uint16_t offset;
	uint8_t lenrecapi=0;
	uint8_t check;
	uint8_t flagerr;
	uint32_t maxtimeoutms =0;
	//uint8_t apiidresponse;
	
	if((sendstr[5]=='N')&&(sendstr[6]=='D'))
	{
		NDMacLength=0;
		 maxtimeoutms = 3000;	//3秒
		printf(" ND ");
		for(uint8_t ind=0; ind<len;ind++)
			printf(" %2x",sendstr[ind]);
		printf(" send ");
	}
	else
	{
	   maxtimeoutms = 100; //100毫秒	
	}

	//发送数据包
	XBee.printstr(sendstr,len); 
//	if(FlagTimeLinshi==0){FlagTimeLinshi=1;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;}
//	else if(FlagTimeLinshi==2){FlagTimeLinshi=3;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;}
//	else if(FlagTimeLinshi==4){FlagTimeLinshi=5;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;}

	//生成接收的规则( 不包含API本身的那个规则)
	RuleStr[0]=0x00;
	//第3位为0x08 表示ATcmd	  问话
	if(sendstr[3]==0x08)
	{	
		//ATcmd 回答 第3位必须为0x88，同时，回答的具体命令和问的命令要一致
		sprintf(rulestrchild,"<X:03:88><X:05:%2x><X:06:%2x>",sendstr[5],sendstr[6]);
	}
	//TX _64   TX _16   要求xbee对外发送数据，第三位为0x00表示目标地址是64位的， 第三位为0x01表示目标地址是16位的，
	else if((sendstr[3]==0x00)||(sendstr[3]==0x01))
	{
		//当xbee知道了单片机要他对外发送数据，则xbee发给单片机的肯定是 0x7e 0x00 0x03 0x89 帧号 是否OK 校验位 其中0x89是对TX的回答
		sprintf(rulestrchild,"<X:03:89><X:01:00><X:02:03>");
	}
	strcat(RuleStr,rulestrchild);
	




	//判别有没有接收到符合规则的数据包
	
	//不停分析规则，先取出前规则来判断，直到所有规则被取出分析完毕
lookaAPIframe:	
	while(1)
	{
		//先找到API包，也就是说0x7E 开头的那种包找出来
		offset=0;
		while(1)
		{
			if( (recnum=XBee.available())==0 )
			{
				delay(1);
				timei++;
				//if((sendstr[6]==0x59)&&(sendstr[7]==0x00))
//				TimeusLinshi=timei;
				if(timei>maxtimeoutms){
					printf("timeout");
					return 2;
				}
			}
			else{
				while(recnum--)
				{
					ch=XBee.read();
#if ZPPDEBUGXBEEPRINT==1
					printf("ch=%2x ",ch);
#endif
					if(offset==0)
					{
						if(ch==0x7e)
						{
							RecAPIFrameStr[offset]=ch;
							offset++;
							continue;
						}
					}
					else
					{ 
						if(offset==2)
						{
							lenrecapi= (uint16_t)((((uint16_t)RecAPIFrameStr[1])<<8)|ch); //这里的ch不能写成 RecAPIFrameStr[2],因为[2]还没有赋值过来
#if ZPPDEBUGXBEEPRINT==1
							printf("lenapi=%2x ",lenrecapi);
#endif
						}
						else if(offset==4)
						{
							//有些frame没有帧数据，所以这个判断有问题
							if(ch!=sendstr[4]){offset=0;continue;}
						}
						else if(offset==(lenrecapi+3))
						{
							
							check=0;
							for(uint8_t icheck=3;icheck<offset;icheck++)
								check = check+RecAPIFrameStr[icheck];
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x ",check);
#endif
							check = 0xff -check;
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x <>",check);
							for(uint8_t icheck=0;icheck<offset;icheck++)
								printf("%2x ",RecAPIFrameStr[icheck]);
#endif
							if(check!=ch){offset=0;continue;}
							else 
							{
#if ZPPDEBUGXBEEPRINT==1
								printf("find a APIframe ");
#endif
								goto loop2;

							}
						}

						RecAPIFrameStr[offset]=ch;
						offset++;
					}
#if ZPPDEBUGXBEEPRINT==1
					printf("ofs=%2x ",offset);	
#endif									
				}
			}
		}
loop2: 	
		//分析API包子规则
		while(1)
		{
			//有一个不符合的，则offset=0;break;
			LenRuleStr = strlen(RuleStr);
			if(LenRuleStr==0)
			{
				//保留一些参数值
				goto loop1;
				
			}
			else
			{	
#if ZPPDEBUGXBEEPRINT==1
				printf("RuleStr:%s ",RuleStr);
#endif
				uint8_t i;
				uint8_t k;
				for(i=LenRuleStr-1;;i--)
				{
					if(RuleStr[i]=='<')break;
					if(i==0)break;

				}
				for(k=i;k<LenRuleStr;k++)
				{
					rulestrchild[k-i]=RuleStr[k];
				}
				rulestrchild[k-i]=0;
#if ZPPDEBUGXBEEPRINT==1
				printf("rulechild:%s ",rulestrchild);
#endif
				RuleStr[i]=0;
				if(rulestrchild[1]=='X')
				{	
					unsigned long indexl;
					unsigned long valuel;			  
					uint8_t index=0;
					uint8_t value = 0;
					if((atohexpart(rulestrchild+3, 2, &indexl)>0)&&(atohexpart(rulestrchild+6, 2, &valuel)>0))
					{
						index = indexl;
						value = valuel;
#if ZPPDEBUGXBEEPRINT==1
						printf(" index=%2x value=%2x ",index,value);
#endif
					}
					if( (lenrecapi+4)>index)//包的长度大于这个index值
					{ 
						if(RecAPIFrameStr[index]!=value){offset=0;break;}
					}
					else {offset=0;break;}
				}


			}
		}
	}


//到这里就是正确的接收到想要的包了，下面对接收的包进行处理
loop1: 
	flagerr=analyserecframe(sendstr, len);
	if((sendstr[5]=='N')&&(sendstr[6]=='D'))
		if(flagerr==0)
		{
			printf(" %d ",NDMacLength);
			goto lookaAPIframe;
		}	

return flagerr;		


}

//如果发现了一帧数据是其他xbee发过来的，那么返回0，把数据存在sendstr里面
//str:  接受到的数据
//len:  接收到的数据的长度，在使用时，注意返回的len要是比带进来的大，那么str的数据截止到带进来的大小数据量
//return: 0 正确接收到一帧0x80的数据， 其他还没有接收到
//注意：  当正确接收到一帧0x80数据时，全局变量LenRx80为0了，所以不可以通过这个全局变量看收到的数据大小，而是看*len这个
uint8_t WaspXBeeCore:: findRxframe(char * str, uint16_t *len)
{
	unsigned int recnum=0;
	uint8_t ch;
	uint8_t lenrecapi=0;
	uint8_t check;

			if( (recnum=XBee.available())==0 )
			{
				return 2;//还没有收到数据				
			}
			else
			{
				while(recnum--)
				{
					ch=XBee.read();
#if ZPPDEBUGXBEEPRINT==1
					printf("c%2x",ch);
#endif
					if(LenRx80==0)
					{
						if(ch==0x7e)
						{
							StrRx80[LenRx80]=ch;
							LenRx80++;
							continue;
						}
						else
							continue;
					}
					else
					{ 
						if(LenRx80==2)
						{
							lenrecapi= (uint16_t)((((uint16_t)StrRx80[1])<<8)|ch); //这里的ch不能写成 RecAPIFrameStr[2],因为[2]还没有赋值过来
#if ZPPDEBUGXBEEPRINT==1
							printf("lenapi=%2x ",lenrecapi);
#endif						//我们假定包都是小于256个字节的，这样，有如下判断
							if(StrRx80[1]!=0)
							{
								LenRx80=0;
								return 4;
								//continue;
							}
						}
						else if((LenRx80==3)&&(ch!=0x80))
						{
							LenRx80=0;
							return 3;
							//continue;
						}
						else if((LenRx80>3)&&(LenRx80==(StrRx80[2]+3)))//这里不能使用lenrecapi这个值
						{
							
							check=0;
							for(uint8_t icheck=3;icheck<LenRx80;icheck++)
								check = check+StrRx80[icheck];
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x ",check);
#endif
							check = 0xff -check;
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x <>",check);
							for(uint8_t icheck=0;icheck<LenRx80;icheck++)
								printf("%2x ",StrRx80[icheck]);
#endif
							if(check!=ch)
							{
								LenRx80=0;
								return 5;
								//continue;
							}
							else 
							{
#if ZPPDEBUGXBEEPRINT==1
								printf("find a APIframe ");
#endif
//								if(StrRx80[3]==0x80)
								StrRx80[LenRx80]=ch;
								LenRx80++;
									goto end;
//								else return 3;
								//return 0;
							}
						}

						StrRx80[LenRx80]=ch;
						LenRx80++;
					}
#if ZPPDEBUGXBEEPRINT==1
					printf("ofs=%2x ",LenRx80);	
#endif									
				}
			}
			
	return 1;

//loop4:
	

end:
//	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
	*len = (*len)<LenRx80 ? (*len):LenRx80 ;
//	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
	for(uint16_t i=0; i<(*len); i++)
		str[i]=StrRx80[i];
	LenRx80 = 0;		
	return 0;

}


//如果发现了一帧数据是其他xbee发过来的，那么返回0，把数据存在sendstr里面
//str:  接受到的数据
//len:  接收到的数据的长度，在使用时，注意返回的len要是比带进来的大，那么str的数据截止到带进来的大小数据量
//return: 0 正确接收到一帧0x80的数据， 其他还没有接收到
//注意：  当正确接收到一帧0x80数据时，全局变量LenRx80为0了，所以不可以通过这个全局变量看收到的数据大小，而是看*len这个
//这个与上面函数不同是，它是直接在Rx80Buffer里面寻找数据，实际上这个数组是由10ms 的TIM2中断处理后的串口3接收数据
//经过每次的10ms的处理，Rx80Buffer里面都是接收(RX80)数据包了，而非普通串口3接收的数据
uint8_t WaspXBeeCore:: findRxframeinRx80Buffer(char * str, uint16_t *len)
{
	unsigned int recnum=0;
	uint8_t ch;
	uint8_t lenrecapi=0;
	uint8_t check;

			if( (recnum=XBee.Rx80available())==0 )
			{
				return 2;//还没有收到数据				
			}
			else
			{
				while(recnum--)
				{
					ch=XBee.Rx80read();
#if ZPPDEBUGXBEEPRINT==1
					printf("c%2x",ch);
#endif
					if(LenRx80==0)
					{
						if(ch==0x7e)
						{
							StrRx80[LenRx80]=ch;
							LenRx80++;
							continue;
						}
						else
							continue;
					}
					else
					{ 
						if(LenRx80==2)
						{
							lenrecapi= (uint16_t)((((uint16_t)StrRx80[1])<<8)|ch); //这里的ch不能写成 RecAPIFrameStr[2],因为[2]还没有赋值过来
#if ZPPDEBUGXBEEPRINT==1
							printf("lenapi=%2x ",lenrecapi);
#endif						//我们假定包都是小于256个字节的，这样，有如下判断
							if(StrRx80[1]!=0)
							{
								LenRx80=0;
								return 4;
								//continue;
							}
						}
						else if((LenRx80==3)&&(ch!=0x80))
						{
							LenRx80=0;
							return 3;
							//continue;
						}
						else if((LenRx80>3)&&(LenRx80==(StrRx80[2]+3)))//这里不能使用lenrecapi这个值
						{
							
							check=0;
							for(uint8_t icheck=3;icheck<LenRx80;icheck++)
								check = check+StrRx80[icheck];
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x ",check);
#endif
							check = 0xff -check;
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x <>",check);
							for(uint8_t icheck=0;icheck<LenRx80;icheck++)
								printf("%2x ",StrRx80[icheck]);
#endif
							if(check!=ch)
							{
								LenRx80=0;
								return 5;
								//continue;
							}
							else 
							{
#if ZPPDEBUGXBEEPRINT==1
								printf("find a APIframe ");
#endif
//								if(StrRx80[3]==0x80)
								StrRx80[LenRx80]=ch;
								LenRx80++;
									goto end;
//								else return 3;
								//return 0;
							}
						}

						StrRx80[LenRx80]=ch;
						LenRx80++;
					}
#if ZPPDEBUGXBEEPRINT==1
					printf("ofs=%2x ",LenRx80);	
#endif									
				}
			}
			
	return 1;

//loop4:
	

end:
//	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
	*len = (*len)<LenRx80 ? (*len):LenRx80 ;
//	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
	for(uint16_t i=0; i<(*len); i++)
		str[i]=StrRx80[i];
	LenRx80 = 0;		
	return 0;

}












/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Parameters: 
   packet : A struct of packetXBee type
*/
uint8_t WaspXBeeCore::sendXBee(packetXBee* packet)
{
    uint8_t estadoSend=0;
    uint8_t maxPayload=0;
    uint8_t numPackets=0;
    uint8_t maxPackets=0;
    uint16_t lastPacket=0;
    uint16_t aux3=0;
    int8_t error=2;
    uint8_t firstPacket=1;
    uint16_t counter1=0;
    uint8_t type=0;
    uint8_t header=0;

  
    it=0;

  //FIXME Add max payloads for Cluster type
    if(protocol==XBEE_802_15_4)
    {
        if(encryptMode==0)
        {
            maxPayload=100;
        }
        else
        {
            if(packet->mode==BROADCAST)
            {
                maxPayload=95;
            }
            else
            {
                if(packet->address_type==_16B)
                {
                    maxPayload=98;
                }
                else
                {
                    maxPayload=94;
                }
            }
        }
    }
 
    
  

    switch(packet->typeSourceID)
    {
        case MY_TYPE: 	type=2;
        break;
        case MAC_TYPE: 	type=8;
        break;
        case NI_TYPE: 	while(packet->niO[it]!='#'){
            counter1++;
            it++;
        }
        type=counter1+1;
        break;
        default:	break;
    }
    header=3+firstPacket+type;
    aux3=packet->data_length;
    if((aux3+header)<=maxPayload)
    {
        lastPacket=aux3+header;
        numPackets=1;
    }
    else
    {
        while((aux3+header)>maxPayload)
        {
            numPackets++;
            aux3=aux3-maxPayload+header;
            firstPacket=0;
            header=3+firstPacket+type;
            if((aux3+header)<=maxPayload)
            {
                lastPacket=aux3+header;
                numPackets++;
            }
        }
    }
    maxPackets=numPackets;
  
  
    while(estadoSend!=1)
    {
        while(numPackets>0)
        {
            packet->numFragment=numPackets;
            if(numPackets==1) // last fragment
            {
                packet->frag_length=lastPacket;
            }
            else
            {
                packet->frag_length=maxPayload;
            }
            if(numPackets==maxPackets)
            {
                start=0;
                firstPacket=1;
                header=3+firstPacket+type;
                packet->endFragment=1;
            }
            else
            {
                start=finish+1;
                firstPacket=0;
                header=3+firstPacket+type;
                packet->endFragment=0;
            }
            if(numPackets==1)
            {
                finish=packet->data_length-1;
            }
            else
            {
                finish=start+packet->frag_length-header-1;
            }
            frag_length=packet->frag_length;

            error=sendXBeePriv(packet);
            if(error==0)
            {
                numPackets--;
                if(numPackets==0)
                {
                    estadoSend=1;
                }
                else delay(50);
            }
            else
            {
                numPackets=0;
                estadoSend=1;
            }
        }
    }
    return error;

}


/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
//int8_t WaspXBeeCore::send(uint8_t* address, uint8_t* data)
//{	
//	char macDest[17];
//	uint8_t length=0;
//	
//	Utils.hex2str(address, macDest);
//	
//	while( !((data[length]==0xAA) && (data[length+1]==0xAA)) ) length++;
//		
//	return send(macDest,(char*) data,0,length);
//}


/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
//int8_t WaspXBeeCore::send(char* address, uint8_t* data)
//{	
//	uint8_t length=0;
//
//	while( !((data[length]==0xAA) && (data[length+1]==0xAA)) ) length++;
//		
//	return send(address,(char*) data,0,length);
//}


/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
//int8_t WaspXBeeCore::send(uint8_t* address, char* data)
//{	
//	char macDest[17];
//	Utils.hex2str(address, macDest);
//		
//	return send(macDest,data,1,0);
//}


/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
//int8_t WaspXBeeCore::send(char* address, char* data)
//{
//	return send(address,data,1,0);
//}


/*
 Function: Send a packet from one XBee to another XBee in API mode
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
//int8_t WaspXBeeCore::send(char* address, char* data, uint8_t type, uint8_t dataMax)
//{
//	uint8_t maxPayload = 0;
//	uint8_t i=0;
//	uint8_t j=0;
//	char aux[2];
//	uint8_t error=2;
//	uint8_t destination[8];
//	uint8_t maxData = 0;
//	uint8_t checksum = 0;
//	uint8_t tipo = 0;
//			
//	switch( protocol )
//	{
//		case XBEE_802_15_4 :	if( !encryptMode ) maxPayload = 100;
//					else
//					{
//						if( !strcmp(address,"000000000000FFFF") ) maxPayload = 95;
//						else maxPayload = 94;
//					}
//					tipo = 15;
//					break;
//		case ZIGBEE :		if( !encryptMode )
//					{
//						if( !strcmp(address,"000000000000FFFF") ) maxPayload = 92;
//						else maxPayload = 84;
//					}
//					else
//					{
//						if( !strcmp(address,"000000000000FFFF") ) maxPayload = 74;
//						else maxPayload = 66;
//					}
//					tipo = 18;
//					break;
//		case DIGIMESH :		tipo = 18;
//					maxPayload = 73;
//					break;
//		case XBEE_900 :		tipo = 18;
//					if(encryptMode) maxPayload=80;
//					else maxPayload=100;
//					break;
//		case XBEE_868 :		tipo = 18;
//					maxPayload = 100;
//					break;
//	}
//
//	if( type==1 )
//	{
//		while( data[i]!='\0' )
//		{
//			maxData++;
//			i++;
//		}
//		i=0;
//	}
//	else if( type==0 ) maxData=dataMax;
//	
//	
//	if( maxData > maxPayload )
//	{
//		error_TX = 2;
//		return -1;
//	}
//	
//	while(j<8)
//	{
//		aux[i-j*2]=address[i];
//		aux[(i-j*2)+1]=address[i+1];
//		destination[j]=Utils.str2hex(aux);
//		i+=2;
//		j++;
//	}
//	
//	uint8_t* command = (uint8_t*) calloc(130,sizeof(uint8_t));
//	if(command==NULL){
//		error_TX = 2;
//		return -1;
//	}
//	
//	switch( protocol )
//	{
//		case XBEE_802_15_4 :	command[0] = 0x7E;
//					command[1] = 0x00;
//					command[2] = maxData+11;
//					command[3] = 0x00;
//					command[4] = 0x01;
//					for(it=0;it<8;it++) command[it+5]=destination[it];
//					command[13]=0x00;
//					for(it=0;it<maxData;it++) command[it+14]=data[it];
//					for(it=3;it<(maxData+14);it++)
//					{
//						checksum=checksum+command[it];
//					}
//					while( (checksum>255))
//					{
//						checksum=checksum-256;
//					}
//					checksum=255-checksum;
//					command[14+maxData]=checksum;
//					break;
//
//		case DIGIMESH	:	
//
//		case XBEE_900	:
//			
//		case XBEE_868	:
//			
//		case ZIGBEE	:	command[0] = 0x7E;
//					command[1] = 0x00;
//					command[2] = maxData+14;
//					command[3] = 0x10;
//					command[4] = 0x01;
//					for(it=0;it<8;it++) command[it+5]=destination[it];
//					command[13]=0xFF;
//					command[14]=0xFE;
//					command[15]=0x00;
//					command[16]=0x00;
//					for(it=0;it<maxData;it++) command[it+17]=data[it];
//					for(it=3;it<(maxData+17);it++)
//					{
//						checksum=checksum+command[it];
//					}
//					while( (checksum>255))
//					{
//						checksum=checksum-256;
//					}
//					checksum=255-checksum;
//					command[17+maxData]=checksum;
//					break;
//	}
//		
//	it=0;
//	while(it<(maxData+tipo))
//	{
//		XBee.print(command[it], BYTE); 
//		it++;
//	}
//	
//    		
//	if( protocol==XBEE_802_15_4 ) command[0]=0xFF;
//	else if( protocol==ZIGBEE || protocol==DIGIMESH || protocol==XBEE_900 || protocol==XBEE_868) command[0]=0xFE;
//		
//	error=parse_message(command);
//		
//	free(command);
//	command=NULL;
//	
//	return error;
//}



/*
 Function: Treats any data from the XBee UART
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/

int8_t WaspXBeeCore::treatData()
{
	int8_t error=2;
	uint8_t pos_aux=0;
	uint8_t pos_old=0;
	uint8_t pos_disor=0;
	uint8_t pos_max=0;
	uint8_t first_stop=1;

	//printf("indexNotModified=%d \r\n",indexNotModified);	
	command[0]=0xEE;
//	printf("-");
	error=parse_message(command);

	#if DEBUG802
	printf("%s=>%d:error_RX=%d pos=%d\r\n",__FUNCTION__,__LINE__,error_RX,pos);
	#endif
	pos_max=pos;
	pos_old=pos;
	pos_aux=pos;
	if(pos_aux>1) pos=1;
	printf("Length=%d ",packet_finished[pos-1]->data_length);
	for(int Li=0;Li<packet_finished[pos-1]->data_length;Li++)
	{
	  printf(" %2x",packet_finished[pos-1]->data[Li]);
	}
	printf("MacH%02x%02x%02x%02xL%02x%02x%02x%02x "\
	,packet_finished[pos-1]->macSH[0]\
	,packet_finished[pos-1]->macSH[1]\
	,packet_finished[pos-1]->macSH[2]\
	,packet_finished[pos-1]->macSH[3]\
	,packet_finished[pos-1]->macSL[0]\
	,packet_finished[pos-1]->macSL[1]\
	,packet_finished[pos-1]->macSL[2]\
	,packet_finished[pos-1]->macSL[3]\
	);
	printf("h=%d L=%d ",rx_buffer_head3,rx_buffer_tail3);
//
//	printf(" opt=%2x ",packet_finished[pos-1]->opt);
//
//	printf(" RSSI=%2x ",packet_finished[pos-1]->RSSI);
//	printf(" address_typeS=%2x ",packet_finished[pos-1]->address_typeS);
//	printf(" typeSourceID=%2x ",packet_finished[pos-1]->typeSourceID);
//	printf(" numFragment=%2x ",packet_finished[pos-1]->numFragment);
//	printf(" endFragment=%2x ",packet_finished[pos-1]->endFragment);



//	printf("+");
	//printf("%s:%d==>pos=%d error_RX=%d\r\n",__FUNCTION__,__LINE__,pos,error_RX);
	if( !error_RX )
	{
//		printf("!");
		while(pos_aux>0)
		{
			printf(">");
			printf("%x,",packet_finished[pos-1]->data[0]);
			switch(packet_finished[pos-1]->data[0])
//			switch(packet_finished[pos-1]->packetID)
			{
				case 0xFA:	error=new_firmware_received();
						if(!error)
						{
							while(pos_max>0)
							{
								free(packet_finished[pos_max-1]);
								packet_finished[pos_max-1]=NULL;
								pos_max--;
								pos_old--;
							}
							pos_aux=1;
						}
						break;
				case 0xFB:	new_firmware_packets();
						if( !firm_info.paq_disordered )
						{
							free(packet_finished[pos-1]);
							packet_finished[pos-1]=NULL;
							pos_old--;
						}
						else if(firm_info.paq_disordered==1){
							pos_disor=pos;
						}
						break;
				case 0xFC:	new_firmware_end();
						while(pos_max>0)
						{
							free(packet_finished[pos_max-1]);
							packet_finished[pos_max-1]=NULL;
							pos_max--;
							pos_old--;
						}
						pos_aux=1;
						break;
				case 0xFD:	upload_firmware();
						pos_old--;
						break;
				case 0xFE:
//						printf("FE");	
						request_ID();
						free(packet_finished[pos-1]);
						packet_finished[pos-1]=NULL;
						pos_old--;
						break;
				case 0xFF:	request_bootlist();
						free(packet_finished[pos-1]);
						packet_finished[pos-1]=NULL;
						pos_old--;
						break;
				case 0xF8:	delete_firmware();
						free(packet_finished[pos-1]);
						packet_finished[pos-1]=NULL;
						pos_old--;
						break;
				case 0xF9:	free(packet_finished[pos-1]);
						packet_finished[pos-1]=NULL;
						pos_old--;
						break;
				default:	if( programming_ON )
						{
							free(packet_finished[pos-1]);
							packet_finished[pos-1]=NULL;
							pos_old--;
						}
						break;
			}
			// Handle disordered packets
			if(!firm_info.paq_disordered && !pos_disor)
			{
				pos_aux--;
				pos++;
			}
			else if(firm_info.paq_disordered==1)
			{
				pos++;
				pos_aux--;
			}
			else if(pos_disor==1)
			{
				if( pos_old <= 1)
				{
					pos=1;
					pos_aux=1;
					pos_disor=0;
				}
				else if( first_stop )
				{
					pos=pos_max;
					first_stop=0;
				}
				else pos--;
				
			}
		}
	}
	else
	{
		// Clear reception struct 
		clearFinishArray();
		pos_old=0;

		// Flush input UART
		printf(" treatData flush ");
		serialFlush(XBEE_UART);
	}
	
	pos=pos_old;
	
	return error;
}


/*
 Function: Free the outcoming data buffer in the XBee
*/
//uint8_t WaspXBeeCore::freeXBee()
//{
//    uint8_t temp=0;
//    uint8_t error=2;
//    while(XBee.available()>0)
//    {
//        temp=XBee.read();
//        error=0;
//    }
//    return error;
//}




/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setDestinationParams(packetXBee* paq, uint8_t* address, const char* data, uint8_t type, uint8_t off_type)
{
	char macDest[17];
	Utils.hex2str(address, macDest);	
	return setDestinationParams(paq,macDest,data,type,off_type);
}

/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setDestinationParams(packetXBee* paq, uint8_t* address, int data, uint8_t type, uint8_t off_type)
{
	char macDest[17];
	Utils.hex2str(address, macDest);
	return setDestinationParams(paq,macDest,data,type,off_type);
}

/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setDestinationParams(packetXBee* paq, const char* address, const char* data, uint8_t type, uint8_t off_type)
{
    uint8_t* destination = (uint8_t*) calloc(8,sizeof(uint8_t));
    if( destination==NULL ) return -1;
    uint8_t i=0;
    uint8_t j=0;
    char aux[2];
	
    if( off_type==DATA_ABSOLUTE )
    {
        if( type==MAC_TYPE )
        {
            while(j<8)
            {
                aux[i-j*2]=address[i];	
                aux[(i-j*2)+1]=address[i+1];

//                aux[0]=address[i];	
//                aux[1]=address[i+1];
                destination[j]=Utils.str2hex(aux);//	printf("<%2x>",destination[j]);
                i+=2;
                j++;
            }
            for(uint8_t a=0;a<4;a++)
            {
                paq->macDH[a]=destination[a];
            }
            for(uint8_t b=0;b<4;b++)
            {
                paq->macDL[b]=destination[b+4];
            }
            paq->address_type=_64B;
        }
        if( type==MY_TYPE )
        {
            while(j<2)
            {
                aux[i-j*2]=address[i]; 
                aux[(i-j*2)+1]=address[i+1];
                destination[j]=Utils.str2hex(aux);
                i+=2;
                j++;
            }
            paq->naD[0]=destination[0];
            paq->naD[1]=destination[1];
            paq->address_type=_16B;
        }
        data_ind=0;
    }
    while( *data!='\0' )
    {
        paq->data[data_ind]=*data++;
        data_ind++;
        if( data_ind>=MAX_DATA ) break;
    }
    paq->data_length=data_ind;
    free(destination);
    return 1;
}


/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setDestinationParams(packetXBee* paq, const char* address, int data, uint8_t type, uint8_t off_type)
{
    uint8_t* destination = (uint8_t*) calloc(8,sizeof(uint8_t));
    if( destination==NULL ) return -1;
    uint8_t i=0;
    uint8_t j=0;
    char aux[2];
    char numb[10];
	
    if( off_type==DATA_ABSOLUTE )
    {
        if( type==MAC_TYPE )
        {
            while(j<8)
            {
                aux[i-j*2]=address[i];
                aux[(i-j*2)+1]=address[i+1];
                destination[j]=Utils.str2hex(aux);
                i+=2;
                j++;
            }
            for(uint8_t a=0;a<4;a++)
            {
                paq->macDH[a]=destination[a];
            }
            for(uint8_t a=0;a<4;a++)
            {
                paq->macDL[a]=destination[a+4];
            }
            paq->address_type=_64B;
        }
        if( type==MY_TYPE )
        {
            while(j<2)
            {
                aux[i-j*2]=address[i];
                aux[(i-j*2)+1]=address[i+1];
                destination[j]=Utils.str2hex(aux);
                i+=2;
                j++;
            }
            paq->naD[0]=destination[0];
            paq->naD[1]=destination[1];
            paq->address_type=_16B;
        }
        data_ind=0;
    }
    i=0;
    Utils.long2array(data,numb);
    while( numb[i]!='\0' )
    {
        paq->data[data_ind]=numb[i]++;
        data_ind++;
        i++;
        if( data_ind>=MAX_DATA ) break;
    }
    paq->data_length=data_ind;
    free(destination);
    return 1;
}

/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setOriginParams(packetXBee* paq, uint8_t type)
{
	return setOriginParams(paq,"",type);
}

/* FUnction: Sets to 'paq' the destination address and data
*/
int8_t WaspXBeeCore::setOriginParams(packetXBee* paq, const char* address, uint8_t type)
{
    uint8_t* origin = (uint8_t*) calloc(8,sizeof(uint8_t));
    if( origin==NULL ) return -1;
    uint8_t i=0;
    uint8_t j=0;
    char aux[2];
	
    if( type==MAC_TYPE )
    {
        if(Utils.sizeOf(address)<5)
        {
            getOwnMac();
            for(uint8_t a=0;a<4;a++)
            {
                paq->macOH[a]=sourceMacHigh[a];
            }
            for(uint8_t b=0;b<4;b++)
            {
                paq->macOL[b]=sourceMacLow[b];
            }
        }
        else
        {
            while(j<8)
            {
                aux[i-j*2]=address[i];
                aux[(i-j*2)+1]=address[i+1];
                origin[j]=Utils.str2hex(aux);
                i+=2;
                j++;
            }
            for(uint8_t a=0;a<4;a++)
            {
                paq->macOH[a]=origin[a];
            }
            for(uint8_t b=0;b<4;b++)
            {
                paq->macOL[b]=origin[b+4];
            }
        }
        paq->typeSourceID=MAC_TYPE;
    }
    if( type==MY_TYPE )
    {
        if(Utils.sizeOf(address)<2)
        {
            getOwnNetAddress();
            for(uint8_t a=0;a<2;a++)
            {
                paq->naO[a]=sourceNA[a];
            }
        }
        else
        {
            while(j<2)
            {
                aux[i-j*2]=address[i];
                aux[(i-j*2)+1]=address[i+1];
                origin[j]=Utils.str2hex(aux);
                i+=2;
                j++;
            }
            paq->naO[0]=origin[0];
            paq->naO[1]=origin[1];
        }
        paq->typeSourceID=MY_TYPE;
    }
    i=0;
    if( type==NI_TYPE )
    {
        while( *address!='\0' )
        {
            paq->niO[i]=*address++;
            i++;
        }
        paq->niO[i]='#';
        paq->typeSourceID=NI_TYPE;
    }
    free(origin);
    return 1;
}
/*
 * Function: Treats and parses the read bytes wich are a message sent by a 
 * remote XBee
 * 
 * Parameters:
 * 	data :	this is the pointer to new packet received by the XBee module. It
 * 			might be a fragment or a packet itself
 * 
 * Returns: Integer that determines if there has been any error 
 * 	error=2 --> The command has not been executed
 * 	error=1 --> There has been an error while executing the command
 * 	error=0 --> The command has been executed with no errors
 * 	error=-1 --> No more memory available
 * 
 * 	'packet_finished' : is the attribute where a maximum of 5 packets are 
 * 		created in order to contain all received information
 * 
 * Values: Stores in global "packet_finished" array the received message 
 * 
 *  
 * ====> 16-bit address frame type (ONLY XBEE-802.15.4)<=====
 * 'data' includes from 'Src Add' to 'checksum'
 *  __________________________________________________________________________
 * |      |           |           |         |      |     |         |          |
 * | 0x7E |   Length  | FrameType | Src Add | RSSI | Ops | RF Data | checksum |
 * |      | MSB | LSB |  (0x81)   |         |      |     |         |          |
 * |______|_____|_____|___________|_________|______|_____|_________|__________|
 *    0      1     2       3          4-5       6     7      8-n       n+1
 *  
 *  ====> 64-bit address frame type <=====
 * 'data' includes from 'Src Add' to 'checksum'
 *  __________________________________________________________________________
 * |      |           |           |         |      |     |         |          |
 * | 0x7E |   Length  | FrameType | Src Add | RSSI | Ops | RF Data | checksum |
 * |      | MSB | LSB |  (0x81)   |         |      |     |         |          |
 * |______|_____|_____|___________|_________|______|_____|_________|__________|
 *    0      1     2       3          4-11     12     13     14-n       n+1
 * 
 * 
 * ====> Receive packet 0x90 format (ONLY XBEE-802.15.4)<=====
 * 'data' includes from 'Src Add' to 'checksum'
 *  
_______________________________________________________________________________
__
 * |      |           |           |         |             |     |         
|          |
 * | 0x7E |   Length  | FrameType | 64-bit  |   Reserved  | Ops | RF Data | 
checksum |
 * |      | MSB | LSB |  (0x90)   | Src Add | 0xFF | 0xFE |     |         
|          |
 * |______|_____|_____|___________|_________|______|______|_____|_________|
__________|
 *    0      1     2       3          4-11     12     13     14     15-
n       n+1
 * 
 * 
 * ====> Explicit Rx Indicator 0x91 format <=====
 * 'data' includes from 'Src Add' to 'checksum'
 *  
_______________________________________________________________________________
________________________
 * |      |           |           |         |             |    |    |     
|     |     |         |          |     
 * | 0x7E |   Length  | FrameType | 64-bit  |   Reserved  | SD | DE | CID | 
PID | Opt | RF Data | checksum |
 * |      | MSB | LSB |  (0x90)   | Src Add | 0xFF | 0xFE |    |    |     
|     |     |         |          |     
 * |______|_____|_____|___________|_________|______|______|____|____|_____|
_____|_____|_________|__________|
 *    0      1     2       3          4-11     12     13    14   15  16-17 18-
19   20     21-n       n+1
 * 
 * 
*/
int8_t WaspXBeeCore::readXBee(uint8_t* data)
{
    uint16_t aux=0;
    uint16_t aux2=0;
    int8_t error=2;
    uint16_t cont3=0;
    uint8_t header=0;
    uint16_t temp=0;
    uint8_t index1=0;
    uint8_t index2=0;
    long time=0;
    uint8_t finishIndex=0; 
    if( data_length < 12 ) return 1; 
    temp=0;   
    aux=0;
    pos++;
	finishIndex=getFinishIndex();	    
	if( pos > MAX_FINISH_PACKETS )
	{
		switch( replacementPolicy )
		{
			case	XBEE_FIFO:	// recalculate index to put the new packet 
								finishIndex=getIndexFIFO();
								break;
			case	XBEE_LIFO:	// recalculate index to put the new packet 
								finishIndex=getIndexLIFO();
								break;
			case	XBEE_OUT:	// last received packet must be discarded								
								pos--;
								return 2;								
								break;
		}
	}					
					
	// memory allocation for a new packet				
	packet_finished[finishIndex] = (packetXBee*) calloc(1,sizeof(packetXBee));
									
	// if no available memory then exit with error
	if(packet_finished[finishIndex]==NULL)
	{		
		return 1;
	}	
                     
    /**************************************************************************
    * Store packet fields in packet_finished structure depending on the RX 
    * frame type. There are four possibilities:
    * 	_16B --> for 16-Bit address RX frames (only XBee-802.15.4 protocol)
    * 	_64B --> for 64-bit address RX frames (only XBee-802.15.4 protocol) 
    * 	NORMAL_RX --> for normal RX frames (for all XBee protocols but 802.15.4)
    * 	EXPLICIT_RX --> for explicit RX indicator frames (for all XBee 
protocols but 802.15.4) 
    **************************************************************************/
	switch(add_type)
	{
		case _16B:	
		////////////////////////////////////////////////////////////////////////
		// when a 16-bit address is used for XBee-802.15.4: (frame type=0x81)
		////////////////////////////////////////////////////////////////////////	
			
			// store information in pendingFragments structure
			packet_finished[finishIndex]->address_typeS=_16B;
			packet_finished[finishIndex]->time=millis();
			
			// store source address
			packet_finished[finishIndex]->naS[0]=data[8];		
			packet_finished[finishIndex]->naS[1]=data[9];
			//printf("%s:%d==>%d[finishIndex]->naS[0]=%x\r\n",__FUNCTION__,__LINE__,finishIndex,data[8]);
			//printf("%s:%d==>%d[finishIndex]->naS[1]=%x\r\n",__FUNCTION__,__LINE__,finishIndex,data[9]);
			// store RSSI
			packet_finished[finishIndex]->RSSI=data[2];
			packet_finished[finishIndex]->opt=data[3];
			
			// set BROADCAST mode if 'Options' indicate so
			// set UNICAST mode otherwise
			if( packet_finished[finishIndex]->opt==0x01 || 
				packet_finished[finishIndex]->opt==0x02 )
			{
				packet_finished[finishIndex]->mode=BROADCAST;
			}
			else
			{
				packet_finished[finishIndex]->mode=UNICAST;
			}	
			
			// calculate cmdData header's length as the following length summatory:
			// Src Address (2B) + RSSI (1B) + Options (1B)+application ID(1B)+
			//Fragment Number(1B)+firstfragment(1B)+opt(1B)+SourceID(2B)
			header = 2+1+1+1+1+1+1+2;		
		
			if( header > data_length ) data_length=header;
		
			// set the fragment length as DATA length
			// 'data_length' is cmdData length
			// 'header' is the header included in cmdData
			packet_finished[finishIndex]->data_length = data_length - header;

		
			// copy DATA field to packet fragment structure
			for( int j=0 ; j<packet_finished[finishIndex]->data_length ; j++ )
			{     
				packet_finished[finishIndex]->data[j] = char(data[j+header]);
			}
			   
			break;		
			
			
		case _64B:
		////////////////////////////////////////////////////////////////////////
		// when a 64-bit address is used for XBee-802.15.4: (frame type=0x80)
		////////////////////////////////////////////////////////////////////////
		
			// store information in pendingFragments structure
			packet_finished[finishIndex]->address_typeS=_64B;
			packet_finished[finishIndex]->time=millis();
			
			// store High Source Address 
			packet_finished[finishIndex]->macSH[0]=data[0];
			packet_finished[finishIndex]->macSH[1]=data[1];
			packet_finished[finishIndex]->macSH[2]=data[2];
			packet_finished[finishIndex]->macSH[3]=data[3];
					
			// store Low Source Address 
			packet_finished[finishIndex]->macSL[0]=data[4];
			packet_finished[finishIndex]->macSL[1]=data[5];
			packet_finished[finishIndex]->macSL[2]=data[6];
			packet_finished[finishIndex]->macSL[3]=data[7];	
					                
			packet_finished[finishIndex]->RSSI = data[8];
			packet_finished[finishIndex]->opt = data[9];
			
			// Depending on the selected 'Options',
			// UNICAST or BROADCAST mode is chosen
			if( (packet_finished[finishIndex]->opt == 0x01) || 
				(packet_finished[finishIndex]->opt == 0x02) 	)
			{
				packet_finished[finishIndex]->mode=BROADCAST;
			}
			else
			{
				packet_finished[finishIndex]->mode=UNICAST;
			}			
			
			// calculate cmdData header's length as the following length summatory:
			// Src Address (8B) + RSSI (1B) + Options (1B)
			header=8+1+1;
		
			if( header>data_length ) data_length=header;
			
			// set the packet length as DATA length
			// 'data_length' is cmdData length
			// 'header' is the header included in cmdData	
			packet_finished[finishIndex]->data_length = data_length - header;
				
			// copy DATA field to packet fragment structure
			for( int j=0 ; j<packet_finished[finishIndex]->data_length ; j++ )
			{     
				packet_finished[finishIndex]->data[j] = char(data[j+header]);
			}		
		   
			break;		
			
			
		case NORMAL_RX:
		////////////////////////////////////////////////////////////////////////
		// when a normal received packet which Frame Type is: 0x90 (NORMAL_RX)
		////////////////////////////////////////////////////////////////////////
		
			// store timeStamp
			packet_finished[finishIndex]->time=millis();

			// store High MAC address from source
			packet_finished[finishIndex]->macSH[0] = data[0];
			packet_finished[finishIndex]->macSH[1] = data[1];
			packet_finished[finishIndex]->macSH[2] = data[2];
			packet_finished[finishIndex]->macSH[3] = data[3];
			
			// store Low MAC address from source
			packet_finished[finishIndex]->macSL[0] = data[4];
			packet_finished[finishIndex]->macSL[1] = data[5];
			packet_finished[finishIndex]->macSL[2] = data[6];
			packet_finished[finishIndex]->macSL[3] = data[7];
			
			// store network address from source
			packet_finished[finishIndex]->naS[0] = data[8];
			packet_finished[finishIndex]->naS[1] = data[9];
			
			// set UNICAST mode in packet
			packet_finished[finishIndex]->opt = data[10];
			
			// set BROADCAST mode if 'Options' indicate so
			if( (packet_finished[finishIndex]->opt & 0x0F) == 0x02 )
			{
				packet_finished[finishIndex]->mode = BROADCAST;
			}			
			else
			{
				packet_finished[finishIndex]->mode=UNICAST;
			}	
			
			// calculate cmdData header's length as the following length summatory:
			// Src_Add (8B) + Reserved(2B) + Options(1B)
			header=8+2+1;
		
			if( header>data_length ) data_length=header;
		
			// set the fragment length as DATA length
			// 'data_length' is cmdData length
			// 'header' is the header included in cmdData
			packet_finished[finishIndex]->data_length=data_length-header;
			
			// copy DATA field to packet fragment structure
			for( int j=0 ; j < packet_finished[finishIndex]->data_length ; j++ )
			{     
				packet_finished[finishIndex]->data[j] = char(data[j+header]);
			}	
							     					
			// no RSSI information in XBee packets
			packet_finished[finishIndex]->RSSI=0;	 
						   
			break;
			
			
			
		case EXPLICIT_RX:
		////////////////////////////////////////////////////////////////////////
		// when an explicit RX packet which Frame Type is: 0x91 (EXPLICIT_RX)
		////////////////////////////////////////////////////////////////////////
			   		
			// store timeStamp
			packet_finished[finishIndex]->time = millis();
				
			// store High MAC address from source
			packet_finished[finishIndex]->macSH[0]=data[0];
			packet_finished[finishIndex]->macSH[1]=data[1];
			packet_finished[finishIndex]->macSH[2]=data[2];
			packet_finished[finishIndex]->macSH[3]=data[3];
			
			// store Low MAC address from source			
			packet_finished[finishIndex]->macSL[0]=data[4];	
			packet_finished[finishIndex]->macSL[1]=data[5];	
			packet_finished[finishIndex]->macSL[2]=data[6];	
			packet_finished[finishIndex]->macSL[3]=data[7];
	
			// store network address from source
			packet_finished[finishIndex]->naS[0]=data[8];
			packet_finished[finishIndex]->naS[1]=data[9];	
			
			// store cluster information
			packet_finished[finishIndex]->mode=CLUSTER;
			packet_finished[finishIndex]->SD=data[10];
			packet_finished[finishIndex]->DE=data[11];
			
			// store Cluster ID
			packet_finished[finishIndex]->CID[0]=data[12];
			packet_finished[finishIndex]->CID[1]=data[13];
			
			// Store Profile ID
			packet_finished[finishIndex]->PID[0]=data[14];
			packet_finished[finishIndex]->PID[1]=data[15];
		
			// Store Options
			packet_finished[finishIndex]->opt = data[16];
			
			// set BROADCAST mode if 'Options' indicate so
			if( (packet_finished[finishIndex]->opt & 0x0F) == 0x02)
			{
				packet_finished[finishIndex]->mode=BROADCAST;
			}			
			
			// calculate cmdData header's length as the following length summatory:
			// Src Address(8B) + Reserved(2B) + SE(1B) + DE(1B) + CID(2B) + PID(2B) + Opts(1B)
			header = 8+2+1+1+2+2+1;
		
			if( header>data_length ) data_length=header;
		
			// set the fragment length as DATA length
			// 'data_length' is cmdData length
			// 'header' is the header included in cmdData	
			packet_finished[finishIndex]->data_length = data_length-header;
		
			// copy DATA field to packet fragment structure
			for( int j=0 ; j < packet_finished[finishIndex]->data_length ; j++ )
			{     
				packet_finished[finishIndex]->data[j] = char(data[j+header]);
			}
			break;
							     					
			// no RSSI information in XBee packets
			packet_finished[finishIndex]->RSSI=0;	 
		
		default:
			break;
			   
	}          
        

    return 0;
}

#if 0
/*
 Function: Treats and parses the read bytes wich are a message sent by a remote XBee
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
   error=-1 --> No more memory available
 Values: Stores in global "packet_finished" array the received message  
*/
int8_t WaspXBeeCore::readXBee(uint8_t* data)
{
    uint16_t aux=0;
    uint16_t aux2=0;
    uint8_t aux3=0;
    int8_t error=2;
    uint16_t cont3=0;
    uint8_t header=0;
    uint16_t temp=0;
    int16_t temp2=0;
    uint16_t temp3=0;
    uint8_t samePacket=0;
    uint8_t comp=0;
    uint8_t index1=0;
    uint8_t index2=0;
    long time=0;
    uint8_t finishIndex=0;
  
    it=0;//printf("index=%d ",indexNotModified);	
    if( indexNotModified )
    {
        for(it=0;it<MAX_FINISH_PACKETS;it++)
        {
            if( pendingFragments[it]==NULL ) break;
        }printf("it=%d ",it);
        nextIndex1=it;
    }
	
    it=0;	
	//printf("lenth=%d ",data_length);
    temp=0;
    if( protocol!=XBEE_802_15_4 && data_length<12 ) return 1;
    if( protocol==XBEE_802_15_4 && add_type==_16B && data_length<5 ) return 1;
    if( protocol==XBEE_802_15_4 && add_type==_64B && data_length<11 ) return 1;
	
    while(temp<MAX_FINISH_PACKETS) // miramos si ya tenemos fragmentos de este paquete global
    {
		//printf("tem=%d ",temp);
        it=0;
        temp2=0;
        temp3=0;
        if( pendingFragments[temp]->time > 0 )
        {
            if(protocol==XBEE_802_15_4)
            {
                if(add_type==_16B)
                {
                    if( (data[4]) == pendingFragments[temp]->packetID )
                    {
                        if((data[6])==35)
                        {
                            temp2=1;
                        }
                        else
                        {
                            temp2=0;
                        }
                        temp3=data[6+temp2];
                        it=0;
                        switch(temp3)
                        {
                            case MY_TYPE:	if(pendingFragments[temp]->naO[0]==data[7+temp2])
                            {
                                if(pendingFragments[temp]->naO[1]==data[8+temp2])
                                {
                                    samePacket=1;
                                }
                            }
                            break;
                            case 1:	for(it=0;it<4;it++)
                            {
                                if(pendingFragments[temp]->macOH[it]!=data[7+temp2+it])
                                {
                                    comp=1;
                                    break;
                                }
                            }
                            for(it=0;it<4;it++)
                            {
                                if(pendingFragments[temp]->macOL[it]!=data[11+temp2+it])
                                {
                                    comp=1;
                                    break;
                                }
                            }
                            if(comp==0)
                            {
                                samePacket=1;
                            }
                            break;
                            case 2:	while(pendingFragments[temp]->niO[it]!=35)
                            {
                                if(pendingFragments[temp]->niO[it]!=data[7+it+temp2])
                                {
                                    comp=1;
                                    break;
                                }
                                it++;
                            }
                            if(comp==0)
                            {
                                samePacket=1;
                            }
                            break;
                        }
                        if(samePacket==1)  // Fragmento del mismo paquete
                        {
                            index1=temp;
                            temp=MAX_FINISH_PACKETS;
                            if(temp2==1)
                            {
                                pendingFragments[index1]->totalFragments=data[5];
                            }
                            pendingFragments[index1]->recFragments++;
                            pendingFragments[index1]->RSSI=pendingFragments[index1]->RSSI+data[2];
                            index2=data[5]-1;
                            if( index2>= MAX_FRAG_PACKETS ){
                                freeIndexMatrix(index1);
                                return -1;
                            }
                        }
                    }
                }
                else if(add_type==_64B)
                {
					//printf("64B");
                    if((data[10])==pendingFragments[temp]->packetID)
                    {
						printf("d10=%x ",data[10]);
                        if((data[12])==35)
                        {
                            temp2=1;
                        }
                        else
                        {
                            temp2=0;
                        }
                        temp3=data[12+temp2];
                        it=0;
                        switch(temp3)
                        {
                            case MY_TYPE:	if(pendingFragments[temp]->naO[0]==data[13+temp2])
                            {
                                if(pendingFragments[temp]->naO[1]==data[14+temp2])
                                {
                                    samePacket=1;
                                }
                            }
                            break;
                            case 1:	for(it=0;it<4;it++)
                            {
                                if(pendingFragments[temp]->macOH[it]!=data[13+temp2+it])
                                {
                                    comp=1;
                                    break;
                                }
                            }
                            for(it=0;it<4;it++)
                            {
                                if(pendingFragments[temp]->macOL[it]!=data[17+temp2+it])
                                {
                                    comp=1;
                                    break;
                                }
                            }
                            if(comp==0)
                            {
                                samePacket=1;
                            }
                            break;
                            case 2:	while(pendingFragments[temp]->niO[it]!=35)
                            {
                                if(pendingFragments[temp]->niO[it]!=data[13+it+temp2])
                                {
                                    comp=1;
                                    break;
                                }
                                it++;
                            }
                            if(comp==0)
                            {
                                samePacket=1;
                            }
                            break;
                        }
                        if(samePacket==1)  // Fragmento del mismo paquete
                        {
                            index1=temp;
                            temp=MAX_FINISH_PACKETS;
                            if(temp2==1)
                            {
                                pendingFragments[index1]->totalFragments=data[11];
                            }
                            pendingFragments[index1]->recFragments++;
                            pendingFragments[index1]->RSSI=pendingFragments[index1]->RSSI+data[8];
                            index2=data[11]-1;
                            if( index2>= MAX_FRAG_PACKETS ){
                                freeIndexMatrix(index1);
                                return -1;
                            }
                        }
                    }
                }
            }

        }
        temp++;
    }
    it=0;
    if(samePacket==0) // Fragmento de un paquete nuevo
    {
		//printf("sam=0");//进行到这里了
        index1=nextIndex1;
        indexNotModified=1;
        if( pendingPackets>= MAX_FINISH_PACKETS ){printf("penMax");
            freeIndex();
            index1=nextIndex1;
        }
        pendingPackets++;//printf("pends=%x ind1=%x ",pendingPackets,index1);
        pendingFragments[index1] = (index*) calloc(1,sizeof(index));
        if(pendingFragments[index1]==NULL){	printf("NULL");
            freeAll();
            return -1;
        }
        if(protocol==XBEE_802_15_4)
        {
            if(add_type==_16B)
            {
                pendingFragments[index1]->address_typeS=add_type;
                pendingFragments[index1]->packetID=data[4];
                pendingFragments[index1]->time=millis(); /**/
                for(it=0;it<2;it++)
                {
                    pendingFragments[index1]->naS[it]=data[it];
                }
                if((data[6])==35)
                {
                    temp2=1;
                }
                else
                {
                    temp2=0;
                }
                pendingFragments[index1]->typeSourceID=data[6+temp2];
                it=0;
                switch(pendingFragments[index1]->typeSourceID)
                {
                    case MY_TYPE:	pendingFragments[index1]->naO[0]=data[7+temp2];
                    pendingFragments[index1]->naO[1]=data[8+temp2];
                    break;
                    case MAC_TYPE:	for(it=0;it<4;it++)
                    {
                        pendingFragments[index1]->macOH[it]=data[7+it+temp2];
                    }
                    for(it=0;it<4;it++)
                    {
                        pendingFragments[index1]->macOL[it]=data[11+it+temp2];
                    }
                    break;
                    case NI_TYPE:	while(data[7+temp2+it]!=35)
                    {
                        pendingFragments[index1]->niO[it]=char(data[7+it+temp2]);
                        it++;
                        if(it>20)
                        {
                            break;
                        }
                    }
                    pendingFragments[index1]->niO[it]=char(35);
                    break;
                }
                if(temp2==1)
                {
                    pendingFragments[index1]->totalFragments=data[5];
                }
                pendingFragments[index1]->recFragments=1;
                index2=data[5]-1;
                if( index2>= MAX_FRAG_PACKETS ){
                    freeIndexMatrix(index1);
                    return -1;
                }
                pendingFragments[index1]->RSSI=data[2];
                pendingFragments[index1]->opt=data[3];
                if( (pendingFragments[index1]->opt==0x01) || (pendingFragments[index1]->opt==0x02) )
                {
                    pendingFragments[index1]->mode=BROADCAST;
                }
                else
                {
                    pendingFragments[index1]->mode=UNICAST;
                }
            }
            else if(add_type==_64B)
            {	
	           // printf("adtyp=%x ID=%x \r\n",add_type,data[10]);
                pendingFragments[index1]->address_typeS=add_type;
                pendingFragments[index1]->packetID=data[10]; //0x52
                pendingFragments[index1]->time=millis(); /**/
#if DEBUG
                XBee.println("Entro en 64B");
                XBee.println(pendingFragments[index1]->packetID,HEX);
                XBee.println(pendingFragments[index1]->time,DEC);
#endif
                for(it=0;it<4;it++)
                {	
	              //  printf("it=%d==>H%x \r\n",it,data[it]);
                    pendingFragments[index1]->macSH[it]=data[it];
#if DEBUG
                    XBee.print(pendingFragments[index1]->macSH[it],HEX);
#endif
                }
                for(it=0;it<4;it++)
                {	
	                //printf("it=%d==>L%x \r\n",it,data[it+4]);
                    pendingFragments[index1]->macSL[it]=data[it+4];
#if DEBUG
                    XBee.print(pendingFragments[index1]->macSL[it],HEX);
#endif
                }
                if((data[12])==35)//0x23
                {
                    temp2=1;
                }
                else
                {
                    temp2=0;
                }
//				printf("%s==>%d:ID=%x \r\n",__FUNCTION__,__LINE__,data[12+temp2]);
                pendingFragments[index1]->typeSourceID=data[12+temp2];
#if DEBUG
                XBee.println("");
                XBee.println(pendingFragments[index1]->typeSourceID,DEC);
#endif
                it=0;
                switch(pendingFragments[index1]->typeSourceID)
                {
                    case MY_TYPE:	pendingFragments[index1]->naO[0]=data[13+temp2];
                    pendingFragments[index1]->naO[1]=data[14+temp2];
#if DEBUG
                    XBee.println(pendingFragments[index1]->naO[0],HEX);
                    XBee.println(pendingFragments[index1]->naO[1],HEX);
#endif
                    break;
                    case MAC_TYPE:	for(it=0;it<4;it++)
                    {
                        pendingFragments[index1]->macOH[it]=data[13+it+temp2];
                    }
                    for(it=0;it<4;it++)
                    {
                        pendingFragments[index1]->macOL[it]=data[17+it+temp2];
                    }
                    break;
                    case NI_TYPE:	while(data[13+temp2+it]!=35)
                    {
                        pendingFragments[index1]->niO[it]=char(data[13+it+temp2]);
#if DEBUG
                        XBee.print(pendingFragments[index1]->niO[it],BYTE);
#endif
                        it++;
                        if(it>20)
                        {
                            break;
                        }
                    }
                    pendingFragments[index1]->niO[it]=char(35);
                    break;
                }
                pendingFragments[index1]->RSSI=data[8];
                pendingFragments[index1]->opt=data[9];
                if( (pendingFragments[index1]->opt==0x01) || (pendingFragments[index1]->opt==0x02) )
                {
                    pendingFragments[index1]->mode=BROADCAST;
                }
                else
                {
                    pendingFragments[index1]->mode=UNICAST;
                }
                if(temp2==1)
                {
                    pendingFragments[index1]->totalFragments=data[11];
                }
                pendingFragments[index1]->recFragments=1;
                index2=data[11]-1;
                if( index2>= MAX_FRAG_PACKETS ){
					printf("%s==>%dNULL\r\n",__FUNCTION__,__LINE__);
                    freeIndexMatrix(index1);
                    return -1;
                }
#if DEBUG
                XBee.print("Index1: ");
                XBee.println(index1,DEC);
                XBee.print("Index2: ");
                XBee.println(index2,DEC);
#endif
            }
        }

    }
    samePacket=0;	
    packet_fragments[index1][index2] = (matrix*) calloc(1,sizeof(matrix));
    if(packet_fragments[index1][index2]==NULL){printf("NULL");
        freeAll();
        return -1;
    }
    if(protocol==XBEE_802_15_4)
    {
        if(add_type==_16B)
        {
            packet_fragments[index1][index2]->numFragment=data[5];
            if(data[6]==35)
            {
                packet_fragments[index1][index2]->endFragment=1;
            }
            else
            {
                packet_fragments[index1][index2]->endFragment=0;
            }
            aux2=packet_fragments[index1][index2]->endFragment;
            it=0;
            cont3=0;
            switch(pendingFragments[index1]->typeSourceID)
            {
                case MY_TYPE:	cont3=2;
                break;
                case MAC_TYPE:	cont3=8;
                break;
                case NI_TYPE:	while(data[7+aux2+it]!=35)
                {
                    it++;
                    cont3++;
                }
                cont3++;
                break;
            }
            header=4+3+aux2+cont3;
            if( header>data_length ) data_length=header;
            packet_fragments[index1][index2]->frag_length=data_length-header;

            for(it=0;it<packet_fragments[index1][index2]->frag_length;it++)
            {     
                packet_fragments[index1][index2]->data[it]=char(data[it+header]);
            }
        }
        if(add_type==_64B)
        {	//printf("da12=%x ",data[12]);
            packet_fragments[index1][index2]->numFragment=data[11];
            if(data[12]==35)
            {
                packet_fragments[index1][index2]->endFragment=1;
            }
            else
            {
                packet_fragments[index1][index2]->endFragment=0;
            }
            aux2=packet_fragments[index1][index2]->endFragment;
            it=0;
            cont3=0;
            switch(pendingFragments[index1]->typeSourceID)
            {
                case MY_TYPE:	cont3=2;
                break;
                case MAC_TYPE:	cont3=8;
                break;
                case NI_TYPE:	while(data[13+aux2+it]!=35)
                {
                    it++;
                    cont3++;
                }
                cont3++;
                break;
            }
            header=10+3+aux2+cont3;
            if( header>data_length ) data_length=header;
            packet_fragments[index1][index2]->frag_length=data_length-header;

            for(it=0;it<packet_fragments[index1][index2]->frag_length;it++)
            {     
                packet_fragments[index1][index2]->data[it]=char(data[it+header]);
            }
        }
    }
	
 
    totalFragmentsReceived++;
	//printf("tot=%x ",totalFragmentsReceived);
    if(totalFragmentsReceived==MAX_FRAG_PACKETS)
    {
        totalFragmentsReceived=0;
    }
	
    if(pendingFragments[index1]->recFragments==pendingFragments[index1]->totalFragments)
    {
        pendingFragments[index1]->complete=1;
    }

    temp=0;
    temp2=0;
    temp3=0;
    aux=0;
    while(temp<MAX_FINISH_PACKETS)
    {	//printf("tm=%d ",temp);
        it=0;
        time=millis();/**/
        if( (pendingFragments[temp]!=NULL) )
        {	//printf("!NU");
            if ( (pendingFragments[temp]->time>0) )
            {	//printf("tim>");
                if( ((time-pendingFragments[temp]->time)<=TIMEOUT) )
                {	//printf("<TIM");
                    if ((pendingFragments[temp]->time>0) ) // se mira el contenido
                    {
                        if(pendingFragments[temp]->complete==1)
                        {	//printf("cmpl=1");
                            nextIndex1=temp;
                            indexNotModified=0;
                            pos++;
                            finishIndex=getFinishIndex();//printf("fnsh=%x ",finishIndex);
                            if( pos>=MAX_FINISH_PACKETS ){//printf("pos>%x ",pos);
                                switch( replacementPolicy )
                                {
                                    case	XBEE_FIFO:	finishIndex=getIndexFIFO();
                                    break;
                                    case	XBEE_LIFO:	finishIndex=getIndexLIFO();
                                    break;
                                    case	XBEE_OUT:	freeIndexMatrix(temp);
                                    return -1;
                                    break;
                                }
                            }
                            packet_finished[finishIndex] = (packetXBee*) calloc(1,sizeof(packetXBee));
                            if(packet_finished[finishIndex]==NULL){//printf("NU");
                                freeIndexMatrix(temp);
                                return -1;
                            }
                            else pendingPackets--; //printf("pens=%x ",pendingPackets);
                            packet_finished[finishIndex]->time=pendingFragments[temp]->time;
                            packet_finished[finishIndex]->packetID=pendingFragments[temp]->packetID;
                            packet_finished[finishIndex]->address_typeS=pendingFragments[temp]->address_typeS;
                            packet_finished[finishIndex]->mode=pendingFragments[temp]->mode;
                            if(protocol==XBEE_802_15_4)
                            {
                                if(packet_finished[finishIndex]->address_typeS==_64B)
                                {
                                    for(it=0;it<4;it++)
                                    {
                                        packet_finished[finishIndex]->macSH[it]=pendingFragments[temp]->macSH[it];
                                    }
                                    for(it=0;it<4;it++)
                                    {
                                        packet_finished[finishIndex]->macSL[it]=pendingFragments[temp]->macSL[it];
                                    }
                                }
                                else if(packet_finished[finishIndex]->address_typeS==_16B)
                                {
                                    packet_finished[finishIndex]->naS[0]=pendingFragments[temp]->naS[0];
                                    packet_finished[finishIndex]->naS[1]=pendingFragments[temp]->naS[1];
                                }
                            }
                            else
                            {
                                for(it=0;it<4;it++)
                                {
                                    packet_finished[finishIndex]->macSH[it]=pendingFragments[temp]->macSH[it];
                                }
                                for(it=0;it<4;it++)
                                {
                                    packet_finished[finishIndex]->macSL[it]=pendingFragments[temp]->macSL[it];
                                }
                                packet_finished[finishIndex]->naS[0]=pendingFragments[temp]->naS[0];
                                packet_finished[finishIndex]->naS[1]=pendingFragments[temp]->naS[1];
                            }
                            aux=(pendingFragments[temp]->RSSI)/(pendingFragments[temp]->totalFragments);
                            packet_finished[finishIndex]->RSSI=aux;//printf("RS=%x ",aux);
                            packet_finished[finishIndex]->typeSourceID=pendingFragments[temp]->typeSourceID;
                            switch(packet_finished[finishIndex]->typeSourceID)
                            {
                                case MY_TYPE:	packet_finished[finishIndex]->naO[0]=pendingFragments[temp]->naO[0];
                                packet_finished[finishIndex]->naO[1]=pendingFragments[temp]->naO[1];//printf("na0=%x,na1=%x ",pendingFragments[temp]->naO[0],pendingFragments[temp]->naO[1]);
                                break;
                                case MAC_TYPE:	for(it=0;it<4;it++)
                                {
                                    packet_finished[finishIndex]->macOH[it]=pendingFragments[temp]->macOH[it];//printf("mH%x",pendingFragments[temp]->macOH[it]);
                                }
                                for(it=0;it<4;it++)
                                {
                                    packet_finished[finishIndex]->macOL[it]=pendingFragments[temp]->macOL[it];//printf("mL%x",pendingFragments[temp]->macOL[it]);
                                }
                                break;
                                case NI_TYPE:	it=0;
                                do
                                {
                                    packet_finished[finishIndex]->niO[it]=pendingFragments[temp]->niO[it];//printf("ni%d=%x ",it,pendingFragments[temp]->niO[it]);
                                    it++;
                                } while(pendingFragments[temp]->niO[it]!=35);
                                packet_finished[finishIndex]->niO[it]=char(35);
                                break;
                            }
                            temp2=(pendingFragments[temp]->totalFragments)-1;//printf("tmp=%x ",temp2);
                            aux=0;
                            temp3=0;
                            while(temp2>=0)
                            {
                                for(it=0;it<packet_fragments[temp][temp2]->frag_length;it++)
                                {
                                    packet_finished[finishIndex]->data[it+temp3]=packet_fragments[temp][temp2]->data[it];
                                    aux++;
                                }
                                temp3=packet_fragments[temp][temp2]->frag_length+temp3;
                                temp2--;
                            }
                            packet_finished[finishIndex]->data_length=aux;//printf("len=%d ",aux);
//                            if(mode==CLUSTER)
//                           {	//printf("CLS");
//                                packet_finished[finishIndex]->SD=pendingFragments[temp]->SD;
//                                packet_finished[finishIndex]->DE=pendingFragments[temp]->DE;
//                                packet_finished[finishIndex]->CID[0]=pendingFragments[temp]->CID[0];
//                                packet_finished[finishIndex]->CID[1]=pendingFragments[temp]->CID[1];
//                                packet_finished[finishIndex]->PID[0]=pendingFragments[temp]->PID[0];
//                                packet_finished[finishIndex]->PID[1]=pendingFragments[temp]->PID[1];
//                            }
						// Vaciamos el index
                            for(it=0;it<MAX_FRAG_PACKETS;it++)
                            {
                                free(packet_fragments[temp][it]);
                                packet_fragments[temp][it]=NULL;
                            }
						
                            pendingFragments[temp]->time=0;
                            free(pendingFragments[temp]);
                            pendingFragments[temp]=NULL;
                        }
                        else if(pendingFragments[temp]->totalFragments==pendingFragments[temp]->recFragments)
                        {	//printf("[]ttl");
                            for(it=0;it<MAX_FRAG_PACKETS;it++)
                            {
							
                                free(packet_fragments[temp][it]);
                                packet_fragments[temp][it]=NULL;
                            }
						
                            pendingFragments[temp]->time=0;
                            free(pendingFragments[temp]);
                            pendingFragments[temp]=NULL;
                            pendingPackets--;
                            nextIndex1=temp;
                            indexNotModified=0;
                        }
                    }
                }
                else // se borra la fila de la matriz y el correspondiente al vector indexador
                {	//printf("else");
                    for(it=0;it<MAX_FRAG_PACKETS;it++)
                    {
						
                        free(packet_fragments[temp][it]);
                        packet_fragments[temp][it]=NULL;
                    }
                    pendingFragments[temp]->time=0;
                    free(pendingFragments[temp]);
                    pendingFragments[temp]=NULL;
                    pendingPackets--;
                    nextIndex1=temp;
                    indexNotModified=0;
                }
            }
        }
        temp++;
    }  
    temp=0; 
    return 0;
}
#endif

/*
 Function: Generates the API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 	param : The param to set
 Returns: Nothing
 Values: Stores in 'command' variable the API frame to send to the XBee module
*/
void WaspXBeeCore::gen_data(const char* data, uint8_t param)
{
    uint8_t inc=0;
    uint8_t inc2=0;
	
    clearCommand();
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    while(inc2<inc){
        command[inc2]=Utils.converter(data[2*inc2],data[2*inc2+1]);
        inc2++;
    }
	
    command[inc-2]=param;
}


/*
 Function: Generates the API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 Returns: Nothing
 Values: Stores in 'command' variable the API frame to send to the XBee module
*/
void WaspXBeeCore::gen_data(const char* data)
{
    uint8_t inc=0;
    uint8_t inc2=0;
	
    clearCommand();
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    while(inc2<inc){
        command[inc2]=Utils.converter(data[2*inc2],data[2*inc2+1]);
        inc2++;
    }
}


/*
 Function: Generates the API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 	param1 : The param to set
 	param2 : The param to set
 Returns: Nothing
 Values: Stores in 'command' variable the API frame to send to the XBee module
*/
void WaspXBeeCore::gen_data(const char* data, uint8_t param1, uint8_t param2)
{
    uint8_t inc=0;
    uint8_t inc2=0;
	
    clearCommand();
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    while(inc2<inc){
        command[inc2]=Utils.converter(data[2*inc2],data[2*inc2+1]);
        inc2++;
    }
	
    command[inc-3]=param1;
    command[inc-2]=param2;
}


/*
 Function: Generates the API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 	param : The param to set
 Returns: Nothing
 Values: Stores in 'command' variable the API frame to send to the XBee module
*/
void WaspXBeeCore::gen_data(const char* data, uint8_t* param)
{
    uint8_t inc=0;
    uint8_t inc2=0;
		
    clearCommand();
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    while(inc2<inc){
        command[inc2]=Utils.converter(data[2*inc2],data[2*inc2+1]);
        inc2++;
    }
	
    if(inc==24) 
    {
        for(it=0;it<16;it++)
        {
            command[inc-17+it]=param[it];
        }
    }
    else if(inc==16) 
    {
        for(it=0;it<8;it++)
        {
            command[inc-9+it]=param[it];
        }
    }
    else if(inc==11)
    {
        for(it=0;it<3;it++)
        {
            command[inc-4+it]=param[it];
        }
    }
    else if(inc==10)
    {
        for(it=0;it<2;it++)
        {
            command[inc-3+it]=param[it];
        }
    }
    else command[inc-2]=param[0];
}


/*
 Function: Generates the API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 	param : The param to set
 Returns: Nothing
 Values: Stores in 'command' variable the API frame to send to the XBee module
*/
void WaspXBeeCore::gen_data(const char* data, const char* param)
{
    gen_data(data,(uint8_t*) param);
}


/*
 Function: Generates the checksum API frame to send to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 Returns: Nothing
 Values: Stores in 'command' variable the checksum API frame to send to the XBee module
*/
uint8_t WaspXBeeCore::gen_checksum(const char* data)
{
    uint8_t inc=0;
    uint8_t checksum=0;
	
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    for(it=3;it<inc;it++)
    {
        checksum=checksum+command[it];
    }
    while( (checksum>255))
    {
        checksum=checksum-256;
    }
    checksum=255-checksum;
    command[inc-1]=checksum;
	
    return checksum;
}


/*
 Function: Sends the API frame stored in 'command' variable to the XBee module
 Parameters:
 	data : The string that contains part of the API frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::gen_send(const char* data)
{
    uint8_t inc=0;
    uint8_t inc2=0;
    int8_t error_int=2;
	
    it=0;
    while(data[it] != '\0') {
        inc++;
        it++;
    }
    inc/=2;
	
    while(inc2<inc)
    {
        //XBee.print((char)command[inc2]); 
		printByte((char)command[inc2],  XBEE_UART);
        inc2++;
    }
    inc2=0;
	
    error_int=parse_message(command);
	



    return error_int;
}


/*
 Function: Generates the API frame when a TX is done
 Parameters:
 	_packet : the packetXBee structure where the data to send is stored
 	TX_array : the array where the API frame is stored
 	start_pos : starting position in API frame
 Returns: Nothing
*/
void WaspXBeeCore::gen_frame(packetXBee* _packet, uint8_t* TX_array, uint8_t start_pos)
{
    uint16_t counter1=0;
	
    if(_packet->endFragment==1)
    {
        TX_array[start_pos]=0x23;
        switch(_packet->typeSourceID)
        {
            case MY_TYPE:  	TX_array[start_pos+1]=_packet->typeSourceID;
            TX_array[start_pos+2]=_packet->naO[0];
            TX_array[start_pos+3]=_packet->naO[1];
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+4]=uint8_t(_packet->data[it+start]);
            }
            break;
            case MAC_TYPE: 	TX_array[start_pos+1]=_packet->typeSourceID;
            for(it=0;it<4;it++)
            {
                TX_array[start_pos+2+it]=_packet->macOH[it];
            }
            for(it=0;it<4;it++)
            {
                TX_array[start_pos+6+it]=_packet->macOL[it];
            }
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+10]=uint8_t(_packet->data[it+start]);
            }
            break;
            case NI_TYPE:   TX_array[start_pos+1]=_packet->typeSourceID;
            while(_packet->niO[it]!='#'){
                counter1++;
                it++;
            }
            counter1++;
            for(it=0;it<counter1;it++)
            {
                TX_array[start_pos+2+it]=uint8_t(_packet->niO[it]);
            }
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+2+counter1]=uint8_t(_packet->data[it+start]);
            }
            break;
            default:    	break;
        }
    }
    else
    {
        switch(_packet->typeSourceID)
        {
            case MY_TYPE: 	TX_array[start_pos]=_packet->typeSourceID;
            TX_array[start_pos+1]=_packet->naO[0];
            TX_array[start_pos+2]=_packet->naO[1];
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+3]=uint8_t(_packet->data[it+start]);
            }
            break;
            case MAC_TYPE:  TX_array[start_pos]=_packet->typeSourceID;
            for(it=0;it<4;it++)
            {
                TX_array[start_pos+1+it]=_packet->macOH[it];
            }
            for(it=0;it<4;it++)
            {
                TX_array[start_pos+5+it]=_packet->macOL[it];
            }
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+9]=uint8_t(_packet->data[it+start]);
            }
            break;
            case NI_TYPE:   TX_array[start_pos]=_packet->typeSourceID;
            while(_packet->niO[it]!='#'){
                counter1++;
                it++;
            }
            counter1++;
            for(it=0;it<counter1;it++)
            {
                TX_array[start_pos+1+it]=uint8_t(_packet->niO[it]);
            }
            for(it=0;it<(finish-start+1);it++) // data
            {
                TX_array[it+start_pos+1+counter1]=uint8_t(_packet->data[it+start]);
            }
            break;
            default:   	break;
        }
    }
}


/*
 Function: Generates the eschaped API frame when a TX is done
 Parameters:
 	_packet : the packetXBee structure where the data to send is stored
 	TX_array : the array where the API frame is stored
 	protect : specifies the number of chars that had been eschaped
 	type : specifies the type of send
 Returns: Nothing
*/
void WaspXBeeCore::gen_frame_ap2(packetXBee* _packet, uint8_t* TX_array, uint8_t &protect, uint8_t type)
{
    uint8_t a=1;
    uint8_t final=0;
    uint8_t unico=0;
    uint16_t aux=0;
    uint16_t aux2=0;
	
    while(a<(_packet->frag_length+type+protect))
    {
        if( (TX_array[a]==17) && (unico==0) )
        {
            TX_array[a]=49;
            protect++;
            aux=TX_array[a];
            TX_array[a]=125;
            uint16_t l=a-1;
            while(final==0)
            {
                aux2=TX_array[l+2];
                TX_array[l+2]=aux;
                if( ((l+3)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                aux=TX_array[l+3];
                TX_array[l+3]=aux2;
                if( ((l+4)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                l++;
                l++;
            }
            final=0;
            unico=1;
        }
        if( (TX_array[a]==19) && (unico==0) )
        {
            TX_array[a]=51;
            protect++;
            aux=TX_array[a];
            TX_array[a]=125;
            uint16_t l=a-1;
            while(final==0)
            {
                aux2=TX_array[l+2];
                TX_array[l+2]=aux;
                if( ((l+3)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                aux=TX_array[l+3];
                TX_array[l+3]=aux2;
                if( ((l+4)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                l++;
                l++;
            }
            final=0;  
            unico=1;      
        }
        if( (TX_array[a]==126) && (unico==0) )
        {
            TX_array[a]=94;
            protect++;
            aux=TX_array[a];
            TX_array[a]=125;
            uint16_t l=a-1;
            while(final==0)
            {
                aux2=TX_array[l+2];
                TX_array[l+2]=aux;
                if( ((l+3)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                aux=TX_array[l+3];
                TX_array[l+3]=aux2;
                if( ((l+4)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                l++;
                l++;
            }
            final=0;
            unico=1;      
        }
        if( (TX_array[a]==125) && (unico==0) )
        {
            TX_array[a]=93;
            protect++;
            aux=TX_array[a];
            TX_array[a]=125;
            uint16_t l=a-1;
            while(final==0)
            {
                aux2=TX_array[l+2];
                TX_array[l+2]=aux;
                if( ((l+3)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                aux=TX_array[l+3];
                TX_array[l+3]=aux2;
                if( ((l+4)>=(_packet->frag_length+type+protect)) )
                {
                    final=1;
                    break;
                }
                l++;
                l++;
            }
            final=0;  
            unico=1;      
        }
        a++;
        unico=0;
    }
}


/*
 Function: Parses the answer received by the XBee module, calling the appropriate function
 Parameters:
 	frame : an array that contains the API frame that is expected to receive answer from if it is an AT command
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
extern	int rx_buffer_head3;
extern	int rx_buffer_tail3;
int rx_buffer_head3Last=0;
int rx_buffer_head3Last200=0;
int rx_buffer_headi=0;
extern	unsigned char rx_buffer3[RX_BUFFER_SIZE_3];
int8_t WaspXBeeCore::parse_message(uint8_t* frame)
{


    uint16_t i=0;
    long previous=millis(); /**/
    long previous2=millis();/**/
    uint8_t num_mes=0;
    uint8_t num_esc=0;
    uint16_t num_data=0;
    uint16_t length_mes=0;
    uint16_t length_prev=0;
    int8_t error=2;
    long interval=50;
    long intervalMAX=40000;
    uint8_t good_frame=0;
	uint8_t good_rx_frame=0;
    uint8_t maxFrame=30;
	uint8_t	flagerrreason=0;

					//serialWrite('e', 2); serialWrite('5', 2);

	// If a frame was truncated before, we set the first byte
    if( frameNext ){
        frameNext=0;
        MemoryArray[0]=0x7E;
        i=1;
        num_mes=1;
		printf("next");
    }
	
	// If it is a TX we have a different behaviour
//    if( frame[0]==0xFF )
//	{   //printf("frm=ff");
//        error_TX=txStatusResponse(MemoryArray);
 //       return error_TX; 
//    }
//    else if( frame[0]==0xFE )
//	{   //printf("frm=fe");
//        error_TX=txZBStatusResponse(MemoryArray);
//        return error_TX; 
//    }
	
	// If a RX we reduce the interval
    if( frame[0]==0xEE ){
        interval=5;
        maxFrame=109;
    }
	
	// Check if a ED is performed
    if( frame[5]==0x45 && frame[6]==0x44 && protocol==XBEE_802_15_4 ) interval=3000;
	
	// Check if a DN is performed
    if( frame[5]==0x44 && frame[6]==0x4E ) interval=1000;
		
	// Check if a ND is performed
    if( frame[5]==0x4E && frame[6]==0x44 ){
        interval=20000;
        if(protocol==DIGIMESH) interval=40000;
        else if( (protocol==XBEE_900) || (protocol==XBEE_868) )
        {
            interval=14000;
        }
    }
	
	// Read data from XBee meanwhile data is available
    previous2=millis();
    previous=millis();/**/


   // while( ((millis()-previous)<interval) && ((millis()-previous2)<intervalMAX) && i<MAX_PARSE && !frameNext )
    while(1)
    {		
		flagerrreason=0;
		if(!((millis()-previous)<interval))flagerrreason |=0x01;
		if(!((millis()-previous2)<intervalMAX))flagerrreason |=0x02;
		if(!(i<MAX_PARSE))flagerrreason |=0x04;
		if(frameNext)flagerrreason |=0x08;
		if(flagerrreason>0)
		{
			printf("flag=%2x ",flagerrreason);
			if(flagerrreason &0x01)
			{
				printf("p=%x m=%x ",previous,millis());
				printf(" head=%d last=%d ",rx_buffer_head3,rx_buffer_head3Last);
				rx_buffer_head3Last=rx_buffer_head3;
			}
			if(flagerrreason &0x02)printf("p2=%x m=%x ",previous2,millis());
			if(flagerrreason &0x04)printf("i%d ",i);
			break;
		}
        //if(XBee.available())
		if(serialAvailable( XBEE_UART))
        {
			MemoryArray[i]=serialRead(XBEE_UART);
            i++;
            if(MemoryArray[i-1]==0x7E){
                if( (MAX_PARSE-i) < maxFrame ){ frameNext=1;   printf("fnx");break;}
                else{ num_mes++; 
				//printf("n%d ",num_mes);
				}
            }
            previous=millis();
        }
	if( (long)millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
        if( (long)millis()-previous2 < 0 ) previous2=millis(); //avoid millis overflow problem
    } /**/
		
//	printf("i=%d tail=%d ",i,rx_buffer_tail3);
    num_data=i;
    i=1;

	// If some corrupted frame has appeared we jump it
    if( MemoryArray[0]!=0x7E ) num_mes++;
	
	//printf("mes%d ",num_mes);
	// Parse the received messages from the XBee
    while( num_mes>0 )
    {
        while( MemoryArray[i]!=0x7E && i<num_data ) i++;
        length_mes=i-length_prev;
		
		// If some char has been eschaped, it must be converted before parsing it
        for( it=0;it<length_mes;it++ )
        {
            if( MemoryArray[it+length_prev]==0x7D ) num_esc++;
        }
        if( num_esc ) des_esc(MemoryArray,length_mes,i-length_mes);
		//printf(" sw=%02x ",MemoryArray[(i-length_mes)+3]);	printf("i%dLm%d ",i,length_mes);
        switch( MemoryArray[(i-length_mes)+3] )
        {
            case 0x88 :	
//			if((MemoryArray[(i-length_mes)+5]=='M'))
//			{
//				serialWrite('e', 2); serialWrite('e', 2);
//			}
//			if(frame[3]==0x08)
//			{			
				error=atCommandResponse(MemoryArray,frame,length_mes-num_esc+length_prev,i-length_mes);
	            error_AT=error;
//			}
//			if((MemoryArray[(i-length_mes)+5]=='M')&&(MemoryArray[(i-length_mes)+6]=='Y'))
//			{
//				serialWrite('e', 2); serialWrite('=', 2);
//				if((error_AT>=0)&&(error_AT<=9)) serialWrite(error_AT+'0', 2);
//				else  serialWrite(' ', 2);
//				serialWrite(' ', 2);
//			}
            break;
            case 0x8A :	error=modemStatusResponse(MemoryArray,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x80 :	
//			serialWrite('e', 2); serialWrite('=', 2);
//			if(frame[3]==0xEE)
//			{
			//printf("start:%d  end:%d ",i-length_mes,length_mes-num_esc+length_prev);
			error=rxData(MemoryArray,length_mes-num_esc+length_prev,i-length_mes);
            error_RX=error;
			if(error_RX==0) good_rx_frame = 1;
//			}
//			if((error_RX>=0)&&(error_RX<=9)) serialWrite(error_RX+'0', 2);
//			else  serialWrite(' ', 2);
//			serialWrite(' ', 2);

            break;
            case 0x81 :	error=rxData(MemoryArray,length_mes-num_esc+length_prev,i-length_mes);
            error_RX=error;
            break;
            case 0x90 :	error=rxData(MemoryArray,length_mes-num_esc+length_prev,i-length_mes);
            error_RX=error;
            break;
            case 0x91 :	error=rxData(MemoryArray,length_mes-num_esc+length_prev,i-length_mes);
            error_RX=error;
            break;
            default   :	break;
        }
		
        num_mes--;
        length_prev=i;
        i++;
        num_esc=0;
        if(!error) good_frame++;
    }
		
    if(good_frame) 
    {
		if(good_rx_frame ==1)
		{
			error_RX = 0;
		}
		return 0;
    }
    else return error;
}


/*
 Function: Generates the correct API frame from an eschaped one
 Parameters:
 	data_in : The string that contains the eschaped API frame
 	end : the end of the frame
 	start : the start of the frame
 Returns: Nothing
*/
void WaspXBeeCore::des_esc(uint8_t* data_in, uint16_t end, uint16_t start)
{
    uint16_t i=0;
    uint16_t aux=0;
		
    while( i<end )
    {
        while( data_in[start+i]!=0x7D && i<end ) i++;
        if( i<end )
        {
            aux=i+1;
            switch( data_in[start+i+1] )
            {
                case 0x31 : 	data_in[start+i]=0x11;
                break;
                case 0x33 : 	data_in[start+i]=0x13;
                break;
                case 0x5E : 	data_in[start+i]=0x7E;
                break;
                case 0x5D : 	data_in[start+i]=0x7D;
                break;
            }
            i++;
            end--;
            while( i<(end) ){
                data_in[start+i]=data_in[start+i+1];
                i++;
            }
            i=aux;
        }
    }
}


/*
 Function: Parses the AT command answer received by the XBee module
 Parameters:
 	data_in : the answer received by the module
 	frame : an array that contains the API frame that is expected to receive answer from if it is an AT command
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::atCommandResponse(uint8_t* data_in, uint8_t* frame, uint16_t end, uint16_t start)
{	
	// Check the checksum
    if(checkChecksum(data_in,end,start)) return 1;
		
	// Check the AT Command Response is from the command expected
    if( data_in[start+5]!=frame[5] || data_in[start+6]!=frame[6] ) return 1;
		
	// Check if there is data in the AT Command Response frame
    if( (end-start)==9 ){
        if( data_in[start+7]==0 ) return 0;
        else return 1;
    }
		
    if( data_in[start+7]!=0 ) return 1;
	// Store the data in the response frame
    for(it=0;it<(end-start-9);it++)
    {
        data[it]=data_in[8+it+start];
    }
	
	// Check if a ND is performed
    data_length=end-start-9;
    if( frame[5]==0x4E && frame[6]==0x44 ){
        if( data_length>1 ) totalScannedBrothers++;
        treatScan();
    }
    return 0;
}


/*
 Function: Parses the Modem Status message received by the XBee module
 Parameters:
 	data_in : the answer received by the module
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::modemStatusResponse(uint8_t* data_in, uint16_t end, uint16_t start)
{		
	// Check the checksum
    if(checkChecksum(data_in,end,start)) return 1;	
		
    modem_status=data_in[start+4];
    return 0;
}


/*
 Function: Parses the TX Status message received by the XBee module
 Parameters:
 	data_in : the answer received by the module
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::txStatusResponse()
{
	uint8_t ByteIN[MAX_PARSE];
    long previous=millis();/**/
    
    uint16_t numberBytes=7;
    uint8_t end=0;
    uint16_t counter3=0;
    uint8_t undesired=0;
    uint8_t status=0;
    uint16_t num_TX=0;
    uint8_t num_esc=0;
    int16_t interval=2000;
    uint8_t num_mes=0;
    uint16_t i=1;
    uint16_t length_mes=0;
    uint16_t length_prev=0;
    uint8_t maxFrame=110;
	
    error_TX=2;
		
    if( frameNext )
    {
        ByteIN[0]=0x7E;
        counter3=1;
        num_mes=1;
        frameNext=0;
    }
	
    while( end==0 && !frameNext )
    {
        //if(XBee.available()>0)
		if(serialAvailable(XBEE_UART)>0)
        {
            //ByteIN[counter3]=XBee.read();
			ByteIN[counter3]=serialRead(XBEE_UART);
            counter3++;
            previous=millis();/**/
            if(ByteIN[counter3-1]==0x7E){
                if( (MAX_PARSE-counter3) < maxFrame ) frameNext=1;
                else num_mes++;
            }
            if( (counter3==1) && (ByteIN[counter3-1]!=0x7E) ) counter3=0;	
            if( counter3>=MAX_PARSE ) end=1;
            if( (counter3==4+status*6+undesired) && (undesired!=1) ) //FIXME
            {
                if( (ByteIN[counter3-1]!= 0x89) && (ByteIN[counter3-1]!=0x8A) ){
                    undesired=1;
                    numberBytes+=3;
                }
            }
            if( undesired==1 ) numberBytes++;
            if( (ByteIN[counter3-1]==0x7D) && (!undesired) )
            {
                numberBytes++;
            }
            if( (ByteIN[counter3-1]==0x8A) && (counter3==(4+status*6)) )
            {
                numberBytes+=6;
                status++;
            }
            if( (ByteIN[counter3-1]==0x7E) && (undesired==1) )
            {
                numberBytes--;
                undesired=numberBytes-7;
            }
            if(counter3==numberBytes)
            {
                end=1;
            }
        }
        if( (long)millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
        if( ((long)millis()-previous) > interval )
        {
            end=1;	 printf("txStatus flush");
            //XBee.flush();
			serialFlush(XBEE_UART);
        } /**/
    }
    num_TX=counter3;
    counter3=0;
	
	// If some corrupted frame has appeared we jump it
    if( ByteIN[0]!=0x7E ) num_mes++;
	
	// Parse the received messages from the XBee
    while( num_mes>0 )
    {
        while( ByteIN[i]!=0x7E && i<num_TX ) i++;
        length_mes=i-length_prev;
		
		// If some char has been eschaped, it must be converted before parsing it
        for( it=0;it<length_mes;it++)
        {
            if( ByteIN[it+length_prev]==0x7D ) num_esc++;
        }
        if( num_esc ) des_esc(ByteIN,length_mes,i-length_mes);
		
        switch( ByteIN[(i-length_mes)+3] )
        {
            case 0x8A :	modemStatusResponse(ByteIN,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x80 :
            case 0x81 :	error_RX=rxData(ByteIN,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x89 :	delivery_status=ByteIN[i-length_mes+5];
            if( delivery_status==0 ) error_TX=0;
            else error_TX=1;
            break;
            default   :	break;
        }
		
        num_mes--;
        length_prev=i;
        i++;
        num_esc=0;
    }
	
    return error_TX;
}

/*
 Function: Parses the ZB TX Status message received by the XBee module
 Parameters:
 	data_in : the answer received by the module
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::txZBStatusResponse()
{	
	uint8_t ByteIN[MAX_PARSE];
    long previous=millis();
    uint16_t numberBytes=11;
    uint8_t end=0;
    uint16_t counter3=0;
    uint8_t undesired=0;
    uint8_t status=0;
    uint16_t num_TX=0;
    uint8_t num_esc=0;
    int16_t interval=2000;
    uint8_t num_mes=0;
    uint16_t i=1;
    uint16_t length_mes=0;
    uint16_t length_prev=0;
    uint8_t maxFrame=110;
	
    error_TX=2;
		
    if( frameNext )
    {
        ByteIN[0]=0x7E;
        counter3=1;
        num_mes=1;
        frameNext=0;
    }
	
    while( end==0 && !frameNext )
    {
        //if(XBee.available()>0)
		if(serialAvailable(XBEE_UART)>0)
        {
            //ByteIN[counter3]=XBee.read();
			ByteIN[counter3]=serialRead(XBEE_UART);
            counter3++;
            previous=millis();
            if(ByteIN[counter3-1]==0x7E){
                if( (MAX_PARSE-counter3) < maxFrame ) frameNext=1;
                else num_mes++;
            }
            if( (counter3==1) && (ByteIN[counter3-1]!=0x7E) ) counter3=0;	
            if( counter3>=MAX_PARSE ) end=1;
            if( (counter3==4+status*6+undesired) && (undesired!=1) ) //FIXME
            {
                if( (ByteIN[counter3-1]!= 0x8B) && (ByteIN[counter3-1]!=0x8A) ){
                    undesired=1;
                    numberBytes+=3;
                }
            }
            if( undesired==1 ) numberBytes++;
            if( (ByteIN[counter3-1]==0x7D) && (!undesired) )
            {
                numberBytes++;
            }
            if( (ByteIN[counter3-1]==0x8A) && (counter3==(4+status*6)) )
            {
                numberBytes+=6;
                status++;
            }
            if( (ByteIN[counter3-1]==0x7E) && (undesired==1) )
            {
                numberBytes--;
                undesired=numberBytes-7;
            }
            if(counter3==numberBytes)
            {
                end=1;
            }
        }
	if( (long)millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
        if( ((long)millis()-previous) > interval )
        {
            end=1; printf("txZBStatus flush");
            //XBee.flush();
			serialFlush(XBEE_UART);
        } 
    }
    num_TX=counter3;
    counter3=0;
	
	// If some corrupted frame has appeared we jump it
    if( ByteIN[0]!=0x7E ) num_mes++;
	
	// Parse the received messages from the XBee
    while( num_mes>0 )
    {
        while( ByteIN[i]!=0x7E && i<num_TX ) i++;
        length_mes=i-length_prev;
		
		// If some char has been eschaped, it must be converted before parsing it
        for( it=0;it<length_mes;it++)
        {
            if( ByteIN[it+length_prev]==0x7D ) num_esc++;
        }
        if( num_esc ) des_esc(ByteIN,length_mes,i-length_mes);
		
        switch( ByteIN[(i-length_mes)+3] )
        {
            case 0x8A :	modemStatusResponse(ByteIN,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x90 :	error_RX=rxData(ByteIN,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x91 :	error_RX=rxData(ByteIN,length_mes-num_esc+length_prev,i-length_mes);
            break;
            case 0x8B :	true_naD[0]=ByteIN[i-length_mes+5];
            true_naD[1]=ByteIN[i-length_mes+6];
            retries_sending=ByteIN[i-length_mes+7];
            discovery_status=ByteIN[i-length_mes+9];
            delivery_status=ByteIN[i-length_mes+8];
            if( delivery_status==0 ) error_TX=0;
            else error_TX=1;
            break;
            default   :	break;
        }
		
        num_mes--;
        length_prev=i;
        i++;
        num_esc=0;
    }

    return error_TX;
}

/*
 Function: Parses the RX Data message received by the XBee module
 Parameters:
 	data_in : the answer received by the module
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
int8_t WaspXBeeCore::rxData(uint8_t* data_in, uint16_t end, uint16_t start)
{
    int8_t error=2;
//	printf("rx");	
	// Check the checksum
    if(checkChecksum(data_in,end,start)){
			printf(" rxData flush ");
		serialFlush(XBEE_UART);
        return 1;
    }	
	
	// Copy the data
    data_length=0;
    for(it=4+start;it<end-1;it++)
    {
        ByteINArray[it-4-start]=data_in[it];
        data_length++;
    }
		
    switch( data_in[start+3] )
    {
        case 0x80 :	add_type=_64B;
        break;
        case 0x81 :	add_type=_16B;
        break;
        case 0x90 :
			add_type=NORMAL_RX;
			//mode=UNICAST;
        break;
        case 0x91 :
			//mode=CLUSTER;
			add_type=EXPLICIT_RX;
        break;
    }

    error=readXBee(ByteINArray);

	
    return error;
}

/*
 Function: Parses the ND message received by the XBee module
 Values: Stores in 'scannedBrothers' variable the data extracted from the answer
*/
void WaspXBeeCore::treatScan()
{
    uint8_t cont2=0;
    uint8_t length_NI=data_length-19;
		
    if(protocol==XBEE_802_15_4)
    {
        cont2=totalScannedBrothers-1;
        for(it=0;it<2;it++)
        {
            scannedBrothers[cont2].MY[it]=data[it];
        }
        for(it=0;it<4;it++)
        {
            scannedBrothers[cont2].SH[it]=data[it+2];
        }
        for(it=0;it<4;it++)
        {
            scannedBrothers[cont2].SL[it]=data[it+6];
        }
        scannedBrothers[cont2].RSSI=data[10];
        if (data_length>12)
        {
            for(it=0;it<(data_length-12);it++)
            {
                scannedBrothers[cont2].NI[it]=char(data[it+11]);
            }
        }
    }
    if( (protocol==ZIGBEE) || (protocol==DIGIMESH) || (protocol==XBEE_900) || (protocol==XBEE_868) )
    {
        cont2=totalScannedBrothers-1;
        for(it=0;it<2;it++)
        {
            scannedBrothers[cont2].MY[it]=data[it];
        }
        for(it=0;it<4;it++)
        {
            scannedBrothers[cont2].SH[it]=data[it+2];
        }
        for(it=0;it<4;it++)
        {
            scannedBrothers[cont2].SL[it]=data[it+6];
        }
        for(it=0;it<length_NI;it++)
        {
            scannedBrothers[cont2].NI[it]=char(data[it+10]);
        }
        for(it=0;it<2;it++)
        {
            scannedBrothers[cont2].PMY[it]=data[it+length_NI+11];
        }
        scannedBrothers[cont2].DT=data[length_NI+13];
        scannedBrothers[cont2].ST=data[length_NI+14];
        for(it=0;it<2;it++)
        {
            scannedBrothers[cont2].PID[it]=data[it+length_NI+15];
        }
        for(it=0;it<2;it++)
        {
            scannedBrothers[cont2].MID[it]=data[it+length_NI+17];
        }
    }
}

/*
 * Function: Calculates the checksum for a TX frame
 * Parameters:
 * 	TX : pointer to the frame whose checksum has to be calculated
 * Returns: the calculated checksum for the frame
*/
uint8_t WaspXBeeCore::getChecksum(uint8_t* TX)
{	
	uint8_t checksum=0;
	
    // calculate checksum
    for( int i=3 ; i < (TX[2]+3);i++) 
    {
		checksum=checksum+TX[i];
	}
	
	while( checksum > 255 )
	{
		checksum = checksum - 256;
	}
	checksum = 255 - checksum;
	
	return checksum;
	
}


/*
 Function: Checks the checksum is good
 Parameters:
 	data_in : the answer received by the module
 	end : the end of the frame
 	start : the start of the frame
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
*/
uint8_t WaspXBeeCore::checkChecksum(uint8_t* data_in, uint16_t end, uint16_t start)
{	
    uint16_t checksum=0;	
		
    for(it=3+start;it<end;it++)
    {
        checksum=checksum+data_in[it];
    }
    if( (checksum==255) ) return 0;
    checksum%=256;
    if( checksum!=255 ) return 1;
    return 0;
}

/*
 Function: Clears the variable 'command'
*/
void WaspXBeeCore::clearCommand()
{
    for(it=0;it<30;it++)
    {
        command[it]=0;
    }
}


/*
 Function: It frees a position in index array
*/
void WaspXBeeCore::freeIndex()
{
    uint16_t counter1=0;
	
    nextIndex1=0;
    while( counter1<MAX_FINISH_PACKETS )
    {
        for(it=counter1;it<(MAX_FINISH_PACKETS-1);it++)
        {
            if( pendingFragments[counter1]->time < pendingFragments[it+1]->time ) nextIndex1++;
            else break;
        }
        if( nextIndex1==(MAX_FINISH_PACKETS-1) ){
            nextIndex1=counter1;
            counter1=MAX_FINISH_PACKETS;
        }
        else nextIndex1=counter1+1;
        counter1++;
    }
    for(it=0;it<MAX_FRAG_PACKETS;it++)
    {
        free(packet_fragments[nextIndex1][it]);
        packet_fragments[nextIndex1][it]=NULL;
    }
    pendingFragments[nextIndex1]->time=0;
    free(pendingFragments[nextIndex1]);
    pendingFragments[nextIndex1]=NULL;
    pendingPackets--;
    indexNotModified=0;
}

/*
 Function: It frees index array and matrix
*/
void WaspXBeeCore::freeAll()
{	
    uint8_t temp=0;
	
    for(temp=0;temp<MAX_FINISH_PACKETS;temp++)
    {
        for(it=0;it<MAX_FRAG_PACKETS;it++)
        {
							
            free(packet_fragments[temp][it]);
            packet_fragments[temp][it]=NULL;
        }
        pendingFragments[temp]->time=0;
        free(pendingFragments[temp]);
        pendingFragments[temp]=NULL;
    }
    pendingPackets=0;
    nextIndex1=0;
    indexNotModified=0;
}

/*
 Function: It gets the next index where store the finished packet
*/
uint8_t WaspXBeeCore::getFinishIndex()
{
    for(it=0;it<MAX_FINISH_PACKETS;it++)
    {
        if( packet_finished[it]==NULL ) break;
    }
    return it;
}

/*
 Function: It clears the finished packets array
*/
void WaspXBeeCore::clearFinishArray()
{
    for(it=0;it<MAX_FINISH_PACKETS;it++)
    {
        free(packet_finished[it]);
        packet_finished[it]=NULL;
    }
}

/*
 Function: It gets the index in 'packet_finished' where store the new packet, according to a FIFO policy
*/
uint8_t WaspXBeeCore::getIndexFIFO()
{
    uint8_t position=0;
    uint16_t counter1=0;
	
    while( counter1<MAX_FINISH_PACKETS )
    {
        for(it=counter1;it<(MAX_FINISH_PACKETS-1);it++)
        {
            if( packet_finished[counter1]->time < packet_finished[it+1]->time ) position++;
            else break;
        }
        if( position==(MAX_FINISH_PACKETS-1) ){
            position=counter1;
            counter1=MAX_FINISH_PACKETS;
        }
        else position=counter1+1;
        counter1++;
    }
    free(packet_finished[position]);
    packet_finished[position]=NULL;
    return position;
}

/*
 Function: It gets the index in 'packet_finished' where store the new packet, according to a LIFO policy
*/
uint8_t WaspXBeeCore::getIndexLIFO()
{
    uint8_t position=0;
    uint16_t counter1=0;
	
    while( counter1<MAX_FINISH_PACKETS )
    {
        for(it=counter1;it<(MAX_FINISH_PACKETS-1);it++)
        {
            if( packet_finished[counter1]->time > packet_finished[it+1]->time ) position++;
            else break;
        }
        if( position==(MAX_FINISH_PACKETS-1) ){
            position=counter1;
            counter1=MAX_FINISH_PACKETS;
        }
        else position=counter1+1;
        counter1++;
    }
    free(packet_finished[position]);
    packet_finished[position]=NULL;
    return position;
}

/*
 Function: It frees the index array and the matrix row corresponding to the position is sent as an input parameter
*/
void WaspXBeeCore::freeIndexMatrix(uint8_t position)
{		
    for(it=0;it<MAX_FRAG_PACKETS;it++)
    {
        free(packet_fragments[position][it]);
        packet_fragments[position][it]=NULL;
    }
    pendingFragments[position]->time=0;
    free(pendingFragments[position]);
    pendingFragments[position]=NULL;
    pendingPackets--;
}

/*
 Function: It receives the first packet of a new firmware
 Returns: Integer that determines if there has been any error 
   error=1 --> There has been an error while executing the function
   error=0 --> The function has been executed with no errors
*/
uint8_t WaspXBeeCore::new_firmware_received()
{


	char aux_array[15];
	bool startSequence = true;
	char* asteriscos;
	uint8_t channel_to_set = 0;
	bool error_sd = false;
	char sdfilename[20];//名字要带有.txt .hex .什么的
	
	
	// Check 'KEY_ACCESS'
	for (it = 0; it < 8;it++)
	{
		//if(packet_finished[pos-1]->data[it] != Utils.readEEPROM(it+107))
		if(packet_finished[pos-1]->data[it+6] != Utils.readEEPROM(it+107))
		{
		     startSequence = false;  
		     break;
	     }
	}
	
	firm_info.multi_type=3;
	
	if( startSequence && !firm_info.already_init )
	{
		asteriscos = (char*) calloc(449,sizeof(char));
		if( asteriscos == NULL ){
			return 1;
		}

		// Set OTA Flag and set last time a OTA packet was received
		programming_ON=1;		
		firm_info.time_arrived = millis();

		for(it=0;it<32;it++)
		{
			//firm_info.ID[it]=packet_finished[pos-1]->data[it+8];
			firm_info.ID[it]=packet_finished[pos-1]->data[it+8+6];
		}
		firm_info.ID[it]='\0';
		printf(" firm_info.ID=%s ",firm_info.ID);
		for(it=0;it<12;it++)
		{
			//firm_info.DATE[it]=packet_finished[pos-1]->data[it+40];
			firm_info.DATE[it]=packet_finished[pos-1]->data[it+40+6];
		}
		firm_info.DATE[it]='\0';
		printf(" firm_info.DATE=%s ",firm_info.DATE);
		
		// 802.15.4 Multicast or DigiMesh direct-access
		//if( packet_finished[pos-1]->data_length==53 && (protocol==XBEE_802_15_4 || protocol==DIGIMESH) )
		//{
		//	channel_to_set = packet_finished[pos-1]->data[52];
		//	getChannel();
		//	firm_info.channel=channel;
		//	if( !error_AT ) setChannel(channel_to_set);
		//	writeValues();
		//	firm_info.multi_type=0;
		//}
		if( packet_finished[pos-1]->data_length==(53+6) && (protocol==XBEE_802_15_4 || protocol==DIGIMESH) )
		{
			channel_to_set = packet_finished[pos-1]->data[52+6];
			getChannel();
			firm_info.channel=channel;
			if( !error_AT ) setChannel(channel_to_set);
			writeValues();
			firm_info.multi_type=0;
		}

		
		// DigiMesh or ZigBee Multicast
		if( packet_finished[pos-1]->data_length==60 && (protocol==ZIGBEE || protocol==DIGIMESH) )
		{
			// Copy 'Auth key'
			for (it = 0; it < 8;it++) firm_info.authkey[it] = Utils.readEEPROM(it+107);
			
			// Set new 'Auth key'
			for (it = 0; it < 8;it++) Utils.writeEEPROM(it+107, packet_finished[pos-1]->data[it+52]);
			
			firm_info.multi_type=1;
		}
		
		// 868 or 900 Multicast
		if( packet_finished[pos-1]->data_length==84 && (protocol==XBEE_868 || protocol==XBEE_900) )
		{
			// Copy 'Encryption key'
			for (it = 0; it < 16;it++) firm_info.encryptionkey[it] = packet_finished[pos-1]->data[it+52];
			
			char auxkey[16];
			for (it = 0; it < 16;it++) auxkey[it] = packet_finished[pos-1]->data[it+68];
			
			setLinkKey(auxkey);
			writeValues();
			
			firm_info.multi_type=2;
		}
		
		
		for(it=0;it<4;it++)
		{
			firm_info.mac_programming[it]=packet_finished[pos-1]->macSH[it];
		}
		for(it=0;it<4;it++)
		{
			firm_info.mac_programming[it+4]=packet_finished[pos-1]->macSL[it];
		}
		
		for(it=0; it<7; it++)
		{
			if( firm_info.ID[it]=='*' ) break;
			firm_info.name_file[it]=firm_info.ID[it];
		}
		firm_info.name_file[it]='\0';
		printf(" name=%s ",firm_info.name_file);
		
		firm_info.packets_received=0;
		firm_info.data_count_packet = 0;
		firm_info.data_count_packet_ant = 0;
		firm_info.paq_disordered = 0;
		firm_info.already_init = 1;
 

		
//		file2.close();
//	
//		file1.close();
//	
//		root.close();
//		
		sd_on=0;
				
		if( !sd_on )
		{
			sd_on=1;
		
			// initializa SPI
//			if(!card.init(SPI_FULL_SPEED)){
//				sd_on=0;
//				error_sd=true;
//			}
	
			// initialize a FAT volume
//			volume.init(&card);

			// open the root directory
//			if(!root.openRoot(&volume)){
//				sd_on=0;
//				error_sd=true;
//			}

		}
		
		// Create the first sector
		for (it = 0;it<448;it++){
			asteriscos[it]='*';
		}
		asteriscos[448] = '\0';
		
//		if( !error_sd )
		{
//			if( !file1.open(&root, firm_info.name_file, O_WRITE | O_CREAT | O_EXCL | O_SYNC | O_APPEND) )
//			{
//				file1.remove(&root,firm_info.name_file);
//				if(!file1.open(&root, firm_info.name_file, O_WRITE | O_CREAT | O_EXCL | O_SYNC | O_APPEND)) error_sd=true;
//			}
//			
//			if( !error_sd )
//			{
//				file1.write(START_SECTOR);
//				file1.write(firm_info.ID);
//				file1.write(asteriscos);
//			}
					   printf("\r\n creat sd file %s ",firm_info.name_file);
					   sprintf(sdfilename,"%s.zhp",firm_info.name_file);
						//创建temp.hex  有则删除新建之
						if(SD.isFile(sdfilename)==1)
						{
							printf(" yes. ");
							SD.del(sdfilename);
											
						}
						else
						{
							printf(" no. ");						
						}
						SD.create(sdfilename);

						uint16_t lenstrsdastr;   
						int32_t offsetfilesd=0;

						lenstrsdastr = strlen(START_SECTOR); printf("strlen=%d ",lenstrsdastr);
						SD.writeSD(sdfilename, (const char*)START_SECTOR, \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr; printf("offset=%d ",offsetfilesd);

						lenstrsdastr = strlen(firm_info.ID);  printf("strlen=%d ",lenstrsdastr);
						SD.writeSD(sdfilename, (const char*)firm_info.ID, \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr;  printf("offset=%d ",offsetfilesd);

						lenstrsdastr = strlen(asteriscos);  printf("strlen=%d ",lenstrsdastr);
						SD.writeSD(sdfilename, (const char*)asteriscos, \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr;  printf("offset=%d ",offsetfilesd);

		}
		
//		if( error_sd )
//		{
//			programming_ON=0;
//			free(asteriscos);
//			asteriscos=NULL;
//			setMulticastConf();
//			return 1;
//		}
		
		free(asteriscos);
		asteriscos=NULL;
		
	}
	// Error mismatch --> Delete the packet
	else
	{
		if(!programming_ON)
		{
			programming_ON=0;
			setMulticastConf();
			return 1;
		}
	}



	return 0;
}


/*
 Function: It receives the data packets of a new firmware
 Returns: Nothing
*/
void WaspXBeeCore::new_firmware_packets()
{

	uint8_t data_bin[92];
	uint16_t sd_index=0;
	bool true_mac = true;
	bool error_sd = false;
	char sdfilename[20];
	long filesize=0;
	it=0;
	
	
	//if( packet_finished[pos-1]->data[it]=='$' && programming_ON )
	if( packet_finished[pos-1]->data[it+6]=='$' && programming_ON )
	{
		//printf(" firm_info.mac ");
//		for(it=0;it<8;it++)
//		{
//			printf(" %2x",firm_info.mac_programming[it]);
//		}
		printf(" %2x",packet_finished[pos-1]->data[7]);

		for(it=0;it<4;it++)
		{
			if( packet_finished[pos-1]->macSH[it] != firm_info.mac_programming[it] )
			{
				true_mac=false;
				break;
			}
		}
		for(it=0;it<4;it++)
		{
			if( packet_finished[pos-1]->macSL[it] != firm_info.mac_programming[it+4] )
			{
				true_mac=false;
				break;
			}
		}
		
		if( true_mac )
		{
			//firm_info.data_count_packet = packet_finished[pos-1]->data[1];
			firm_info.data_count_packet = packet_finished[pos-1]->data[1+6];
			
			if(	(firm_info.data_count_packet == 0 && firm_info.packets_received==0) ||
               	  		(firm_info.data_count_packet - firm_info.data_count_packet_ant) == 1 ||
               			(firm_info.data_count_packet == 0)&&(firm_info.data_count_packet_ant == 255) )
            {
				//for(sd_index=0;sd_index<(packet_finished[pos-1]->data_length-2);sd_index++)
				//{
                //   			data_bin[sd_index]=packet_finished[pos-1]->data[sd_index+2];
               	//}
				for(sd_index=0;sd_index<(packet_finished[pos-1]->data_length-2-6);sd_index++)
				{
                   			data_bin[sd_index]=packet_finished[pos-1]->data[sd_index+2+6];
               	}
				sprintf(sdfilename,"%s.zhp",firm_info.name_file);
				filesize=SD.getFileSize(sdfilename);   
//				printf(" filesize=%d ",filesize);
//				file1.write(data_bin,sd_index);
				SD.writeSD(sdfilename, (const char*)data_bin, \
						filesize, sd_index);				
				firm_info.already_init = 0;

				// Set new OTA previous packet arrival time 
				firm_info.time_arrived=millis();
				
				if(error_sd)
				{
					programming_ON=0;
//					file1.remove(&root,firm_info.name_file);
					firm_info.packets_received=0;
					firm_info.paq_disordered=0;
					setMulticastConf();
				}
				else
				{
					firm_info.packets_received++;
					firm_info.data_count_packet_ant = firm_info.data_count_packet;
					firm_info.paq_disordered=0;
					sd_index=0;
				}
			}
			else if( !(firm_info.data_count_packet == firm_info.data_count_packet_ant) )
			{
				if( (firm_info.data_count_packet - firm_info.data_count_packet_ant) == 2 )
				{
					// re-order one packet lost
					firm_info.paq_disordered=1;
				}
				else
				{
					programming_ON=0;
//					file1.remove(&root,firm_info.name_file);
					firm_info.packets_received=0;
					firm_info.paq_disordered=0;
					setMulticastConf();
					
					OFF();
					delay(10000);
					ON();
					delay(5000);
					//XBee.flush();
					serialFlush(XBEE_UART);
					
				}
			}

		}
		else
		{
			programming_ON=0;
			setMulticastConf();
		}

	}

	else
	{
		if(programming_ON)
		{
			programming_ON=0;
			setMulticastConf();
		}
	}


}


/*
 Function: It receives the last packet of a new firmware
*/
void WaspXBeeCore::new_firmware_end()
{

	char aux_array[5];
	uint8_t data_bin[92];
	uint16_t sd_index=0;
	bool true_mac = true;
	char num_packets_char[5];
	uint16_t num_packets=0;
	packetXBee* paq_sent;
	uint8_t destination[8];
	bool send_ok = true;
	bool error_sd = false;
	char sdfilename[20];
	long filesize=0;	
	long offsetfilesd=0;
	long lenstrsdastr;		
	for(it=0; it<3; it++)
	{
		//aux_array[it]=packet_finished[pos-1]->data[it];
		aux_array[it]=packet_finished[pos-1]->data[it+6];
	}
	aux_array[it]='\0';
	
	if( !strcmp(aux_array,"###") && programming_ON )
	{
		for(it=0;it<4;it++)
		{
			if( packet_finished[pos-1]->macSH[it] != firm_info.mac_programming[it] ){
				true_mac=false;
				break;
			}
		}
		for(it=0;it<4;it++)
		{
			if( packet_finished[pos-1]->macSL[it] != firm_info.mac_programming[it+4] ){
				true_mac=false;
				break;
			}
		}
		
		if( true_mac )
		{
			//for(it=0;it<(packet_finished[pos-1]->data_length-3);it++){
			//	num_packets_char[it]=packet_finished[pos-1]->data[it+3];
			//}
			for(it=0;it<(packet_finished[pos-1]->data_length-3-6);it++){
				num_packets_char[it]=packet_finished[pos-1]->data[it+3+6];
			}
			num_packets_char[it]='\0';
			num_packets = Utils.array2long(num_packets_char);
			
			if( num_packets!=firm_info.packets_received ){
				send_ok = false;
			}
			else send_ok = true;
			
			if( send_ok )
			{
//				file1.close();
//				delay(10);
//				
//				if(!file1.open(&root, firm_info.name_file, O_READ)){
//					send_ok = false;
//				}
			} 
		}
		else{
			send_ok = false;
		} 
	}
	else{
		 send_ok = false;
	}


	
	if( send_ok )
	{
		programming_ON=0;
		firm_info.packets_received=0;

//		if(!file2.open(&root, BOOT_LIST, O_WRITE | O_CREAT | O_EXCL | O_SYNC | O_APPEND) )
//		{
//			if(!file2.open(&root, BOOT_LIST, O_WRITE | O_SYNC | O_APPEND)) error_sd=true;
//		}
				
//		file2.write(firm_info.ID);
//		file2.write(firm_info.DATE);
//		file2.write('\r');
//		file2.write('\n');

				sprintf(sdfilename,"boot.zhp",firm_info.name_file);
						if(SD.isFile(sdfilename)==1)
						{
							printf(" yes. ");											
						}
						else
						{
							printf(" no. ");
							SD.create(sdfilename);						
						}				
				filesize=SD.getFileSize(sdfilename);
				offsetfilesd = filesize;

						lenstrsdastr = strlen(firm_info.ID);
						SD.writeSD(sdfilename, (const char*)firm_info.ID, \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr;

						lenstrsdastr = strlen(firm_info.DATE);
						SD.writeSD(sdfilename, (const char*)firm_info.DATE, \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr;

						lenstrsdastr = strlen("\r\n");
						SD.writeSD(sdfilename, (const char*)"\r\n", \
						offsetfilesd, lenstrsdastr);
						offsetfilesd +=	lenstrsdastr;



		
		if( !error_sd )
		{
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xFC; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE); 
			it=0;
			while(it<4) 
			{ 
				destination[it]=packet_finished[pos-1]->macSH[it]; 
				it++; 
			} 
			while(it<8) 
			{ 
				destination[it]=packet_finished[pos-1]->macSL[it-4]; 
				it++; 
			} 
	
			setDestinationParams(paq_sent, destination, NEW_FIRMWARE_MESSAGE_OK, MAC_TYPE, DATA_ABSOLUTE);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}  
			free(paq_sent); 
			paq_sent=NULL;	
		}
		else
		{
//			file1.remove(&root,firm_info.name_file);
			programming_ON=0;
			firm_info.packets_received=0;
				
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xFC; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE); 
			it=0;
			while(it<4) 
			{ 
				destination[it]=packet_finished[pos-1]->macSH[it]; 
				it++; 
			} 
			while(it<8) 
			{ 
				destination[it]=packet_finished[pos-1]->macSL[it-4]; 
				it++; 
			} 

			setDestinationParams(paq_sent, destination, NEW_FIRMWARE_MESSAGE_ERROR, MAC_TYPE, DATA_ABSOLUTE);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}  
			free(paq_sent); 
			paq_sent=NULL;	
		}
		
		setMulticastConf();	
	}
	else
	{
//		file1.remove(&root,firm_info.name_file);
		programming_ON=0;
		firm_info.packets_received=0;
				
		paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
		paq_sent->mode=UNICAST; 
		paq_sent->MY_known=0; 
		paq_sent->packetID=0xFC; 
		paq_sent->opt=0; 
		hops=0; 
		setOriginParams(paq_sent, "5678", MY_TYPE); 
		it=0;
		while(it<4) 
		{ 
			destination[it]=packet_finished[pos-1]->macSH[it]; 
			it++; 
		} 
		while(it<8) 
		{ 
			destination[it]=packet_finished[pos-1]->macSL[it-4]; 
			it++; 
		} 

		setDestinationParams(paq_sent, destination, NEW_FIRMWARE_MESSAGE_ERROR, MAC_TYPE, DATA_ABSOLUTE);
		srand(millis());
		delay( (rand()%delay_end + delay_start) );
		// Try to send the answer for several times
		for(int k=0; k<MAX_OTA_RETRIES; k++)
		{		
		   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
		   else delay(rand()%delay_end + delay_start);
		}  
		free(paq_sent); 
		paq_sent=NULL;
		
		setMulticastConf();	 
	}

//	file2.close();
//	
//	file1.close();
//	
//	root.close();
		
	sd_on=0;
	
}


/*
 Function: It uploads the new firmware
*/
void WaspXBeeCore::upload_firmware()
{

	// Buscar en boot_list el ID que nos mandan
	// Si no existe responder ERROR
	// Si existe responder OK
	// Resetear al bootloader
	uint16_t num_lines = 0;
	bool id_exist = true;
	uint16_t offset = 0;
	packetXBee* paq_sent;
	uint8_t destination[8];
	long previous=0;
	uint8_t buf_sd[46];
	bool end_file=false;
	uint8_t num_bytes = 0;
	bool reset = false;
	bool startSequence = true;
	bool error_sd = false;
	uint8_t reintentos=0;
	char sdfilename[20];
	long filesize=0;	
	long offsetfilesd=0;
		
	// Check 'KEY_ACCESS'
	for (it = 0; it < 8;it++){
		//if(packet_finished[pos-1]->data[it] != Utils.readEEPROM(it+107))
		if(packet_finished[pos-1]->data[it+6] != Utils.readEEPROM(it+107))
		{
			startSequence = false;  
			break;
		}
	}
	
	if( startSequence )
	{
		if( !sd_on )
		{
			sd_on=1;
		
		// initializa SPI
//			if(!card.init(SPI_FULL_SPEED)){
//				sd_on=0;
//				error_sd=true;
//			}
	
		// initialize a FAT volume
//			volume.init(&card);
	
		// open the root directory
//			if(!root.openRoot(&volume)){
//				sd_on=0;
//				error_sd=true;
//			}
			
//			if(file2.open(&root, BOOT_LIST, O_READ))
//			{
//				sd_on=1;
//				error_sd=false;
//				file2.close();
//			}
		}
	
		if( !error_sd )
		{
			sprintf(sdfilename,"boot.zhp");			  						
			
			//if(file2.open(&root, BOOT_LIST, O_READ))
			if(SD.isFile(sdfilename)==1)
			{
				for(it=0;it<32;it++)
				{
					//firm_info.ID[it]=packet_finished[pos-1]->data[it+8];
					firm_info.ID[it]=packet_finished[pos-1]->data[it+8+6];
				}
				firm_info.ID[it]='\0';
			
				filesize=SD.getFileSize(sdfilename);
				offsetfilesd=0;

				previous=millis();
				//	while( num_lines>0 && (millis()-previous<5000) )
				while( millis()-previous<5000 && !end_file)
				{		
					//if( (num_bytes=file2.read(buf_sd,sizeof(buf_sd))) == 0) end_file=true;
					if((filesize-offsetfilesd)<46) end_file=true;
					SD.readSD(sdfilename, buf_sd, offsetfilesd,46);
					offsetfilesd +=46;

					for(it=0;it<32;it++)
					{
						if(buf_sd[it]!=firm_info.ID[it]){
							id_exist=false;
							break;
						}
					}
					if(!id_exist && !end_file ) id_exist=true;
					else if(id_exist) break;
				}
			}
			else
			{
				id_exist=false;
			}
			
			if(id_exist)
			{
				printf(" boot have the prog %s  ",firm_info.name_file);
				for(it=0; it<7; it++)
				{
					if( firm_info.ID[it]=='*' ) break;
					firm_info.name_file[it]=firm_info.ID[it];
				}
				firm_info.name_file[it]='\0';
			
//				if(!file1.open(&root, firm_info.name_file, O_READ)){
//					id_exist=false;
//				}
				sprintf(sdfilename,"%s.zhp",firm_info.name_file);
				if(SD.isFile(sdfilename)==1);
				else
				   id_exist=false;
			}
			
			
		}
		else
		{
			id_exist=false;
		}
		
		if( id_exist)
		{
			printf(" there hava the prog ^_^ ");
			for(it=0;it<32;it++){
				Utils.writeEEPROM(it+2,firm_info.ID[it]);
			}
		
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xFD; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE); 
			it=0;
			while(it<4) 
			{ 
				destination[it]=packet_finished[pos-1]->macSH[it]; 
				it++; 
			} 
			while(it<8) 
			{ 
				destination[it]=packet_finished[pos-1]->macSL[it-4]; 
				it++; 
			} 

			setDestinationParams(paq_sent, destination, UPLOAD_FIRWARE_MESSAGE_OK, MAC_TYPE, DATA_ABSOLUTE);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}  
			free(paq_sent); 
			paq_sent=NULL;
		
//			file2.close();
//				
//			file1.close();
//				
//			root.close();
		
			sd_on=0;
			
			free(packet_finished[pos-1]);
			packet_finished[pos-1]=NULL;
			
			// Save the transmitter MAC to answer later
			for(it=0;it<8;it++) Utils.writeEEPROM(99+it,destination[it]);
		
			previous=millis();
			while( !reset && millis()-previous<5000 )
			{
				Utils.writeEEPROM(0x01,0x01);
				if( Utils.readEEPROM(0x01)!=0x01 ) Utils.writeEEPROM(0x01,0x01);
				else reset=true;
				delay(10);
				if( millis()-previous < 0 ) previous=millis(); //avoid millis overflow problem
			}
//			PWR.reboot();
			printf(" enter reboot.........................");
			NVIC_SystemReset();
		}
		else
		{
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xFD; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE); 
			it=0;
			while(it<4) 
			{ 
				destination[it]=packet_finished[pos-1]->macSH[it]; 
				it++; 
			} 
			while(it<8) 
			{ 
				destination[it]=packet_finished[pos-1]->macSL[it-4]; 
				it++; 
			} 
		
			setDestinationParams(paq_sent, destination, UPLOAD_FIRWARE_MESSAGE_ERROR, MAC_TYPE, DATA_ABSOLUTE);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}  
			free(paq_sent); 
			paq_sent=NULL;
		  
//			file2.close();
//				
//			file1.close();
//				
//			root.close();
		
			sd_on = 0;	
			
			free(packet_finished[pos-1]);
			packet_finished[pos-1]=NULL;
		} 
	}
	
	
		
}


/*
 Function: It answers the ID requested
*/
void WaspXBeeCore::request_ID()
{
	char PID_aux[33];
	packetXBee* paq_sent;
	uint8_t destination[8];
	char ID_aux[17];
	bool startSequence = true;
	uint8_t readeepromvalue;
			
	
	// Check 'KEY_ACCESS'
	for (it = 0; it < 8;it++){
		readeepromvalue = Utils.readEEPROM(it+107);
		//if(packet_finished[pos-1]->data[it] != readeepromvalue)
		if(packet_finished[pos-1]->data[it+6] != readeepromvalue)		
		{
			printf("i=%d zh=%x d=%x ",it,readeepromvalue,packet_finished[pos-1]->data[it+6]);
			startSequence = false;  
			break;
		}
	}
	
	
	if( startSequence )
	{
		for(it=0;it<32;it++)
		{
			PID_aux[it]=Utils.readEEPROM(it+34);
		}
		PID_aux[32]='\0';
		printf(" PID=%s ",PID_aux);
//		sprintf(PID_aux,"prog001*************************");
//		printf(" PID=%s ",PID_aux);
		
		for(it=0;it<16;it++)
		{
			ID_aux[it]=Utils.readEEPROM(it+147);
		}
		ID_aux[16]='\0';
		printf(" IDaux=%s ",ID_aux);
//		sprintf(ID_aux,"WASPMOTE00000001");
//		printf(" IDaux=%s ",ID_aux);
	
		paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
		paq_sent->mode=UNICAST; 
		//paq_sent->mode=BROADCAST;				   
		paq_sent->MY_known=0; 
		paq_sent->packetID=0xFE; 
		paq_sent->opt=0; 
		hops=0; 
		setOriginParams(paq_sent, "5678", MY_TYPE);
	
		it=0;
		while(it<4) 
		{ 
			destination[it]=packet_finished[pos-1]->macSH[it]; 
			it++; 
		} 
		while(it<8) 
		{ 
			destination[it]=packet_finished[pos-1]->macSL[it-4]; 
			it++; 
		} 
		
		setDestinationParams(paq_sent, destination, PID_aux, MAC_TYPE, DATA_ABSOLUTE);
		setDestinationParams(paq_sent, destination, ID_aux, MAC_TYPE, DATA_OFFSET);
		srand(millis());
		delay( (rand()%delay_end + delay_start) );
		// Try to send the answer for several times
		int kmaxzhi=0;
		for(int k=0; k<MAX_OTA_RETRIES; k++)
		{		
			if(!sendXBee(paq_sent))
			{				
				k=MAX_OTA_RETRIES;
			}
			else delay(rand()%delay_end + delay_start);
			kmaxzhi++;
		}  
		free(paq_sent); 
		paq_sent=NULL;
		printf("k=%d ",kmaxzhi);
	}
	else
	{
	}
}


/*
 Function: It answers the boot list file
*/
void WaspXBeeCore::request_bootlist()
{

	// Buscar en boot_list el ID que nos mandan
	// Si no existe responder ERROR
	// Si existe responder OK
	// Resetear al bootloader
	uint16_t num_lines = 0;
	bool id_exist = true;
	uint16_t offset = 0;
	packetXBee* paq_sent;
	uint8_t destination[8];
	long previous=0;
	uint8_t buf_sd[46];
	char buf_sd_aux[47];
	bool end_file=false;
	uint8_t num_bytes = 0;
	bool reset = false;
	bool startSequence = true;
	uint8_t errors_tx = 0;
	char sdfilename[20];
	long filesize=0;	
	long offsetfilesd=0;
//	long lenstrsdastr;				
	uint8_t readeepromvalue;
	// Check 'KEY_ACCESS'
	for (it = 0; it < 8;it++){
		readeepromvalue = Utils.readEEPROM(it+107);
		//if(packet_finished[pos-1]->data[it] != readeepromvalue)
		if(packet_finished[pos-1]->data[it+6] != readeepromvalue)		
		{
			printf("i=%d zh=%x ",it,readeepromvalue);
			startSequence = false;  
			break;
		}
	}
	
	if( startSequence )
	{
		if( !sd_on )
		{
			sd_on=1;
		
		// initializa SPI
//			if(!card.init(SPI_FULL_SPEED)){
//				sd_on=0;
//			}
	
		// initialize a FAT volume
//			volume.init(&card);
	
		// open the root directory
//			if(!root.openRoot(&volume)){
//				sd_on=0;
//			}
		}
	
		if( sd_on )
		{
			sprintf(sdfilename,"boot.zhp",firm_info.name_file);			  
			//if(file2.open(&root, BOOT_LIST, O_READ))							
			if(SD.isFile(sdfilename)==1)
			{
				printf(" yes. ");
				filesize=SD.getFileSize(sdfilename);
				offsetfilesd=0;
				previous=millis();
				while( millis()-previous<30000 && !end_file)
				{
					
					
					//if( (num_bytes=file2.read(buf_sd,sizeof(buf_sd))) <= 3) end_file=true;
					if((filesize-offsetfilesd)<46) end_file=true;
					SD.readSD(sdfilename, buf_sd, offsetfilesd,46);//(const TCHAR *)
					offsetfilesd +=46;
					if( !end_file )
					{
						for(it=0;it<(sizeof(buf_sd));it++)
						{
							buf_sd_aux[it]=buf_sd[it];
						}
						buf_sd_aux[it]='\0';
				
						paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 

						//paq_sent->mode=UNICAST; 
						paq_sent->mode=BROADCAST;
						 
						paq_sent->MY_known=0; 
						paq_sent->packetID=0xFF; 
						paq_sent->opt=0; 
						hops=0; 
						setOriginParams(paq_sent, "5678", MY_TYPE);
	
						it=0;
						while(it<4) 
						{ 
							destination[it]=packet_finished[pos-1]->macSH[it]; 
							it++; 
						} 
						while(it<8) 
						{ 
							destination[it]=packet_finished[pos-1]->macSL[it-4]; 
							it++; 
						} 
		
						setDestinationParams(paq_sent, destination, buf_sd_aux, MAC_TYPE, DATA_ABSOLUTE);
						srand(millis());
						delay( (rand()%delay_end + delay_start) );
						// Try to send the answer for several times
					        for(int k=0; k<MAX_OTA_RETRIES; k++)
						{		
						   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
						   else delay(rand()%delay_end + delay_start);
						}  
						if( error_TX ) errors_tx++;
						free(paq_sent); 
						paq_sent=NULL;
					}
				}
			
//				file2.close();
//				
//				file1.close();
//				
//				root.close();
		
				sd_on=0;
				
				if( errors_tx )
				{
					paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
					paq_sent->mode=UNICAST; 
					paq_sent->MY_known=0; 
					paq_sent->packetID=0xFF; 
					paq_sent->opt=0; 
					hops=0; 
					setOriginParams(paq_sent, "5678", MY_TYPE);
	
					it=0;
					while(it<4) 
					{ 
						destination[it]=packet_finished[pos-1]->macSH[it]; 
						it++; 
					} 
					while(it<8) 
					{ 
						destination[it]=packet_finished[pos-1]->macSL[it-4]; 
						it++; 
					} 
		
					setDestinationParams(paq_sent, destination, REQUEST_BOOTLIST_MESSAGE, MAC_TYPE, DATA_ABSOLUTE);
					setDestinationParams(paq_sent, destination, "ER", MAC_TYPE, DATA_OFFSET);
					srand(millis());
					delay( (rand()%delay_end + delay_start) );
					// Try to send the answer for several times
		        		for(int k=0; k<MAX_OTA_RETRIES; k++)
					{		
					   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
					   else delay(rand()%delay_end + delay_start);
					}  
					free(paq_sent); 
					paq_sent=NULL;
				}
				else
				{
					paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
					paq_sent->mode=UNICAST; 
					paq_sent->MY_known=0; 
					paq_sent->packetID=0xFF; 
					paq_sent->opt=0; 
					hops=0; 
					setOriginParams(paq_sent, "5678", MY_TYPE);
	
					it=0;
					while(it<4) 
					{ 
						destination[it]=packet_finished[pos-1]->macSH[it]; 
						it++; 
					} 
					while(it<8) 
					{ 
						destination[it]=packet_finished[pos-1]->macSL[it-4]; 
						it++; 
					} 
		
					setDestinationParams(paq_sent, destination, REQUEST_BOOTLIST_MESSAGE, MAC_TYPE, DATA_ABSOLUTE);
					setDestinationParams(paq_sent, destination, "OK", MAC_TYPE, DATA_OFFSET);
					srand(millis());
					delay( (rand()%delay_end + delay_start) );
					// Try to send the answer for several times
		        		for(int k=0; k<MAX_OTA_RETRIES; k++)
					{		
					   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
					   else delay(rand()%delay_end + delay_start);
					}  
					free(paq_sent); 
					paq_sent=NULL;
				}
			}
			else
			{
			} /**/
		}
		else
		{
		}  
	}
	else
	{
	}


}


void WaspXBeeCore::checkNewProgram(){
	
	uint8_t current_ID[32];
	char MID[17];
	uint8_t m = 0;
	bool reprogrammingOK = true;
	uint8_t byte_aux[32];
	packetXBee* paq_sent;
	uint8_t destination[8];
    
//	pinMode(SPI_SCK_PIN, INPUT);
	for(it=0;it<32;it++){
		current_ID[it]= Utils.readEEPROM(it+34);
		Utils.writeEEPROM(it+66,current_ID[it]);
	}
	
	for(it=0;it<16;it++)
	{
		MID[it]=Utils.readEEPROM(it+147);
	}
	MID[16]='\0';
	
	if( Utils.readEEPROM(0x01)==0x01 )
	{
		// Checking if programID and currentID are the same --> the program has been changed properly
		for(it = 0;it<32;it++){
			byte_aux[it] = Utils.readEEPROM( it+2);
		}
	
		for(it = 0;it<32;it++){
			if (byte_aux[it] != Utils.readEEPROM( it+34)){
				reprogrammingOK = false;
			}
		}
		Utils.writeEEPROM(0x01,0x00);
		
		// If both IDs are equal a confirmation message is sent to the trasmitter
		if (reprogrammingOK){
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xF9; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE);
	
			for(it=0;it<8;it++) destination[it]=Utils.readEEPROM(99+it);
		
			setDestinationParams(paq_sent, destination, MID, MAC_TYPE, DATA_ABSOLUTE);
			setDestinationParams(paq_sent, destination, ANSWER_START_WITH_FIRMWARE_OK, MAC_TYPE, DATA_OFFSET);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}  
			free(paq_sent); 
			paq_sent=NULL;
		}
		// If the IDs are different an error message is sent to the transmitter
		else
		{
			paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
			paq_sent->mode=UNICAST; 
			paq_sent->MY_known=0; 
			paq_sent->packetID=0xF9; 
			paq_sent->opt=0; 
			hops=0; 
			setOriginParams(paq_sent, "5678", MY_TYPE);
		
			for(it=0;it<8;it++) destination[it]=Utils.readEEPROM(99+it);
		
			setDestinationParams(paq_sent, destination, MID, MAC_TYPE, DATA_ABSOLUTE);
			setDestinationParams(paq_sent, destination, ANSWER_START_WITH_FIRMWARE_ERR, MAC_TYPE, DATA_OFFSET);
			srand(millis());
			delay( (rand()%delay_end + delay_start) );
			// Try to send the answer for several times
		        for(int k=0; k<MAX_OTA_RETRIES; k++)
			{		
			   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
			   else delay(rand()%delay_end + delay_start);
			}   
			free(paq_sent); 
			paq_sent=NULL;
		}
	}
	else
	{
		paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
		paq_sent->mode=BROADCAST; 
		paq_sent->MY_known=0; 
		paq_sent->packetID=0xF9; 
		paq_sent->opt=0; 
		hops=0; 
		setOriginParams(paq_sent, "5678", MY_TYPE);
				
		setDestinationParams(paq_sent, destination, MID, MAC_TYPE, DATA_ABSOLUTE);
		setDestinationParams(paq_sent, destination, RESET_MESSAGE, MAC_TYPE, DATA_OFFSET);
		srand(millis());
		delay( (rand()%delay_end + delay_start) );
		// Try to send the answer for several times
		for(int k=0; k<MAX_OTA_RETRIES; k++)
		{		
		   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
		   else delay(rand()%delay_end + delay_start);
		}  
		free(paq_sent); 
		paq_sent=NULL;
	}

}

void WaspXBeeCore::delete_firmware()
{


	// Buscar en boot_list el ID que nos mandan y lo borra. También borra el fichero con ese nombre en la SD
	// Si no existe responder ERROR
	// Si existe responder OK
	packetXBee* paq_sent;
	uint8_t destination[8];
	long previous=0;
	char buf_sd[46];
	char buf_sd_aux[47];
	bool end_file=false;
	uint8_t num_bytes = 0;
	bool startSequence = true;
	char file_to_delete[8];
	bool error=false;
	char* file_aux = "FILEAUX";
	bool match_id = true;
	char sdfilename[20];
	long filesizeboot=0;	
	long offsetfilesdboot=0;
	long filesizeboot2=0;	
	long offsetfilesdboot2=0;
//	uint8_t bufreadsd[100];			
	// Check 'KEY_ACCESS'
	for (it = 0; it < 8;it++){
		//if(packet_finished[pos-1]->data[it] != Utils.readEEPROM(it+107))
		if(packet_finished[pos-1]->data[it+6] != Utils.readEEPROM(it+107))		
		{
			startSequence = false;  
			break;
		}
	}

	if( startSequence )
	{  
		if( !sd_on )
		{
			sd_on=1;
		
		// initialize SPI
//			if(!card.init(SPI_FULL_SPEED)){
//				sd_on=0;
//			}
	
		// initialize a FAT volume
//			volume.init(&card);
	
		// open the root directory
//			if(!root.openRoot(&volume)){
//				sd_on=0;
//			}
		}
			
		if( sd_on )
		{  
			// Store the file to delete
//			for(it=0;it<7;it++){
//				if(packet_finished[pos-1]->data[it+8]=='*') break;
//				file_to_delete[it]=packet_finished[pos-1]->data[it+8];
//			}
			for(it=0;it<7;it++){
				if(packet_finished[pos-1]->data[it+8+6]=='*') break;
				file_to_delete[it]=packet_finished[pos-1]->data[it+8+6];
			}
			file_to_delete[7]='\0';

			sprintf(sdfilename,"%s.zhp",file_to_delete);	
					  									
			// Open boot list
			//if(file2.open(&root, BOOT_LIST, O_READ))
			//if(SD.isFile(sdfilename)==1)
			if((SD.isFile(sdfilename)==1)&&(SD.isFile("boot.zhp")==1))
			{
				// Delete firmware file
			/*	if(!file1.remove(&root, file_to_delete)) error=true;
				file1.close();
				
				// Create auxiliary file
				if(!file1.open(&root, file_aux, O_WRITE | O_CREAT | O_EXCL | O_SYNC | O_APPEND)) error=true;
			*/	
				SD.create("boot2.zhp");
				filesizeboot2=0;	
				offsetfilesdboot2=0;
				filesizeboot=SD.getFileSize("boot.zhp");
				offsetfilesdboot=0; 

				// Algorithm to copy boot_list but the line we want to delete
				previous=millis();
				while( millis()-previous<60000 && !end_file)
				{						
					//if( (num_bytes=file2.read(buf_sd,sizeof(buf_sd))) <= 3) end_file=true;
					if( (offsetfilesdboot+46) > filesizeboot) end_file=true;
					SD.readSD("boot.zhp", buf_sd, offsetfilesdboot,46);	
					offsetfilesdboot+=46 ;
					if( !end_file )
					{
//						for(it=0;it<(sizeof(buf_sd));it++)
//						{
//							buf_sd_aux[it]=buf_sd[it];
//						}
//						buf_sd_aux[it]='\0';
//						
//						for(it=0;it<7;it++)
//						{
//							if(buf_sd_aux[it]!=file_to_delete[it])
//							{
//								match_id=false;
//								break;
//							}
//						}
//
//						if(!match_id)
//						{
//							file1.write(buf_sd_aux);
//							match_id = true;
//						}
						for(it=0;it<7;it++)
						{
							if(buf_sd[it]!=file_to_delete[it])
							{
								match_id=false;
								break;
							}
						}
						if(!match_id)
						{
							SD.writeSD("boot2.zhp", (const char*)buf_sd,offsetfilesdboot2, 46);
							offsetfilesdboot2 +=	46; 
							match_id = true;
						}												
					}
				} 				
				end_file=false;
//				file1.close();
//				file2.close();
//				
//				// Delete previous boot_list
//				if(!file2.remove(&root,BOOT_LIST)) error=true;
//				
//				// Create a new boot_list file and copy the content of auxiliary file
//				if(!file2.open(&root, BOOT_LIST, O_WRITE | O_CREAT | O_EXCL | O_SYNC | O_APPEND) ) error=true;

				SD.del("boot.zhp");
				SD.del(sdfilename);
				SD.create("boot.zhp");

				filesizeboot=0;	
				offsetfilesdboot=0;
				filesizeboot2=SD.getFileSize("boot2.zhp");
				offsetfilesdboot2=0; 
				/*	*/
//				if(file1.open(&root, file_aux, O_READ))
//				{
					previous=millis();	
					while( millis()-previous<60000 && !end_file)
					{
						//if( (num_bytes=file1.read(buf_sd,sizeof(buf_sd))) <= 3) end_file=true;
						if( (offsetfilesdboot2+46) > filesizeboot2) end_file=true;
						SD.readSD("boot2.zhp", buf_sd, offsetfilesdboot2,46);	
						offsetfilesdboot2+=46 ;							
						if( !end_file )
						{
//							for(it=0;it<(sizeof(buf_sd));it++)
//							{
//								buf_sd_aux[it]=buf_sd[it];
//							}
//							buf_sd_aux[it]='\0';							 
							//file2.write(buf_sd_aux);
							SD.writeSD("boot.zhp", (const char*)buf_sd,offsetfilesdboot, 46);
							offsetfilesdboot +=	46;
						}
					}
//					file1.close();
//					file1.remove(&root,file_aux);
//				}
//				else error=true;  
			}
			else error=true;
		}
		else error=true;  
	}
	else error=true;
	
	
	if(!error)
	{
		paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
		paq_sent->mode=UNICAST; 
		paq_sent->MY_known=0; 
		paq_sent->packetID=0xF8; 
		paq_sent->opt=0; 
		hops=0; 
		setOriginParams(paq_sent, "5678", MY_TYPE);
	
		it=0;
		while(it<4) 
		{ 
			destination[it]=packet_finished[pos-1]->macSH[it]; 
			it++; 
		} 
		while(it<8) 
		{ 
			destination[it]=packet_finished[pos-1]->macSL[it-4]; 
			it++; 
		} 
		
		setDestinationParams(paq_sent, destination, DELETE_MESSAGE_OK, MAC_TYPE, DATA_ABSOLUTE);
		srand(millis());
		delay( (rand()%delay_end + delay_start) );
		// Try to send the answer for several times
		for(int k=0; k<MAX_OTA_RETRIES; k++)
		{		
		   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
		   else delay(rand()%delay_end + delay_start);
		}  
		free(paq_sent); 
		paq_sent=NULL;
	}
	else
	{
		paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
		paq_sent->mode=UNICAST; 
		paq_sent->MY_known=0; 
		paq_sent->packetID=0xF8; 
		paq_sent->opt=0; 
		hops=0; 
		setOriginParams(paq_sent, "5678", MY_TYPE);
	
		it=0;
		while(it<4) 
		{ 
			destination[it]=packet_finished[pos-1]->macSH[it]; 
			it++; 
		} 
		while(it<8) 
		{ 
			destination[it]=packet_finished[pos-1]->macSL[it-4]; 
			it++; 
		} 
		
		setDestinationParams(paq_sent, destination, DELETE_MESSAGE_ERROR, MAC_TYPE, DATA_ABSOLUTE);
		srand(millis());
		delay( (rand()%delay_end + delay_start) );
		// Try to send the answer for several times
	        for(int k=0; k<MAX_OTA_RETRIES; k++)
		{		
		   if(!sendXBee(paq_sent)) k=MAX_OTA_RETRIES;
		   else delay(rand()%delay_end + delay_start);
		}  
		free(paq_sent); 
		paq_sent=NULL;
	}

//	file2.close();
//
//	file1.close();
//
//	root.close();

	sd_on=0;
	
		
}

void WaspXBeeCore::setMulticastConf()
{
	switch( firm_info.multi_type )
	{
		case 0:		setChannel(firm_info.channel);
				writeValues();
				break;
			
		case 1: 	// Set previous 'Auth key'
				for (it = 0; it < 8;it++) Utils.writeEEPROM(it+107, firm_info.authkey[it]);
				break;
			
		case 2: 	setLinkKey(firm_info.encryptionkey);
				writeValues();
				break;
	}
}


/*
 Function: Checks if timeout is up while sending program packets
 Returns: Integer that determines if there has been any error 
   1 --> Timeout is up
   0 --> The function has been executed with no errors   
*/
uint8_t WaspXBeeCore::checkOtapTimeout()
{
	long total_time;

	if( programming_ON )
   	{
	   // Check millis crossing through zero
	   if( (millis()-firm_info.time_arrived)<0 ) total_time=millis();//we don't count time till zero
	   else total_time=millis()-firm_info.time_arrived;

	   if( OTA_TIMEOUT < total_time )
	   {
		// Reach Timeout 
		programming_ON=0;
//		file1.remove(&root,firm_info.name_file);
		firm_info.packets_received=0;
		firm_info.paq_disordered=0;
		setMulticastConf();
		
		OFF();
		delay(1000);
		ON();
		delay(1000);
		//XBee.flush();
		serialFlush(XBEE_UART);
		return 1;
	   }
   	}
	return 0;
}


uint8_t WaspXBeeCore::sendTx64Simple(char datastr[],unsigned char lendata,unsigned long addressH,unsigned long addressL)
{
	uint8_t oldnetaddressH = xbee802.sourceNA[0];
	uint8_t oldnetaddressL = xbee802.sourceNA[1];
	uint8_t i;
	uint8_t flagerr;
	uint8_t flagerr2;

//	FlagTimeLinshi=0;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;
	flagerr = xbee802.setOwnNetAddress(0xff,0xff);
	if(flagerr!=0)return flagerr;
//	FlagTimeLinshi=2;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;

	SendAPIFrameStr[0] = 0x7e;
	SendAPIFrameStr[3] = 0x00;
	SendAPIFrameStr[4] = 0x52;

	SendAPIFrameStr[5]  =  (uint8_t)(addressH>>24);
	SendAPIFrameStr[6]  =  (uint8_t)(addressH>>16);
	SendAPIFrameStr[7]  =  (uint8_t)(addressH>>8);
	SendAPIFrameStr[8]  =  (uint8_t)(addressH>>0);
	SendAPIFrameStr[9]  =  (uint8_t)(addressL>>24);
	SendAPIFrameStr[10] =  (uint8_t)(addressL>>16);
	SendAPIFrameStr[11] =  (uint8_t)(addressL>>8);
	SendAPIFrameStr[12] =  (uint8_t)(addressL>>0);

	SendAPIFrameStr[13] = 0x00;//要求应答不应答什么的
	for(i=0;i<lendata;i++)
	{
		SendAPIFrameStr[14+i] = datastr[i];	
	}

	//数据长度暂时只做小于256个的

	SendAPIFrameStr[2] = 1+1+8+1+lendata;//命令1个 帧1个 地址8个 option 1个 数据若干个	
	LenSendAPIFrameStr =  SendAPIFrameStr[2]+4;
	get_check((char *)SendAPIFrameStr, LenSendAPIFrameStr);
	
//	printf("len=%d ",LenSendAPIFrameStr);
//	for(i=0;i<LenSendAPIFrameStr;i++)
//		printf(" %2x,",SendAPIFrameStr[i]);
//	printf("send simple 64Tx ");


	flagerr = sendhandleresponse((char *)SendAPIFrameStr, LenSendAPIFrameStr);
//	FlagTimeLinshi=4;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;
	//if(flagerr!=0)return flagerr;
	

	flagerr2 = xbee802.setOwnNetAddress(oldnetaddressH,oldnetaddressL);
//	FlagTimeLinshi=6;TimemsArray[FlagTimeLinshi] = timer0_overflow_count;
	if(flagerr!=0)return flagerr;
	else return flagerr2;
}

