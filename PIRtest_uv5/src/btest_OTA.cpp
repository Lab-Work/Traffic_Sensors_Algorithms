/*
这个是板子的LED例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：    
#define BTESTUTILSLED 0		 LED例程
#define BTESTUTILSNULL 100	 不选用Util例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTOTA 0
#define BTESTOTAANDOTHERREC 1
#define BTESTOTANULL 100

#define EXAMPLEOTA  BTESTOTANULL

//LED
#if EXAMPLEOTA==	BTESTOTA
uint8_t DataFromEeprom[200]={"LIBELIUM0102030405060708090"};
void setup()
{
	uint8_t datai;
	Utils.initLEDs();
	Utils.setLED(2, 1);
	delay_ms(50);
	Utils.setLED(2, 0);
	delay_ms(50);

	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, 1); chooseuartinterrupt(1);
	delay_ms(200);
	serialWrite('7', 1);
	delay_ms(200);
//	destMacAddagain[16]='\0';

	delay_ms(200);
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	serialWrite('o', 1);
	delay_ms(200);
	xbee802.ON(); 	
	monitor_offuart3TX();monitor_onuart3RX(); //放到xbee初始化前就是不能捕获数据（看不到收到的数据），也不知道为什么
	xbee802.getChannel();
	printf("firstgetchannel=%x ",xbee802.channel);
	delay_ms(30);
	xbee802.getChannel();	
	printf("secondgetchannel=%x ",xbee802.channel);
	xbee802.getMacMode();
	printf("getMacMode=%x\t\t",xbee802.macMode);
	xbee802.setOwnNetAddress(0xff, 0xff);
	printf("getMY");
	xbee802.getOwnNetAddress();
	printf("=%x %x \t\t",xbee802.sourceNA[0],xbee802.sourceNA[1]);
	Eeprom.ON();	
	Eeprom.begin();
	delay_ms(200);	
	printf("  read eeprom from 107 len=8 ");
	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
	for(datai=0;datai<8;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
	else
	{	
		printf("\r\n WRG  or else \r\n");
		Eeprom.writeEEPROMStr(0xa0,107,(uint8_t *)"LIBELIUM0102030405060708090",16);
		for(datai=0;datai<200;datai++)
		{
			DataFromEeprom[datai]=datai;	
		}
	}
	printf(" 2th read eeprom from 107 len=8 ");
	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
	for(datai=0;datai<8;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
	else 
	{
		printf("\r\n\r\n ********************WRONG****************** \r\n\r\n");
	}

	printf("read eeprom:");
	Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
	for(datai=0;datai<200;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	uint8_t flag0xff=0;
	for(datai=0;datai<200;datai++)
	{
		if(DataFromEeprom[datai]==0xff)
		{
			flag0xff=1;
			if(datai<=33)DataFromEeprom[datai]=0x31;
			else if(datai<=97)DataFromEeprom[datai]=0x32;
			else if(datai<=106)DataFromEeprom[datai]=0x33;
			else if(datai<=162)DataFromEeprom[datai]=0x34;
			else DataFromEeprom[datai]=0x39;			 
		}	
	}
	if(flag0xff==1)
	{
		Eeprom.writeEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
		printf("read eeprom again:");
		Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
		for(datai=0;datai<200;datai++)
		{
			printf("%2x ",DataFromEeprom[datai]);	
		}
	}

	printf("end read eeprom");

		//SD卡电源开
		printf("\n  hi SD poweron. ");		 
	 	SD.ON();
	
	//SD初始化，成功的话也初始化一下FAT	
		printf(" SD init. ");    
		SD.init();
		if(SD.flag!=NOTHING_FAILED)
		{
			printf("  failed!  ");
			delay_ms(500);
			while(1);
		}
	printf("enter ota choose ");
	while(1)
	{
	  // Check if new data is available
	  if( XBee.available() )
	  {
//	  	printf("&");
	    xbee802.treatData();
	    // Keep inside this loop while a new program is being received
	    while( xbee802.programming_ON  && !xbee802.checkOtapTimeout() )
	    {
	      if( XBee.available() )
	      {
	        xbee802.treatData();

	      }
	    }

	  }
	}

}

void loop()
{
	delay_ms(100);
	
}
#endif // end OTA

#if EXAMPLEOTA==	BTESTOTAANDOTHERREC
uint8_t DataFromEeprom[200]={"LIBELIUM0102030405060708090"};
void setup()
{
	uint8_t datai;
	Utils.initLEDs();
	Utils.setLED(2, 1);
	delay_ms(50);
	Utils.setLED(2, 0);
	delay_ms(50);
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, 1); chooseuartinterrupt(1);
	delay_ms(200);
	serialWrite('5', 1);
	delay_ms(200);
//	destMacAddagain[16]='\0';

	delay_ms(200);
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	serialWrite('o', 1);
	delay_ms(200);
	xbee802.ON(); 	
	monitor_offuart3TX();monitor_onuart3RX(); //放到xbee初始化前就是不能捕获数据（看不到收到的数据），也不知道为什么
	xbee802.getChannel();
	printf("firstgetchannel=%x ",xbee802.channel);
	delay_ms(30);
	xbee802.getChannel();	
	printf("secondgetchannel=%x ",xbee802.channel);
	xbee802.getMacMode();
	printf("getMacMode=%x\t\t",xbee802.macMode);
	xbee802.setOwnNetAddress(0xff, 0xff);
	printf("getMY");
	xbee802.getOwnNetAddress();
	printf("=%x %x \t\t",xbee802.sourceNA[0],xbee802.sourceNA[1]);
	Eeprom.ON();	
	Eeprom.begin();
	delay_ms(200);	
	printf("  read eeprom from 107 len=8 ");
	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
	for(datai=0;datai<8;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
	else
	{	
		printf("\r\n WRG  or else \r\n");
		Eeprom.writeEEPROMStr(0xa0,107,(uint8_t *)"LIBELIUM0102030405060708090",16);
		for(datai=0;datai<200;datai++)
		{
			DataFromEeprom[datai]=datai;	
		}
	}
	printf(" 2th read eeprom from 107 len=8 ");
	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
	for(datai=0;datai<8;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
	else 
	{
		printf("\r\n\r\n ********************WRONG****************** \r\n\r\n");
	}

	printf("read eeprom:");
	Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
	for(datai=0;datai<200;datai++)
	{
		printf("%2x ",DataFromEeprom[datai]);	
	}
	uint8_t flag0xff=0;
	for(datai=0;datai<200;datai++)
	{
		if(DataFromEeprom[datai]==0xff)
		{
			flag0xff=1;
			if(datai<=33)DataFromEeprom[datai]=0x31;
			else if(datai<=97)DataFromEeprom[datai]=0x32;
			else if(datai<=106)DataFromEeprom[datai]=0x33;
			else if(datai<=162)DataFromEeprom[datai]=0x34;
			else DataFromEeprom[datai]=0x39;			 
		}	
	}
	if(flag0xff==1)
	{
		Eeprom.writeEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
		printf("read eeprom again:");
		Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
		for(datai=0;datai<200;datai++)
		{
			printf("%2x ",DataFromEeprom[datai]);	
		}
	}

	printf("end read eeprom");

		//SD卡电源开
		printf("\n  hi SD poweron. ");		 
	 	SD.ON();
	
	//SD初始化，成功的话也初始化一下FAT	
		printf(" SD init. ");    
		SD.init();
		if(SD.flag!=NOTHING_FAILED)
		{
			printf("  failed!  ");
			delay_ms(500);
			while(1);
		}
	printf("enter ota choose ");
	unsigned char k;
	while(1)
	{
		// Check if new data is available
		if( XBee.available() )
		{
		//	  	printf("&");
		    xbee802.treatData();
			if( !xbee802.error_RX )
			{
				// Sending answer back
				while(xbee802.pos>0)
				{				
					//printf("The whole data is =%s",xbee802.packet_finished[xbee802.pos-1]->data); 
					//printf("seq %d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data); 
		
					printf("RSSI: =%x",xbee802.packet_finished[xbee802.pos-1]->RSSI);
					printf("len: =%x",xbee802.packet_finished[xbee802.pos-1]->data_length); 
					for(k=0;k<xbee802.packet_finished[xbee802.pos-1]->data_length;k++)
						printf(" %2x",xbee802.packet_finished[xbee802.pos-1]->data[k]);
									 
					free(xbee802.packet_finished[xbee802.pos-1]);
					xbee802.packet_finished[xbee802.pos-1]=NULL;
					xbee802.pos--;
				} 
		
			}
			printf("-");
		
			// Keep inside this loop while a new program is being received
			while( xbee802.programming_ON  && !xbee802.checkOtapTimeout() )
			{
				if( XBee.available() )
				{
					xbee802.treatData();
				}
			}
		
		}
	}

}

void loop()
{
	delay_ms(100);
	
}
#endif // end OTAandrec


