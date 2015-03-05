/*
这个是20120902蒋工提出来的测试例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
（1）	连接好主PCB和扩展板，固定连接好电池监测板和电池。主PCB，扩展板，以及电池（电池监测板）放入保护壳内； 
（2）	连接传感器（两组红外，每组3个，加一个声纳），用传感器收数，数据打包，然后用XBEE将数据传出来，用另一个节点接收。传感器采样频率10HZ，XBEE发送每秒钟四个包，数据包的结构可以参见附近中的程序。要求至少测试某一个节点一天的数据，检查其稳定性；
（3）	测试一下PCB在带外壳里直接点对点直接通讯距离（使用要求带外壳户外至少200米）；
（4）	测试一下节能模式的使用，要求在电池电量不足（电压小于2.8V）时可以自动转入节能模式：XBEE断电，其他模块断电，只保留RTC和单片机，单片机进入低功耗模式，电池电量恢复时，通过中断（PC1）唤醒单片机，接通其他模块电源， 该功能具体由以下电路实施； 
（5）	测试一下用RTC中断使系统进入节能模式，通过设置RTC时间，进入，跳出节能模式； 
（6）	估算一下在天气良好情况下的每天通过太阳板获得的能量，并估计一下每秒4个包，采样频率为10HZ（每0。1秒收一个传感器的数据）时板子的耗电量，建议一个可以可持续工作（获得能量与消耗能量相抵）的最佳CPU频率。
例程有如下几种：    
#define BTESTDATE0902 0		  测试	20120902
#define BTESTDATE0902NULL 100  不测试
不管选择哪个，程序都是有setup()和loop()两部分。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTDATE0902_2 2
#define BTESTDATE0902_3_send 31
#define BTESTDATE0902_3_rec 32
#define BTESTDATE0902_4 4
#define BTESTDATE0902_5 5
#define BTESTDATE0902_6 6
#define BTESTDATE0902_4and5 7 //这个是为了监测硬件用的，先是通过RTC复位，然后通过电池电量恢复复位
#define BTESTDATE0902_2pre 8
#define BTESTDATE0902_2messadd 9


#define BTESTDATE0902NULL 0



#define EXAMPLEDATE0902 BTESTDATE0902NULL
static packetXBee* paq_sent;
//LED
#if EXAMPLEDATE0902==BTESTDATE0902_2
uint32_t CntSend=1;
uint32_t CntSendedOk=0;
uint8_t LastSecond=0;
uint8_t Cnt10=0;

uint8_t t_hour;

void sendBroadcastSignal(char data[])
{
  

              paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
			  if(paq_sent==NULL)printf(" you wenti ****************** ");
              paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
              paq_sent->MY_known=0;
              paq_sent->packetID=0x52;  //Think about changing it each time you send
              paq_sent->opt=0;
              xbee802.hops=0;
              xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
              xbee802.setDestinationParams(paq_sent, "000000000000FFFF", data, MAC_TYPE, DATA_ABSOLUTE);
              xbee802.sendXBee(paq_sent);
            
              if( !xbee802.error_TX )
              {
                //XBee.println("ok");
				CntSendedOk++;
              }
              free(paq_sent);
              paq_sent=NULL;
}

void setup()
{
//	char destAdd[17];

	Utils.initLEDs();
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	

	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();//用USB监控一下xbee接收	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	printf("1");
	printf("2");	
//	destAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();

	//Setting the RTC
	RTCbianliang.ON();
	RTCbianliang.begin();
	delay_ms(100); //初始化RTC及上电之后 要延时一点时间，要不然直接settime不管用的
	RTCbianliang.setTime("Sun, 12/10/04 - 05:59:07");
	RTCbianliang.getTime();
	printf("%s\t\r\n", RTCbianliang.timeStamp);
	t_hour = RTCbianliang.hour;
	//  print_memory();	 //原来的api里面没有
	
	beginMb7076(); //初始化MB7076,实际上这里有三件事情，1.开启串口2,2串口2接收中断使能，3MB7076电源上电
	printf("7"); 	printf("%s\t\r\n", RTCbianliang.timeStamp);
	printf("SMBus_Init");
	SMBus_Init();//做了两件事1.SMBUS那边电源上电，2初始化I2C那两个口， ***注意***这里的I2C和RTC的不是公用的
	printf(" enter the SMBus protocol.");	
	SMBus_Apply();//使能MLX90614的MSBUS的功能
 	delay(300);
   //changeMlx90614Subadd(0xa2, 0xc0);




}


static char Str0902_3_Send[200]="123";


void loop()
{
	int valuesonar;
	int flagsonar;
//	int i;
//	int head2;
//	int flagrtc=I2CTRUE;
//	int slave_add;
	Cnt10++;
	if(Cnt10>=7)
	{
		Cnt10=0;
		while(1)
		{
			RTCbianliang.getTime();


			if(RTCbianliang.second!=LastSecond)
			{
				LastSecond = RTCbianliang.second;
				printf("%s\t\r\n"\
				, RTCbianliang.timeStamp);
				break;
			}
			delay_ms(10);
		}

	}

	if(Cnt10==0)Utils.setLED(0, 1);
	else if(Cnt10==1)Utils.setLED(0, 0);

//	slave_add= MEM_READ1(0x00,0x2e);
//	printf(" add=%x ",slave_add);
	
	flagsonar=readMb7076(&valuesonar);
	float temp1,temp2,temp3;
	//关于读温度这个可以参考btest_ii2c.cpp里面的#if EXAMPLEI2C==BTESTMLX90614  
	temp1=readMlx90614ObjectTemp(0xa0);		
	temp2=readMlx90614ObjectTemp(0xb0);	
	temp3=readMlx90614ObjectTemp(0xc0);	
		
	sprintf(Str0902_3_Send\
	,"%d\t%d\t%4d\t%f'C\t%f'C\t%f'C\r\n"\
	,CntSend,CntSendedOk,valuesonar,temp1,temp2,temp3);

	printf("%s",Str0902_3_Send);
	sendBroadcastSignal(Str0902_3_Send);

	CntSend++;			
}
#endif // end 	BTESTDATE0902_2



#if EXAMPLEDATE0902==BTESTDATE0902_2pre
void setup()
{
	Utils.initLEDs();
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	
	delay_ms(100);
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	
	delay_ms(100);
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	
	delay_ms(100);

	Utils.setLED(1, 1);
	delay_ms(100);
	Utils.setLED(1, 0);	
	delay_ms(100);
	Utils.setLED(1, 1);
	delay_ms(100);
	Utils.setLED(1, 0);	
	delay_ms(100);
	Utils.setLED(1, 1);
	delay_ms(100);
	Utils.setLED(1, 0);	
	delay_ms(100);





	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	printf("1");
	printf("2");	



	//Setting the RTC
	RTCbianliang.ON();
	RTCbianliang.begin();
	RTCbianliang.setTime("Sun, 12/10/04 - 05:59:07");
	RTCbianliang.getTime();


	SMBus_Init();
	printf(" enter the SMBus protocol.");	
	SMBus_Apply();
 	delay_ms(300);
   //changeMlx90614Subadd(0xa2, 0xc0);

}
static char Str0902_3_Send[200]="123";
void loop()
{
	int slave_add;
	uint8_t newadd;
	int flagzhi;

	slave_add= MEM_READ1(0x00,0x2e);
	printf(" add=%x ",slave_add);

	float temp1,temp2,temp3;
	 
	temp1=readMlx90614ObjectTemp((uint8_t)slave_add);		
//	temp2=readMlx90614ObjectTemp(0xb0);	
//	temp3=readMlx90614ObjectTemp(0xc0);	
		
	sprintf(Str0902_3_Send\
	,"%f'C\r\n"\
	,temp1);
	printf("%s",Str0902_3_Send);
	delay_ms(500);

	printf("enter choose, 0 dont change address  1 change it");
	scanf("%d",&flagzhi);
	//printf(" <%d> ",flagzhi);
	if(flagzhi==1)
	{
		printf("enter new add(hex) ",&flagzhi);
		scanf("%x",&flagzhi);
		if(flagzhi<256)
		{
			//printf("new add is %x",flagzhi);
			newadd = flagzhi;
			changeMlx90614Subadd(0x00, newadd);
			printf(" newadd should be %x",newadd);
		}
		else printf("wrong address!");
	}

			
}
#endif // end 	BTESTDATE0902_2pre




#if EXAMPLEDATE0902==BTESTDATE0902_2messadd
uint32_t CntSend=1;
uint32_t CntSendedOk=0;
uint8_t LastSecond=0;
uint8_t Cnt10=0;
uint8_t AllMessAdd[10]={0,0,0,0,0,0,0,0,0,0};
uint8_t MaxMessAdd=0;
float   AllMessTemp[10];

//********************************************************************************************************
//****************Methods used to send the packets to the sink******************
//********************************************************************************************************
void sendBroadcastSignal(char data[])
{
  

              paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
			  if(paq_sent==NULL)printf(" you wenti ****************** ");
              paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
              paq_sent->MY_known=0;
              paq_sent->packetID=0x52;  //Think about changing it each time you send
              paq_sent->opt=0;
              xbee802.hops=0;
              xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
              xbee802.setDestinationParams(paq_sent, "000000000000FFFF", data, MAC_TYPE, DATA_ABSOLUTE);
              xbee802.sendXBee(paq_sent);
            
              if( !xbee802.error_TX )
              {
                //XBee.println("ok");
				CntSendedOk++;
              }
              free(paq_sent);
              paq_sent=NULL;
}

void setup()
{
	Utils.initLEDs();
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	

	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	printf("1");
	printf("2");	
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();

	//Setting the RTC
	RTCbianliang.ON();
	RTCbianliang.begin();
	RTCbianliang.setTime("11:08:16:03:14:58:00");
	RTCbianliang.getTime();

	//  print_memory();	 //原来的api里面没有
	beginMb7076(); 
		printf("7"); 
	printf("SMBus_Init");
	SMBus_Init();
	printf(" enter the SMBus protocol.");	
	SMBus_Apply();
 	delay(300);

	uint8_t k;
	uint8_t xulie=0;
	float temp;
	 
	
	for(k=0x70;k<0x77;k++)
	{
		temp=readMlx90614ObjectTemp(k);	
		if((temp>10.0)&&(temp<50.0))
		{
			AllMessAdd[xulie]=k;
			xulie++;
		}	
	}
	for(k=0xA0;k<0xC1;k+=0x10)
	{
		temp=readMlx90614ObjectTemp(k);	
		if((temp>10.0)&&(temp<350.0))
		{
			AllMessAdd[xulie]=k;
			xulie++;
		}	
	}

	MaxMessAdd = xulie;
	printf("MaxMessAdd =%d ",MaxMessAdd);
	for(k=0;k<MaxMessAdd;k++)
	{
		printf(" %x,",AllMessAdd[k]);
	}
}


static char Str0902_3_Send[200]="123";
static char Str0902_3_Sendapacket[40]="123";

void loop()
{
	int valuesonar;
	int flagsonar;
	int i;
	int head2;
	int flagrtc=I2CTRUE;
	int slave_add;
	Cnt10++;
	if(Cnt10>=5)
	{
		Cnt10=0;
		while(1)
		{
			RTCbianliang.getTime();


			if(RTCbianliang.second!=LastSecond)
			{
				LastSecond = RTCbianliang.second;
				printf("%s\t\r\n"\
				, RTCbianliang.timeStamp);
				break;
			}
			delay_ms(10);
		}

	}

	if(Cnt10==0)Utils.setLED(0, 1);
	else if(Cnt10==1)Utils.setLED(0, 0);

	slave_add= MEM_READ1(0x00,0x2e);
	printf(" add=%x ",slave_add);
	
	flagsonar=readMb7076(&valuesonar);

	sprintf(Str0902_3_Send\
	,"%d\t%d\t%4d\td%x\t"\
	,CntSend,CntSendedOk,valuesonar,slave_add);

	for(i=0;i<MaxMessAdd;i++)
	{
		AllMessTemp[i]=readMlx90614ObjectTemp(AllMessAdd[i]);
			sprintf(Str0902_3_Sendapacket\
			,"%x\t%f'C\t"\
			,AllMessAdd[i],AllMessTemp[i]);
		strcat(Str0902_3_Send,Str0902_3_Sendapacket);
	}
	strcat(Str0902_3_Send,"\r\n");

	

//	float temp1,temp2,temp3;
	

	 
//	temp1=readMlx90614ObjectTemp(0xa0);		
//	temp2=readMlx90614ObjectTemp(0xb0);	
//	temp3=readMlx90614ObjectTemp(0xc0);	
		
//	sprintf(Str0902_3_Send\
//	,"%d\t%d\t%4d\t%f'C\t%f'C\t%f'C\r\n"\
//	,CntSend,CntSendedOk,valuesonar,temp1,temp2,temp3);

	printf("%s",Str0902_3_Send);
	sendBroadcastSignal(Str0902_3_Send);

	CntSend++;			
}
#endif // end 	BTESTDATE0902_2messadd














#if EXAMPLEDATE0902==BTESTDATE0902_3_send
void sendBroadcastSignal(char data[])
{
  

              paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
              paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
              paq_sent->MY_known=0;
              paq_sent->packetID=0x52;  //Think about changing it each time you send
              paq_sent->opt=0;
              xbee802.hops=0;
              xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
              xbee802.setDestinationParams(paq_sent, "000000000000FFFF", data, MAC_TYPE, DATA_ABSOLUTE);
              xbee802.sendXBee(paq_sent);
            
              if( !xbee802.error_TX )
              {
                //XBee.println("ok");
              }
              free(paq_sent);
              paq_sent=NULL;
}

static char LastSecond=0;
void setup()
{
	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	delay_ms(200);
	serialWrite('2', 1);
	delay_ms(2000);

	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	serialWrite('b', 1);
	delay_ms(200);
	xbee802.ON();
	xbee802.getMacMode();
	printf("getMacMode=%x\t\t",xbee802.macMode);
	xbee802.getChannel();
	printf("firstgetchannel=%x ",xbee802.channel);
	RTCbianliang.ON(); 
	RTCbianliang.begin();
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
	RTCbianliang.getTime();
	printf("\r\n  year=%d ",RTCbianliang.year) ;//串口2打印
	printf(" month=%d ",RTCbianliang.month);
	printf(" date=%d ",RTCbianliang.date) ;
	printf(" hour=%d ",RTCbianliang.hour);
	printf(" min=%d ",RTCbianliang.minute) ;
	printf(" sec=%d ",RTCbianliang.second);
}
static char Str0902_3_Send[200]="123";
static long Num0902_3_Send=0;
//基本上1秒发一次数据包
void loop()
{
	while(1)
	{
		RTCbianliang.getTime();
		if(RTCbianliang.second!=LastSecond)
		{
			LastSecond=RTCbianliang.second;
			break;
		}
		else
		{
			delay_ms(10);
		}
	}
	printf("\r\n  year=%d ",RTCbianliang.year) ;//串口2打印
	printf(" month=%d ",RTCbianliang.month);
	printf(" date=%d ",RTCbianliang.date) ;
	printf(" hour=%d ",RTCbianliang.hour);
	printf(" min=%d ",RTCbianliang.minute) ;
	printf(" sec=%d ",RTCbianliang.second);

	Num0902_3_Send++;	  //if(Num0902_3_Send>=16) Num0902_3_Send=0;
	sprintf(Str0902_3_Send,"%10d\t %2d\t%2d:%2d:%2d\t"
	,Num0902_3_Send
	,RTCbianliang.date
	,RTCbianliang.hour
	,RTCbianliang.minute
	,RTCbianliang.second);
	sendBroadcastSignal(Str0902_3_Send);
	printf("send%10d",Num0902_3_Send);
//	RTCbianliang.getTime();
//	printf("stamp<%s> ",RTCbianliang.timeStamp);
	delay_ms(100);
}
#endif // end BTESTDATE0902_3_send

#if EXAMPLEDATE0902==BTESTDATE0902_3_rec
static unsigned long NumRec=0;
static unsigned long RecSdName=9999;
static char FlagSdExist=0;//0 认为没有SD卡，1认为有SD卡
static char StrRecSdName[20]="name0000.txt";
static char StrSdRecNum[40];
static  uint8_t LenStrSdRecNum;
  static int32_t OffsetFileRecSd=0;
//static char StrRecSdName[20]="name0000.txt";

void handle_received_data0902()
{
	unsigned char k;
	unsigned char flagled=0;
//	unsigned char kmax;
//	unsigned char i=0;
	
	  // Waiting the answer
  //previous=millis();
//  while( (millis()-previous) < 20000 )
	{
		if( XBee.available() )
		{	
			//printf("at=%x ",Flag1ms);
			xbee802.treatData();
			if( !xbee802.error_RX )
			{
				// Sending answer back
				while(xbee802.pos>0)
				{				
					//printf("The whole data is =%s",xbee802.packet_finished[xbee802.pos-1]->data); 
					//printf("seq %d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data); 
					printf("%d\t",NumRec); 
					printf("RSSI: =%x",xbee802.packet_finished[xbee802.pos-1]->RSSI);
//					printf("len: =%x",xbee802.packet_finished[xbee802.pos-1]->data_length); 
//					for(k=0;k<xbee802.packet_finished[xbee802.pos-1]->data_length;k++)
//						printf(" %2x",xbee802.packet_finished[xbee802.pos-1]->data[k]);
					for(k=6;(k<xbee802.packet_finished[xbee802.pos-1]->data_length);k++)
						printf("%c",xbee802.packet_finished[xbee802.pos-1]->data[k]);
					printf("\r\n");
					

				
//					//这个是发送板子的mac地址
//					printf("smac=%02x%02x%02x%02x  %02x%02x%02x%02x \r\n"\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[0]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[1]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[2]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[3]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[0]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[1]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[2]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[3]\
//					);




					if(FlagSdExist==1)
					{
					sprintf(StrSdRecNum,"\r\n%x rssi=%x",NumRec,xbee802.packet_finished[xbee802.pos-1]->RSSI);
					LenStrSdRecNum = strlen(StrSdRecNum);
//					SD.writeSD(StrRecSdName, (const char*)StrSdRecNum, \
//					OffsetFileRecSd, LenStrSdRecNum);

					writeSDc(StrRecSdName, (const char*)StrSdRecNum, \
					OffsetFileRecSd, LenStrSdRecNum);
					OffsetFileRecSd += LenStrSdRecNum;
//					SD.writeSD(StrRecSdName, (const char*)(xbee802.packet_finished[xbee802.pos-1]->data+6), \
//					OffsetFileRecSd, xbee802.packet_finished[xbee802.pos-1]->data_length-6);
					writeSDc(StrRecSdName, (const char*)(xbee802.packet_finished[xbee802.pos-1]->data+6), \
					OffsetFileRecSd, xbee802.packet_finished[xbee802.pos-1]->data_length-6);
					OffsetFileRecSd += xbee802.packet_finished[xbee802.pos-1]->data_length-6;

//					printf(" T%d ",timer0_overflow_count);

					}
			 		flagled= xbee802.packet_finished[xbee802.pos-1]->data[15];
					if((flagled>='0')&&(flagled<='9'))
					{
						flagled = flagled-'0';
						if(flagled%2==0)flagled = 2;
						else flagled =3;
					}
					else 
						flagled=0;

					free(xbee802.packet_finished[xbee802.pos-1]);
					xbee802.packet_finished[xbee802.pos-1]=NULL;
					xbee802.pos--;
					NumRec++;
					Utils.setLED(2, 1);	GPIO_SetBits(GPIOC, GPIO_Pin_6);
					delay_ms(20);
					Utils.setLED(2, 0);GPIO_ResetBits(GPIOC, GPIO_Pin_6);
					delay_ms(30);
					if(flagled==2)
					{
						Utils.setLED(0, 1);
						delay_ms(100);
						Utils.setLED(0, 0);
					}
					else if(flagled==3)
					{
						Utils.setLED(1, 1);
						delay_ms(100);
						Utils.setLED(1, 0);
					}
					else 
					{
						Utils.setLED(0, 0);Utils.setLED(1, 0);
					}
				} 

			}
			//printf("et=%x ",Flag1ms);
			printf("-");
		}
	}   
}
void setup()
{
	GPIO_InitTypeDef  GPIO_InitStructured;
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_6;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOC, &GPIO_InitStructured);


	GPIO_SetBits(GPIOC, GPIO_Pin_6);//PC6是管那个串口6多路输出的，但是我为什么要加这个在这里呢?
	Utils.initLEDs();
	monitor_offuart3TX();monitor_onuart3RX();
	muluart6init();
	muluart6choose(1);	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	delay_ms(200);
	serialWrite('2', 1);
	delay_ms(2000);

	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	serialWrite('b', 1);
	delay_ms(200);
	xbee802.ON();
	xbee802.getMacMode();
	printf("getMacMode=%x\t\t",xbee802.macMode);
	xbee802.getChannel();
	printf("firstgetchannel=%x ",xbee802.channel);
	RTCbianliang.ON(); 
	RTCbianliang.begin();
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
	RTCbianliang.getTime();
	printf("\r\n  year=%d ",RTCbianliang.year) ;//串口2打印
	printf(" month=%d ",RTCbianliang.month);
	printf(" date=%d ",RTCbianliang.date) ;
	printf(" hour=%d ",RTCbianliang.hour);
	printf(" min=%d ",RTCbianliang.minute) ;
	printf(" sec=%d ",RTCbianliang.second);

		//SD卡电源开
		printf("\n  SD poweron. ");		 
	 	SD.ON();
	
	//SD初始化，成功的话也初始化一下FAT	
		printf(" SD init. ");    
		SD.init();
		if(SD.flag!=NOTHING_FAILED)
		{
			printf("  failed!  ");
			FlagSdExist=0;
			//while(1);
		}
		else FlagSdExist=1;
	


		if(FlagSdExist==1)
		{
			sprintf(StrRecSdName,"%04x.txt",RecSdName);
			//创建temp.hex  有则删除新建之
			//if(SD.isFile(StrRecSdName)==1)
			if(isFilec(StrRecSdName)==1)
			{
				//printf(" yes. ");
				//SD.del(StrRecSdName);
				delc(StrRecSdName);				
			}
			else
			{
				//printf(" no. ");						
			}
			//SD.create(StrRecSdName);
			createc(StrRecSdName);	
		}
}

void loop()
{
	handle_received_data0902();
}
#endif // end BTESTDATE0902_3_rec

#if EXAMPLEDATE0902==BTESTDATE0902_4	   
extern uint16_t CNTEXIT;
//这个例程给出的是通过按wakeup按键唤醒 或者通过电池那边的PC1上升沿唤醒
//操作： 1）进入sleep后，按wakeup按键 , 2) 进入sleep后，短路R40来达到模拟电池复位，然后就能唤醒单片机了


void setup()
{
	PWR2745.initPower(PWR_BAT);//必须开这个电池电源，要不然PC1（负责电池电量比较2.8V的）不起作用
	PWR2745.switchesON(PWR_BAT);
	Mux_poweron();//只是为了弄一个电源正

	beginSerial(115200, PRINTFPORT); //串口
	delay_ms(1000);	//在初始化这个PC1这个外部中断要等一会，要不然，一上电就进入外部中断
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
		Utils.initLEDs();
	while(1)
	{
	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(10+CNTEXIT*200);
	CNTEXIT++;

//	printf("cnt=%d ",CNTEXIT);
//	Utils.blinkLEDs(500);
//	CNTEXIT++;
	if(CNTEXIT>5)break;
	}

	//enter sleep mode, if any interrupt(such as key PA0 put down ,or uart PRINTFPORT receive datas ) hanppen, MCU can wake
    __WFI();//进入睡眠，假如有任何中断都可以唤醒它，这里的中断比如外部中断，串口接收（接收中断开的话），定时器
	//按按键wakeup就可以唤醒单片机
	// 短路R40 即人工制造一个电池复位
	printf("I am back! hello\n");

	while(1)
	{
		printf("cnt=%d ",CNTEXIT);
		Utils.blinkLEDs(100);
		CNTEXIT++;
		if(CNTEXIT>=2000)CNTEXIT=0;
		if(CNTEXIT==40)
		{
			__WFI();		   	
		}
	}
}

void loop()
{
	;
}
#endif // end BTESTDATE0902_4

#if EXAMPLEDATE0902==BTESTDATE0902_5	   
//这个例程给出的是通过按wakeup按键唤醒 或者通过电池那边的PC1上升沿唤醒	或者闹钟复位
//操作： 1）进入sleep后，按wakeup按键 ,
//       2) 进入sleep后，短路R40来达到模拟电池复位，然后就能唤醒单片机了
//		 3) 闹钟时间到了复位


void setup()
{
	PWR2745.initPower(PWR_BAT);//必须开这个电池电源，要不然PC1（负责电池电量比较2.8V的）不起作用
	PWR2745.switchesON(PWR_BAT);

	Utils.initLEDs();

	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	

	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	printf("1");
	

	//Setting the RTC
	RTCbianliang.ON();
	RTCbianliang.begin();
	printf("2");
	//关于srttime str方式的时间格式是怎样的我不清楚，这个最好要客户确认一下
	//RTCbianliang.setTime("11:08:16:03:14:58:00");  //Sun, 12/10/04 - 05:59:07
	RTCbianliang.setTime("Sun, 12/10/04 - 05:59:07"); 
	RTCbianliang.getTime();


	printf("%s",RTCbianliang.timeStamp);
	printf("\r\n  year=%d ",RTCbianliang.year) ;//串口2打印
	printf(" month=%d ",RTCbianliang.month);
	printf(" date=%d ",RTCbianliang.date) ;
	printf(" hour=%d ",RTCbianliang.hour);
	printf(" min=%d ",RTCbianliang.minute) ;
	printf(" sec=%d ",RTCbianliang.second);


//	RTCbianliang.setAlarm1test();
	RTCbianliang.setAlarm1andOn(0,0,0,5,1);
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
	delay_ms(1000);
	printf("sleep...");
    __WFI(); //进入睡眠，假如有任何中断都可以唤醒它，这里的中断比如外部中断，串口接收（接收中断开的话），定时器
	//按按键wakeup就可以唤醒单片机
	// 短路R40 即人工制造一个电池复位
	//不用手动按键 或者短路R40，闹钟时间到了它自己也会醒来

	RTCbianliang.clearAlarm1();


	printf("heihei!");
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
}

void loop()
{
uint8_t i;
	for(i=0;i<10;i++)
	{		
		Utils.setLED(0, 1);
		delay_ms(100);
		Utils.setLED(0, 0);	
		delay_ms(500);

	}
		RTCbianliang.getTime();
		printf("%s",RTCbianliang.timeStamp);
	 	printf("sl...");
    __WFI();

	RTCbianliang.clearAlarm1();
//	uint8_t temp;
//	int flagerr;
//	while(1)
//	{
//		flagerr=readExternalRTC(0x0f);
//		if(flagerr!=I2CFALSE)break;
//		printf("e%x ",flagerr);
//	}
//	temp=flagerr;
//	temp = temp & 0xfe;
//	while(1)
//	{
//		flagerr=writeExternalRTC(0x0f,temp);
//		if(flagerr!=I2CFALSE)break;
//		printf("e%x ",flagerr);
//	} 	
//	printf("b %x ",temp);

	printf("\r\nback ");
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
//	}
//	else
//	{
//		delay_ms(10);
//	}
}
#endif // end BTESTDATE0902_5


#if EXAMPLEDATE0902==BTESTDATE0902_4and5	   
void setup()
{
	Utils.initLEDs();
	Utils.setLED(1, 1);
	delay_ms(100);
	Utils.setLED(1, 0);	
	delay_ms(100);

	//为了监控电源的PC1有作用，把对应的芯片使能
	GPIO_InitTypeDef  GPIO_InitStructured;
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_7;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructured);


	GPIO_SetBits(GPIOA, GPIO_Pin_7);
	Utils.setLED(0, 1);
	delay_ms(100);
	Utils.setLED(0, 0);	

	Mux_poweron();//只是为了弄一个电源正
	monitor_offuart3TX();monitor_onuart3RX();	
	beginSerial(115200, 1); chooseuartinterrupt(1);
	printf("1");
	printf("2");	

//		xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
//	xbee802.ON();
			Utils.setLED(1, 1);
	//Setting the RTC
	RTCbianliang.ON();
	RTCbianliang.begin();
	RTCbianliang.setTime("11:08:16:03:14:58:00");
	RTCbianliang.getTime();


	printf("%s",RTCbianliang.timeStamp);
	printf("\r\n  year=%d ",RTCbianliang.year) ;//串口2打印
	printf(" month=%d ",RTCbianliang.month);
	printf(" date=%d ",RTCbianliang.date) ;
	printf(" hour=%d ",RTCbianliang.hour);
	printf(" min=%d ",RTCbianliang.minute) ;
	printf(" sec=%d ",RTCbianliang.second);

RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR, ENABLE);
//PWR_BackupAccessCmd(ENABLE);	

//	for(int i=0;i<1000;i++)
//	{
//		Utils.setLED(1, 1);
//		delay_ms(1000);
//		Utils.setLED(1, 0);	
//		delay_ms(1000);
//	}
//		Timer2_Init(200,1000);
//	RTCbianliang.setAlarm1test();
//	RTCbianliang.setAlarm1asAwake(0,0,0,5,1);
  	RTCbianliang.setAlarm1andOn(0,0,0,5,1);
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
//	setPowerEnoughAsAwake();
	for(int i=0;i<3;i++)
	{		
		Utils.setLED(2, 1);
		delay_ms(1000);
		Utils.setLED(2, 0);	
		delay_ms(1000);

	}
	printf("sleep...");
    __WFI();


//	RTCbianliang.clearAlarm1();


	printf("heihei!");
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
}

void loop()
{
uint8_t i;
	for(i=0;i<10;i++)
	{		
		Utils.setLED(0, 1);
		delay_ms(100);
		Utils.setLED(0, 0);	
		delay_ms(500);

	}
		RTCbianliang.getTime();
		printf("%s",RTCbianliang.timeStamp);
	 	printf("sl...");
    __WFI();

//	RTCbianliang.clearAlarm1();
//	uint8_t temp;
//	int flagerr;
//	while(1)
//	{
//		flagerr=readExternalRTC(0x0f);
//		if(flagerr!=I2CFALSE)break;
//		printf("e%x ",flagerr);
//	}
//	temp=flagerr;
//	temp = temp & 0xfe;
//	while(1)
//	{
//		flagerr=writeExternalRTC(0x0f,temp);
//		if(flagerr!=I2CFALSE)break;
//		printf("e%x ",flagerr);
//	} 	
//	printf("b %x ",temp);

	printf("back\r\n");
	RTCbianliang.getTime();
	printf("%s",RTCbianliang.timeStamp);
//	}
//	else
//	{
//		delay_ms(10);
//	}
}
#endif // end BTESTDATE0902_4and5



#if EXAMPLEDATE0902==BTESTDATE0902_6	   
void setup()
{


}

void loop()
{
	;
}
#endif // end BTESTDATE0902_6






