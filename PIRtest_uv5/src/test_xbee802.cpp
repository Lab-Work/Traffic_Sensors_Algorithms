
#include "allboardinc.h"

#define BTESTXBEEMULCHOOSE 1  //根据输入的要求发送还是接受还是转发数据
#define BTESTXBEESENDFASTBROADCAST 2
#define BTESTXBEESENDFASTUNICAST 3
#define BTESTXBEEREC 4
#define BTESTXBEERECTHENFASTBROADCAST 6
//#define BTESTXBEEOTA 7
//#define BTESTXBEEOTAANDOTHERREC 8
#define BTESTXBEENULL 0

#define EXAMPLEXBEE BTESTXBEENULL//



#if  EXAMPLEXBEE==BTESTXBEEMULCHOOSE

 static char destMacAddagain[17]="0013A20040905D58"; // add the mac address for the sink
// static char destMacAddagain[17]="0013A200409EDCB6"; // add the mac address for the sink
static char destMacAddagainA[17]="0013A20040710845";
static char destMacAddagainB[17]="0013A20040905D58";
static char destMacAddagainC[17]="0013A200409EDCB6";


  
packetXBee* paq_sentagain;

static  uint8_t destinationagain[8];
static char  testdata[500]="test meg!";
static char strdata2[30]="";
// unsigned char CntMeg=0;

//static unsigned char FlagMacOk=0;

static uint32_t Time802_1,Time802_2,Time802_3,Time802_4; 

extern uint32_t Flag1ms;

extern unsigned char FlagTimeLinshi;
extern uint32_t TimemsArray[];
extern uint32_t TMsArray2[];
static unsigned long NumRec=0;

//extern unsigned char FlagXbeeLinshi  ;

static unsigned long CountSendBroadcast=0;

static unsigned long RecSdName=0;//名字暂时用十六进制数据代替
static unsigned int  StepSdName=1;
static unsigned char FlagSdNameNew=1;
static char StrRecSdName[20]="name0000.txt";
 static int32_t OffsetFileRecSd=0;


void setupagain()
{
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);
		delay_ms(200);
		Utils.setLED(2, 0);
		delay_ms(200);
	}

  beginSerial(115200, 1);
  delay_ms(200);
  serialWrite('2', 1);
  delay_ms(2000);
  destMacAddagain[16]='\0';
  xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
  serialWrite('b', 1);
  delay_ms(200);
  xbee802.ON();
  //delay_ms(300);
  
  RTCbianliang.ON(); 
  
  // monitor_onuart3TX();monitor_offuart3RX();delay_ms(300);
   xbee802.getChannel();
  printf("firstgetchannel=%x ",xbee802.channel);
  //delay_ms(300);
  //delay_ms(300);

//   xbee802.setChannel(0x0D);
  //delay_ms(300);
 // monitor_offuart3TX();monitor_onuart3RX();	delay_ms(300);

   xbee802.getChannel();
  //delay_ms(300);
  printf("secondgetchannel=%x ",xbee802.channel);

//   xbee802.getChannel();
//  printf("3ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("4ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("5ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("6ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("7ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("8ch=%x ",xbee802.channel);
  
  if( !xbee802.error_AT ) XBee.println("Channel set OK");
  else XBee.println("Error while changing channel");
  //delay_ms(300);

  xbee802.getOwnMac();
  printf("macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);

//  xbee802.getOwnMac();
//  printf(" 2macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 22macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 23macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 24macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMac();
//  printf(" 25macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 26macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMac();
//  printf(" 27macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 3macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 4macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMacLow();
//  printf(" 5macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMacLow();
//  printf(" 6macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 7macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  printf(" 8macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);

  if(xbee802.sourceMacHigh[1]==0x13)
  {
  	//FlagMacOk=1;
	sprintf(testdata,"mac=%2x%2x%2x%2x  %2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]);

	printf("%s",testdata);
  }
//  printf("set time");
//  RTCbianliang.setTime("09:10:20:03:17:35:00");printf("get time");
//  delay_ms(90);	  //printf( RTCbianliang.getTime());
  //RTCbianliang.getTime(); 
  printf("over");
  //while(1)
  {
  GPIO_SetBits(GPIOC, GPIO_Pin_13);
  xbee802.setOwnNetAddress(0xFF,0xFF); 
  GPIO_ResetBits(GPIOC, GPIO_Pin_13);
  xbee802.setOwnNetAddress(0xFF,0xFF);
  }
  printf("Own ok");
//  FlagXbeeLinshi=1;
}

void broadcastSignalagain(char data[])
{  
  paq_sentagain=(packetXBee*) calloc(1,sizeof(packetXBee));
  paq_sentagain->mode=BROADCAST;  // BroadCast; you need to update everyone !
  paq_sentagain->MY_known=0;
  paq_sentagain->packetID=0x52;  //Think about changing it each time you send
  paq_sentagain->opt=0;
  xbee802.hops=0;
  xbee802.setOriginParams(paq_sentagain, "5678", MY_TYPE); // Think about this in the future as well
  xbee802.setDestinationParams(paq_sentagain, "000000000000FFFF", data, MAC_TYPE, DATA_ABSOLUTE);
  //  printf("setdes ");	//delay_ms(500);
  Time802_1 = Flag1ms;
  xbee802.sendXBee(paq_sentagain);
  //  printf("sendxbee ");//	delay_ms(500);
  Time802_2 = Flag1ms;
  CountSendBroadcast++;
  printf("%d\t",CountSendBroadcast);
  if( !xbee802.error_TX )
  {
    printf(" ok\r\n");//	delay_ms(300);
  }
  else   
  {
    printf("WRG\r\n");//	delay_ms(300);
  }	   
  Time802_3 = Flag1ms;
  free(paq_sentagain);
  paq_sentagain=NULL; 
  Time802_4 = Flag1ms;           
}
void sendUnicastSignalagain(char data[],char destMacAdd[17])
{
 // you might get the size here and then call the fragmentation function upon needing that
 // you also have to call the fuunction to check if the channel is Free"I think this is being done by default !" 
  //printf("%s",destMacAdd);
  paq_sentagain=(packetXBee*) calloc(1,sizeof(packetXBee)); 
  paq_sentagain->mode=UNICAST;  // BroadCast; you need to update everyone !
  paq_sentagain->MY_known=0;
  paq_sentagain->packetID=0x52;  //Think about changing it each time you send 
  paq_sentagain->opt=0; 
  xbee802.hops=0;
  xbee802.setOriginParams(paq_sentagain, "5678", MY_TYPE); // Think about this in the future as well
  xbee802.setDestinationParams(paq_sentagain, destMacAdd, data, MAC_TYPE, DATA_ABSOLUTE);
  xbee802.sendXBee(paq_sentagain);
  CountSendBroadcast++;
  printf("%d\t",CountSendBroadcast);
  if( !xbee802.error_TX )
  {
    printf(" ok\r\n");//	delay_ms(300);
  }
  else   
  {
    printf("WRG\r\n");//	delay_ms(300);
  }
  free(paq_sentagain);
  paq_sentagain=NULL;
  
}
extern volatile unsigned long timer0_overflow_count;
extern    uint8_t MemoryArray[MAX_PARSE];
char StrSdRecNum[40];
uint8_t LenStrSdRecNum;
extern	int rx_buffer_head3;
extern	int rx_buffer_tail3;
void handle_received_dataagain()
{
	unsigned char k;
	unsigned char kmax;
	unsigned char i=0;
	
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
//					printf("RSSI: =%x",xbee802.packet_finished[xbee802.pos-1]->RSSI);
//					printf("len: =%x",xbee802.packet_finished[xbee802.pos-1]->data_length); 
//					for(k=0;k<xbee802.packet_finished[xbee802.pos-1]->data_length;k++)
//						printf(" %2x",xbee802.packet_finished[xbee802.pos-1]->data[k]);
					for(k=16;(k<xbee802.packet_finished[xbee802.pos-1]->data_length)&&(k<20);k++)
						printf("%c",xbee802.packet_finished[xbee802.pos-1]->data[k]);
					printf("\t");
					for(k=20;(k<xbee802.packet_finished[xbee802.pos-1]->data_length)&&(k<26);k++)
						printf("%c",xbee802.packet_finished[xbee802.pos-1]->data[k]);
					printf(" H%d L%d",rx_buffer_head3,rx_buffer_tail3);
					printf("t=%d ",timer0_overflow_count);
					
				
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




					if(FlagSdNameNew==1)
					{
						sprintf(StrRecSdName,"%04x%04d.txt",RecSdName,StepSdName);
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
						FlagSdNameNew=0;
					}
					sprintf(StrSdRecNum,"\r\n%x ",NumRec);
					LenStrSdRecNum = strlen(StrSdRecNum);
//					SD.writeSD(StrRecSdName, (const char*)StrSdRecNum, \
//					OffsetFileRecSd, LenStrSdRecNum);
					printf("t=%d ",timer0_overflow_count);
					writeSDc(StrRecSdName, (const char*)StrSdRecNum, \
					OffsetFileRecSd, LenStrSdRecNum);
					OffsetFileRecSd += LenStrSdRecNum;
					printf("d=%d ",timer0_overflow_count);
//					SD.writeSD(StrRecSdName, (const char*)(xbee802.packet_finished[xbee802.pos-1]->data+6), \
//					OffsetFileRecSd, xbee802.packet_finished[xbee802.pos-1]->data_length-6);
					writeSDc(StrRecSdName, (const char*)(xbee802.packet_finished[xbee802.pos-1]->data+6), \
					OffsetFileRecSd, xbee802.packet_finished[xbee802.pos-1]->data_length-6);
					OffsetFileRecSd += xbee802.packet_finished[xbee802.pos-1]->data_length-6;

					if((FlagSdNameNew==0)&&(Flag1ms>((uint32_t)StepSdName*3600000)))
					{
						StepSdName++;
						FlagSdNameNew=1;OffsetFileRecSd=0;
					}
					printf(" T%d ",timer0_overflow_count);
					printf("\r\n");
			 
					free(xbee802.packet_finished[xbee802.pos-1]);
					xbee802.packet_finished[xbee802.pos-1]=NULL;
					xbee802.pos--;
					NumRec++;
				} 

			}
			//printf("et=%x ",Flag1ms);
			printf("-");
		}
	}   
}
static uint8_t RecBuffer[500];
static uint16_t RecLenWant=490;
static uint32_t RecMeSeq=0;

static void handle_received_datame()
{
	uint8_t oldlen=190;
//	int recnum;
	uint8_t flagerr=0;
	long i;
	long offset =0;
	long sum=0;
	long k=1;
	//if( (recnum=XBee.Rx80available())>0 )
	//{
//		printf("num=%d ",recnum);

		RecLenWant = oldlen;
		//printf("RecLenWant=%d ",RecLenWant);
		flagerr=xbee802.findRxframe((char *)RecBuffer, &RecLenWant);
		if(flagerr==0)
		{
			printf("rev:seq=%d ",RecMeSeq);
			
//			printf("RecLenWant=%d ",RecLenWant);
			//打印下收到的数据,含有包头，地址，校验位等
//			for(i=0;i<RecLenWant;i++)
//				printf("%2x ",RecBuffer[i]);

			//打印下收到的数据,含有包头，地址，校验位等
//			for(i=0;i<RecLenWant;i++)
//				printf("%c",RecBuffer[i]);
			if(RecLenWant>14)
			{
				// 打印下收到的数据关键值，也就是去掉包头，地址那些，是纯数据
				for(i=14;i<(RecLenWant-1);i++)
					printf("%c",RecBuffer[i]);								
			}


			RecMeSeq++;
										
		}
		else
		{
			//如果没有收到有用的数据包，那打印下出错信息
//			printf("flagerr=%d ",flagerr);
//			if(flagerr==2)printf("H=%dT=%d ",Rx80BufferHead,Rx80BufferTail);

			//printf("n");

		}  
}

static void handle_recandsend_datame(char fromMacAdd[17], char destMacAdd[17])
{
	uint8_t oldlen=190;
//	int recnum;
	uint8_t flagerr=0;
	long i;
	long offset =0;
	long sum=0;
	long k=1;
	//if( (recnum=XBee.Rx80available())>0 )
	//{
//		printf("num=%d ",recnum);

		RecLenWant = oldlen;
		//printf("RecLenWant=%d ",RecLenWant);
		flagerr=xbee802.findRxframe((char *)RecBuffer, &RecLenWant);
		if(flagerr==0)
		{
			printf("rev:seq=%d ",RecMeSeq);

			
//			printf("RecLenWant=%d ",RecLenWant);
			//打印下收到的数据,含有包头，地址，校验位等
//			for(i=0;i<RecLenWant;i++)
//				printf("%2x ",RecBuffer[i]);

			//打印下收到的数据,含有包头，地址，校验位等
//			for(i=0;i<RecLenWant;i++)
//				printf("%c",RecBuffer[i]);
			if(RecLenWant>14)
			{
				// 打印下收到的数据关键值，也就是去掉包头，地址那些，是纯数据
				for(i=14;i<(RecLenWant-1);i++)
					printf("%c",RecBuffer[i]);								
			}

			sprintf(testdata,"seq:%d ",RecMeSeq); 
			for(i=20;i<(RecLenWant-1);i++)
			{
				sprintf(strdata2,"%c",RecBuffer[i]); 						
				strcat(testdata,strdata2);			
			}	
			sendUnicastSignalagain(testdata,destMacAdd);
			RecMeSeq++;



										
		}
		else
		{
			//如果没有收到有用的数据包，那打印下出错信息
//			printf("flagerr=%d ",flagerr);
//			if(flagerr==2)printf("H=%dT=%d ",Rx80BufferHead,Rx80BufferTail);

			//printf("n");

		}  
}
void tihuan(uint8_t * ostr, uint8_t * aimstr, uint8_t begin, uint8_t len)
{
	uint8_t i;
	for(i=0;i<len;i++)
	{
		ostr[begin+i]=aimstr[i];
	}
}
void handle_recandsend_dataagain(char fromMacAdd[17], char destMacAdd[17])
{
	long k=1,k2;
	unsigned char kmax;
	unsigned char i=0;
	uint8_t datalen=0;
	uint8_t datayiyang=0;
	uint32_t sendnum=0;

	{
		if( XBee.available() )
		{
			xbee802.treatData();
			if( !xbee802.error_RX )
			{
				// Sending answer back
				while(xbee802.pos>0)
				{
				
					//printf("The whole data is =%s",xbee802.packet_finished[xbee802.pos-1]->data); 
					//printf("seq %d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data);
					//sprintf(testdata,"seq:%d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data); 


//					printf("seq %d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data); 
//					printf("RSSI: =%x\r\n",xbee802.packet_finished[xbee802.pos-1]->RSSI);
//					printf("len: =%x\r\n",xbee802.packet_finished[xbee802.pos-1]->data_length); 

					

					sprintf(testdata,""); 
					for(k=0;k<xbee802.packet_finished[xbee802.pos-1]->data_length;k++)
					{
//						printf(" %2x",xbee802.packet_finished[xbee802.pos-1]->data[k]);
						if(k>5)
						{
							sprintf(strdata2,"%c",xbee802.packet_finished[xbee802.pos-1]->data[k]); 
													
							strcat(testdata,strdata2);
						}
												
					}
					testdata[k]=0x00;

					datalen=strlen(testdata);
					for(k=0;k<(datalen-9);k=k+10)
					{
						datayiyang=testdata[k+2];
						for(k2=3;k2<10;k2++)
							if(testdata[k+k2]!=datayiyang)break;
						if(k2==10)
						{
							sprintf(strdata2,"%02x%02x%06x",xbee802.sourceMacLow[2],xbee802.sourceMacLow[3],NumRec);
							tihuan((uint8_t *)testdata,(uint8_t *)strdata2,k,10);
							break;
						}
					}

//					//这个是发送板子的mac地址
//					printf("smac=%2x%2x%2x%2x  %2x%2x%2x%2x "\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[0]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[1]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[2]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSH[3]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[0]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[1]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[2]\
//					,xbee802.packet_finished[xbee802.pos-1]->macSL[3]\
//					);



					free(xbee802.packet_finished[xbee802.pos-1]);
					xbee802.packet_finished[xbee802.pos-1]=NULL;
					xbee802.pos--;
					NumRec++;


//					sprintf(testdata,"c\t%7d\t1\t%7d\t2\t%7d\t3\t%7d\t4\t%7d\td\t%7d\t%02x%02x%02x%02x%02x%02x%02x%02x\r\n"\
//					,sendnum\
//					,TMsArray2[1],TMsArray2[2],TMsArray2[3],TMsArray2[4],TMsArray2[14]\
//					,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//					,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]
//					);

					

					sendUnicastSignalagain(testdata,destMacAdd);
					//sendUnicastSignalagain(xbee802.packet_finished[xbee802.pos-1]->data,destMacAdd);
					sendnum++;

				} 
		
			}
		}
	}   
}



void testxbee802moteagain(void)
{
//	long diff;
//	init();

uint8_t i=0;
	setupagain();
	uint32_t sendnum=0;
	
	for (;;)
	{
		//Time802_1 = Flag1ms;
//		if(i==0){monitor_onuart3TX();monitor_offuart3RX();}
//		else {monitor_offuart3TX();monitor_onuart3RX();}
//		i++; if(i>=2)i=0;  	
			//,Time802_1,Time802_2,Time802_3,Time802_4\
		//Time802_2 = Flag1ms;																	  //8901234567aaaaa12345
//		sprintf(testdata,"c\t%7d\t1\t%7d\t2\t%7d\t3\t%7d\t4\t%7d\tb\t%7d\tc\t%7d\td\t%7d\t"\
//		,sendnum\
//		,TMsArray2[1],TMsArray2[2],TMsArray2[3],TMsArray2[4],TMsArray2[12],TMsArray2[13],TMsArray2[14]
//		);



//		sprintf(testdata,"c\t%7d\t1\t%7d\t2\t%7d\t3\t%7d\t4\t%7d\td\t%7d\t%02x%02x%02x%02x%02x%02x%02x%02x\r\n"\
//		,sendnum\
//		,TMsArray2[1],TMsArray2[2],TMsArray2[3],TMsArray2[4],TMsArray2[14]\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]
//		);

		sprintf(testdata,"c\t%7d\r\n"\
		,sendnum\
		);

//		sprintf(testdata,"c\t%6d\r\n"\
//		,sendnum
//		);


		//Time802_3 = Flag1ms;

		////广播发送数据
		broadcastSignalagain(testdata);
		//Time802_4 = Flag1ms;

		////单播
		//sendUnicastSignalagain(testdata,destMacAddagain);


		//处理接收的数据，接收的打印出来
		//handle_received_dataagain();
		
		//用我自己写的接收程序来接收xbee内容
		//handle_received_datame();


		//转包
		//handle_recandsend_dataagain(destMacAddagainA,destMacAddagainC);
		
		//用我自己写的接收程序来接收xbee内容 然后转包，这里发送用waspmote的
		//handle_recandsend_datame(destMacAddagainA,destMacAddagainC);


		sendnum++; //
		
		//diff = millis() -previous;
		//printf("send packet spend %d ms\r\n",diff);
		//delay(2000);
		//delay_ms(500);
		//delay_ms(160);////
		//delay_ms(60);//80		
		//delay_ms(50);//70
		delay_ms(40);//60//
		//delay_ms(30);//50
		//delay_ms(20);//40
		//delay_ms(10);//30
	}
}
static unsigned char FlagKeyPressed=0;
static unsigned char FlagMode=0;//0send  1 rec    2 recandsend
static unsigned long AimMacLAddress=0xffff;
static unsigned long IntervalTimeMs=100;
static unsigned char FlagXbeePro=0;
static unsigned long TimeTotalSend=0;
static unsigned long CountTotalSend=0;

void initkey(void)
{
 	GPIO_InitTypeDef  GPIO_InitStructure;//GPIO初始化结构体
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);//外设时钟使能
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;	//
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
  	//GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	//GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
  	//GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructure);	
}
void setupinteract()
{
	unsigned long biaohao=0xff;
	uint8_t flagerr=0xff;
	int getachar; int readachar=0x70;
	uint64_t fsize;

	//for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);
		delay_ms(50);
		Utils.setLED(2, 0);
		delay_ms(50);
	//}
	initkey();
	if(GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_0)==0)//没有按下按键
	{
		for(uint16_t i=0;i<10;i++){
			Utils.setLED(1, 1);
			delay_ms(50);
			Utils.setLED(1, 0);
			delay_ms(50);
		}	
	}
	else//按下按键
	{
		for(uint16_t i=0;i<10;i++){
			Utils.setLED(0, 1);
			delay_ms(10);
			Utils.setLED(0, 0);
			delay_ms(100);
		}
		FlagKeyPressed=1;
		monitor_offuart3TX();monitor_onuart3RX();			
	}
  beginSerial(115200, 1); chooseuartinterrupt(1);
  delay_ms(200);
  serialWrite('3', 1);
  delay_ms(2000);
  destMacAddagain[16]='\0';
  xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
  serialWrite('b', 1);
  delay_ms(200);
  xbee802.ON();
  //delay_ms(300);



	flagerr=xbee802.encryptionMode(0); 
	//flagerr=xbee802.getencryptionMode();
	if(flagerr==0)
	{
		printf("set eemode ok ");
	}
	else 
	{
		printf("err=%x ",flagerr);
	}
//
//	flagerr=xbee802.setLinkKey("012345679ABCDEF");
//	//flagerr=xbee802.setLinkKey("012345679ABCDEF");
//	if(flagerr==0)
//	{
//		printf(" setkeyok ");	
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//	xbee802.writeValues();

  if(FlagKeyPressed==1)
  {

//
//	printf("get macmode");
//	flagerr=xbee802.getMacMode();
//	if(flagerr==0){
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//	}
//	else 		printf("err=%x ",flagerr);


//	printf("set macmode=2 ");	
//	flagerr=xbee802.setMacMode(2);
//	if(flagerr==0){
//	printf("ok ",xbee802.macMode);
//	}
//	else 		printf("err=%x ",flagerr);
//	//xbee802.macMode=4;


//	printf(" I wish ee=0 ");
//	xbee802.encryptionMode(0); xbee802.encryptMode=1;
//	flagerr=xbee802.getencryptionMode();
//	if(flagerr==0)
//	{
//		printf("get eemode =%x\t\t ",xbee802.encryptMode);
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//
//
//	printf(" I wish ee=1 ");
//
//	flagerr=xbee802.encryptionMode(1); 
//	if(flagerr==0)
//	{
//		printf(" seteeok ");	
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//	xbee802.encryptMode=0;
//	flagerr=xbee802.getencryptionMode();
//	if(flagerr==0)
//	{
//		printf("get eemode =%x\t\t ",xbee802.encryptMode);
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//
//	flagerr=xbee802.setLinkKey("012345679ABCDEF");
//	if(flagerr==0)
//	{
//		printf(" setkeyok ");	
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//	xbee802.linkKey[0]='5';xbee802.linkKey[15]='6';
//	flagerr=xbee802.getLinkKey( );
//	if(flagerr==0)
//	{
//		printf(" getkey0 1 15=%x %x %x ",xbee802.linkKey[0],xbee802.linkKey[1],xbee802.linkKey[15]);	
//	}
//	else 
//	{
//		printf("err=%x ",flagerr);
//	}
//	xbee802.writeValues();
  	printf("what mode?\r\n");
  	printf("0 send\r\n");
  	//printf("1 unicast\r\n");
 	printf("1 rec\r\n");
  	printf("2 recthensend\r\n");
//	while(1)
//	{
//		readachar=serialRead(PRINTFPORT);
//		if(readachar<0);
//		else break;
//	}
//	printf("readachar=%x ",readachar);
//	delay_ms(10000); delay_ms(10000);delay_ms(10000);delay_ms(10000);delay_ms(10000);
//	printf("wait ",readachar);
//	delay_ms(10000); delay_ms(10000);delay_ms(10000);delay_ms(10000);delay_ms(10000);
//	printf("wait ",readachar);
//	delay_ms(10000); delay_ms(10000);delay_ms(10000);delay_ms(10000);delay_ms(10000);
//	printf("wait ",readachar);
// 	delay_ms(10000); delay_ms(10000);delay_ms(10000);delay_ms(10000);delay_ms(10000);
//	printf("wait ",readachar);

//	getachar=getchar();
//	printf("getachar=%x ",getachar);

	scanf("%d",&biaohao);
	
	if(biaohao==0)
	{	
		FlagMode=biaohao;
		printf(" sendmode.\r\n");	
	}
	else if(biaohao==1)
	{
		FlagMode=biaohao;
		printf(" recmode.\r\n");		
	}
	else if(biaohao==2)
	{
		FlagMode=biaohao;
		printf(" recthensendmode.\r\n");						
	}
	else
	{
		printf("errzhi biaohao=%x ,we set biaohao=0\r\n",biaohao);
		FlagMode=0;
		printf(" sendmode.\r\n");
	}

  	printf("what xbee pro?\r\n");
  	printf("0 A macmode=0\r\n");
 	printf("1 B macmode=2\r\n");
  	printf("2 C macmode=1\r\n");
  	printf("3 D macmode=3\r\n");
 	printf("4 AE\r\n");
  	printf("5 BE\r\n");
  	printf("6 CE\r\n");
 	printf("7 DE\r\n");
	scanf("%d",&biaohao);
	if((biaohao<8))//&&(biaohao>=4)
		FlagXbeePro=biaohao;
	else
	{
		printf("errzhi biaohao=%x ,we set biaohao=4\r\n",biaohao);
		biaohao=0;
		FlagXbeePro=0;
	}

	printf("you select ");
	switch(FlagXbeePro%4)
	{
		case 0: 
			printf("A");
			xbee802.setMacMode(0);		
			break;
		case 1: 
			printf("B");
			xbee802.setMacMode(2);		
			break;
		case 2: 
			printf("C");
			xbee802.setMacMode(1);		
			break;
		case 3: 
			printf("D");
			xbee802.setMacMode(3);		
			break;
	}
	printf("\t"); xbee802.macMode=4;
	xbee802.getMacMode();
	printf("getMacMode=%x\t\t",xbee802.macMode);

	
//  	printf("0 no E\r\n");
// 	printf("1 E\r\n");
//	scanf("%d",&biaohao);
//	if(biaohao<2);
//		//FlagXbeePro=biaohao;
//	else
//	{
//		printf("errzhi biaohao=%x ,we set biaohao=0\r\n",biaohao);
//		biaohao=0;
//	}
//	if(biaohao==0)
//	{
//		
//	}
//	else
//	{
//	
//	}




	if((FlagMode==0)||(FlagMode==2))
	{
		printf("please enter the rec macL(hex). if num==0xffff, it is broadcast mode\r\n");	
		scanf("%x",&biaohao);
		printf("macL=%x\r\n",biaohao);
		AimMacLAddress=	biaohao;	
	}

	if(FlagMode==0)
	{
		printf("please enter the interval time(dec). unit is millisecond\r\n");	
		scanf("%d",&biaohao);
		printf("interval=%dms\r\n",biaohao);
		IntervalTimeMs=	biaohao;
		
		printf("if dont stop please enter 0,else enter 1\r\n");	
		scanf("%d",&biaohao);
		if((biaohao==0)||(biaohao==1));
		else biaohao=0;
		
		if(biaohao==1)
		{
			printf("if choose count please enter 0,else choose time enter 1\r\n");	
			scanf("%d",&biaohao);
			if((biaohao==0)||(biaohao==1));
			else biaohao=0;		
		
			if(biaohao==0)
			{
				printf(" please enter count\r\n");	
				scanf("%d",&biaohao);
				CountTotalSend=biaohao;
				printf("CountTotalSend=%d \r\n",CountTotalSend);			
			}
			else
			{
				printf(" please enter time\r\n");	
				scanf("%d",&biaohao);
				TimeTotalSend=biaohao;
				printf("TimeTotalSend=%d \r\n",TimeTotalSend);				
			}
		}			
	}
	if(FlagMode==1)
	{
		//SD卡电源开
		printf("\n  SD poweron. ");		 
	 	SD.ON();
	
	//SD初始化，成功的话也初始化一下FAT	
		printf(" SD init. ");    
		SD.init();
		if(SD.flag!=NOTHING_FAILED)
		{
			printf("  failed!  ");
			while(1);
		}
	
	//得到卡的实际能使用的所有空间
		printf(" Get disk fat size. ");
		fsize=SD.getDiskSize();
		if(SD.flag==FR_OK)
		{
			printf(" fatsize=%d KB ",(uint32_t)(fsize>>10));
		}
		else
		{
			printf(" errflag=%x ",SD.flag);
			while(1);
		}

		printf("please enter a hex data as recname such as: AB2E \r\n");	
		scanf("%x",&biaohao);
		RecSdName = biaohao&0x0000ffff;
		printf(" now recname= %04x",RecSdName);

	}
  }


  
  RTCbianliang.ON(); 
  //
   //monitor_onuart3TX();monitor_offuart3RX();delay_ms(300);
   xbee802.getChannel();
  printf("firstgetchannel=%x ",xbee802.channel);

//	xbee802.getMacMode();
//	printf("getMacMode=%x ",xbee802.macMode);
//
//	printf("set macmode=0 ");
//	xbee802.setMacMode(0);
//	//printf("getMacMode=%x ",xbee802.macMode);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//
//	printf("set macmode=1 ");
//	xbee802.setMacMode(1);
//	//printf("getMacMode=%x ",xbee802.macMode);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//
//	printf("set macmode=2 ");
//	xbee802.setMacMode(2);
//	//printf("getMacMode=%x ",xbee802.macMode);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//
//	printf("set macmode=3 ");
//	xbee802.setMacMode(3);
//	//printf("getMacMode=%x ",xbee802.macMode);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);





  //delay_ms(300);
  //delay_ms(300);

//   xbee802.setChannel(0x0D);
  //delay_ms(300);


   xbee802.getChannel();
  //delay_ms(300);
  printf("secondgetchannel=%x ",xbee802.channel);

//   xbee802.getChannel();
//  printf("3ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("4ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("5ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("6ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("7ch=%x ",xbee802.channel);
//
//   xbee802.getChannel();
//  printf("8ch=%x ",xbee802.channel);
  
  if( !xbee802.error_AT ) XBee.println("Channel set OK");
  else XBee.println("Error while changing channel");
  //delay_ms(300);

  xbee802.getOwnMac();
  printf("macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);

//  xbee802.getOwnMac();
//  printf(" 2macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 22macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 23macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 24macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMac();
//  printf(" 25macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMac();
//  printf(" 26macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMac();
//  printf(" 27macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 3macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 4macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMacLow();
//  printf(" 5macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//
//  xbee802.getOwnMacLow();
//  printf(" 6macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  xbee802.getOwnMacLow();
//  printf(" 7macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//
//  printf(" 8macL =%x %x %x %x  ",xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]); //delay_ms(500);
//  printf("macH =%x %x %x %x  ",xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]); //delay_ms(500);

  if(xbee802.sourceMacHigh[1]==0x13)
  {
  	//FlagMacOk=1;
	sprintf(testdata,"mac=%2x%2x%2x%2x  %2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]);

	printf("%s",testdata);
  }
//  printf("set time");
//  RTCbianliang.setTime("09:10:20:03:17:35:00");printf("get time");
//  delay_ms(90);	  //printf( RTCbianliang.getTime());
  //RTCbianliang.getTime(); 
  printf("over");
  //while(1)
  {
  GPIO_SetBits(GPIOC, GPIO_Pin_13);
  xbee802.setOwnNetAddress(0xFF,0xFF); 
  GPIO_ResetBits(GPIOC, GPIO_Pin_13);
  xbee802.setOwnNetAddress(0xFF,0xFF);
  }
  printf("Own ok");
//  FlagXbeeLinshi=1;
}
char struprzpp(char *str)
{
	uint8_t k;
	k=0;
	while(1)
	{
		if(str[k]==0x00)break;
		if((str[k]>='a')&&(str[k]<='z'))str[k]= str[k]-32;
		k++;
		if(k>200)break;
	}
	return 0;
}

extern	int rx_buffer_head3;
extern	int rx_buffer_tail3;
extern	int rx_buffer_head1;
extern	int rx_buffer_tail1;
extern	 	unsigned char rx_buffer1[RX_BUFFER_SIZE_1];
extern	 	unsigned char rx_buffer3[RX_BUFFER_SIZE_3];
const uint8_t PaiArray[]={3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3};
uint8_t PaiWeiZhi=0;
uint8_t PaiStrLen=strlen((char *)PaiArray);
void setup(void)
{
//	long diff;
//	init();

	uint8_t i=0;
	setupinteract();
	uint32_t sendnum=0;
 	char destmavaddaim[20]="0013A20040905D58";	
	unsigned long biaohao=0xff;
	int rx_buffer_head1old=0;
	int temp3;
	char flagpause=0;
    unsigned long pausetimems=100;
	uint8_t countpause=0;
	uint8_t countpauseaim=0; 	
	uint8_t countpausemin=0;
	uint8_t countpausemax=0;


	
looptestsendagain:
	if((FlagMode==0)||(FlagMode==2))
	{
		if((TimeTotalSend!=0)||(CountTotalSend!=0))
		{
			printf("go or exit?\r\n");
			printf("1 just go\t 2 exit\t 3 pause go\r\n");
			scanf("%d",&biaohao);
			if((biaohao==1)||(biaohao==2)||(biaohao==3));
			else biaohao=2;
	
			if(biaohao==2){printf("exit");return;}
			else 
			{
				Flag1ms=0;
				CountSendBroadcast=0;
			}
			if(biaohao==3)
			{	 
				flagpause=1;
				printf("enter pause time\r\n");
				scanf("%d",&biaohao);
				pausetimems = biaohao;
				printf("puasetime=%dms ",pausetimems);
				printf("enter countmin\r\n");
				scanf("%d",&biaohao);
				countpausemin = biaohao;
				printf("countpausemin=%d ",countpausemin);
				printf("enter countmax\r\n");
				scanf("%d",&biaohao);
				countpausemax = biaohao;
				printf("countpausemax=%d ",countpausemax);
				countpauseaim=countpausemin; 

			}
			printf("CountTotalSend=%d \r\n",CountTotalSend);			
			printf("TimeTotalSend=%d \r\n",TimeTotalSend);	

		}
		printf(" MacMode=%x\t\t",xbee802.macMode);
		printf("my mac=%2x%2x%2x%2x  %2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]);

		if(AimMacLAddress!=0xffff)
		{
			sprintf(destmavaddaim,"0013A200%08x",AimMacLAddress);
			//printf(" aim=%s\r\n",destmavaddaim);			
		}
		else
		{
			sprintf(destmavaddaim,"00000000%08x",AimMacLAddress);
			//printf(" aim=%s\r\n",destmavaddaim);		
		}
		struprzpp(destmavaddaim);
		printf(" aim=%s\r\n",destmavaddaim);

		printf(" IntervalTimeMs=%d ",IntervalTimeMs);




	}
	else if(FlagMode==1)
	{
	
	
	}


	


	for (;;)
	{
		//Time802_1 = Flag1ms;
//		if(i==0){monitor_onuart3TX();monitor_offuart3RX();}
//		else {monitor_offuart3TX();monitor_onuart3RX();}
//		i++; if(i>=2)i=0;  	
			//,Time802_1,Time802_2,Time802_3,Time802_4\
		//Time802_2 = Flag1ms;																	  //8901234567aaaaa12345
//		sprintf(testdata,"c\t%7d\t1\t%7d\t2\t%7d\t3\t%7d\t4\t%7d\tb\t%7d\tc\t%7d\td\t%7d\t"\
//		,sendnum\
//		,TMsArray2[1],TMsArray2[2],TMsArray2[3],TMsArray2[4],TMsArray2[12],TMsArray2[13],TMsArray2[14]
//		);



//		sprintf(testdata,"c\t%7d\t1\t%7d\t2\t%7d\t3\t%7d\t4\t%7d\td\t%7d\t%02x%02x%02x%02x%02x%02x%02x%02x\r\n"\
//		,sendnum\
//		,TMsArray2[1],TMsArray2[2],TMsArray2[3],TMsArray2[4],TMsArray2[14]\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]
//		);


//		//如果真正的数据是下面的100个字节，在测试中看到waspmote把它拆成两个包发送的
//						//  1 |  2  	  |3   4   5   6   7   8   9  10
//		sprintf(testdata,"%010x%02x%02x%06x%10x%10x%10x%10x%10x%10x%10x%10x"\
//		,TMsArray2[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3],CountSendBroadcast\
//		,0x33333333\
//		,0x44444444\
//		,0x55555555\
//		,0x66666666\
//		,0x77777777\
//		,0x88888888\
//		,0x99999999\
//		,0xaaaaaaaa
//		);


		//
						//  1 |  2  	  |3   4   5   6   7   8   9  
		sprintf(testdata,"%010x%02x%02x%06x%10x%10x%10x%10x%10x%10x%10x"\
		,Flag1ms,xbee802.sourceMacLow[2],xbee802.sourceMacLow[3],CountSendBroadcast\
		,0x33333333\
		,0x44444444\
		,0x55555555\
		,0x66666666\
		,0x77777777\
		,0x88888888\
		,0x99999999\
		);


//		sprintf(testdata,"c\t%6d\r\n"\
//		,sendnum
//		);


		//Time802_3 = Flag1ms;
		if(FlagMode==0)
		{
			if(AimMacLAddress==0xffff)
			{
				////广播发送数据		
				broadcastSignalagain(testdata);
				//Time802_4 = Flag1ms;
			}
			else
			{
				////单播
				sendUnicastSignalagain(testdata,destmavaddaim);
			}
			delay_ms(IntervalTimeMs);
			if(flagpause>0)
			{
				countpause++;
				if(countpause>countpauseaim)
				{
					countpause=0;
					countpauseaim++;
					if(countpauseaim>countpausemax)
						countpauseaim=countpausemin;
					delay_ms(pausetimems);
				}
			}
			if((CountTotalSend!=0)&&(CountSendBroadcast>CountTotalSend))goto looptestsendagain;
			if((TimeTotalSend!=0)&&(Flag1ms>TimeTotalSend))goto looptestsendagain;
		}

		else if(FlagMode==1)
		{
			//处理接收的数据，接收的打印出来
			handle_received_dataagain();
			if(rx_buffer_head1old!=rx_buffer_head1)
			{
				 rx_buffer_head1old=rx_buffer_head1;
				 if(rx_buffer_head1>5)
				 {
				 	if((rx_buffer1[rx_buffer_head1-1]=='a')&&(rx_buffer1[rx_buffer_head1-2]=='a'))
					{
							rx_buffer1[rx_buffer_head1-1]=0;
							rx_buffer1[rx_buffer_head1-2]=0;
							printf("head3=%d tail3=%d ",rx_buffer_head3,rx_buffer_tail3);
							temp3= rx_buffer_tail3;
							int tempi=0;
							for(tempi=0;tempi<200;tempi++)
							{
								printf("%2x ",rx_buffer3[temp3]);
								temp3 = (temp3+RX_BUFFER_SIZE_3 -1)%RX_BUFFER_SIZE_3;
							}
					}
				 	if((rx_buffer1[rx_buffer_head1-1]=='b'))
					{
							rx_buffer1[rx_buffer_head1-1]=0;
							printf("head3=%d tail3=%d ",rx_buffer_head3,rx_buffer_tail3);
							temp3= rx_buffer_tail3;
					}
				 	if((rx_buffer1[rx_buffer_head1-1]=='g')&&(rx_buffer1[rx_buffer_head1-2]=='g'))
					{
							rx_buffer1[rx_buffer_head1-1]=0;
							rx_buffer1[rx_buffer_head1-2]=0;
							printf("head3=%d tail3=%d ",rx_buffer_head3,rx_buffer_tail3);
							temp3= rx_buffer_tail3;
							int tempi=0;
							for(tempi=0;tempi<4000;tempi++)
							{
								printf("%2x ",rx_buffer3[temp3]);
								temp3 = (temp3+RX_BUFFER_SIZE_3 -1)%RX_BUFFER_SIZE_3;
							}
					}
				 }
			}
		}
		//用我自己写的接收程序来接收xbee内容
		//handle_received_datame();

		else if(FlagMode==2)
		{

			//转包
			handle_recandsend_dataagain(destMacAddagainA,destmavaddaim);

		}
		
		//用我自己写的接收程序来接收xbee内容 然后转包，这里发送用waspmote的
		//handle_recandsend_datame(destMacAddagainA,destMacAddagainC);


		sendnum++; //
		
		//diff = millis() -previous;
		//printf("send packet spend %d ms\r\n",diff);
		//delay(2000);
		//delay_ms(500);
		//delay_ms(160);////
		//delay_ms(60);//80		
		//delay_ms(50);//70
		//delay_ms(40);//60//
		//delay_ms(30);//50
		//delay_ms(20);//40
		//delay_ms(10);//30

		
	}
}


//void sendUnicastSignalagain(char data[],char destMacAdd[17])
//{
// // you might get the size here and then call the fragmentation function upon needing that
// // you also have to call the fuunction to check if the channel is Free"I think this is being done by default !" 
//  //printf("%s",destMacAdd);
//  paq_sentagain=(packetXBee*) calloc(1,sizeof(packetXBee)); 
//  paq_sentagain->mode=UNICAST;  // BroadCast; you need to update everyone !
//  paq_sentagain->MY_known=0;
//  paq_sentagain->packetID=0x52;  //Think about changing it each time you send 
//  paq_sentagain->opt=0; 
//  xbee802.hops=0;
//  xbee802.setOriginParams(paq_sentagain, "5678", MY_TYPE); // Think about this in the future as well
//  xbee802.setDestinationParams(paq_sentagain, destMacAdd, data, MAC_TYPE, DATA_ABSOLUTE);
//  xbee802.sendXBee(paq_sentagain);
//  CountSendBroadcast++;
//  printf("%d\t",CountSendBroadcast);
//  if( !xbee802.error_TX )
//  {
//    printf(" ok\r\n");//	delay_ms(300);
//  }
//  else   
//  {
//    printf("WRG\r\n");//	delay_ms(300);
//  }
//  free(paq_sentagain);
//  paq_sentagain=NULL;
//  
//}
void loop()
{
;
}
#endif

#if EXAMPLEXBEE==	BTESTXBEESENDFASTBROADCAST
packetXBee* paq_sent;
void setup()
{
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, 1); 
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(200);
}

void loop()
{
  paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
  paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
  paq_sent->MY_known=0;
  paq_sent->packetID=0x52;  //Think about changing it each time you send
  paq_sent->opt=0;
  xbee802.hops=0;
  xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
  xbee802.setDestinationParams(paq_sent, "000000000000FFFF", "the string of sending", MAC_TYPE, DATA_ABSOLUTE);
  xbee802.sendXBee(paq_sent);
  if( !xbee802.error_TX )
  {
    printf(" ok\r\n");//	delay_ms(300);
  }
  else   
  {
    printf("WRG\r\n");//	
  }	   
  free(paq_sent);
  paq_sent=NULL; 
  delay_ms(1000);
}
#endif //BTESTXBEESENDFASTBROADCAST


#if EXAMPLEXBEE==	BTESTXBEESENDFASTUNICAST
packetXBee* paq_sent;
long CountSendUnicast=0;
void setup()
{
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, 1); 
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(200);
}

void loop()
{	
	//char destmavaddaim[20]="0013A20040905D58";
  paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
  paq_sent->mode=UNICAST;  // BroadCast; you need to update everyone !
  paq_sent->MY_known=0;
  paq_sent->packetID=0x52;  //Think about changing it each time you send 
  paq_sent->opt=0; 
  xbee802.hops=0;
  xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
  xbee802.setDestinationParams(paq_sent, "0013A200409EDC5A", "the string of sending by unicast", MAC_TYPE, DATA_ABSOLUTE);//这里的mac字母必须是大写
  xbee802.sendXBee(paq_sent);
  CountSendUnicast++;
  printf("%d\t",CountSendUnicast);
  if( !xbee802.error_TX )
  {
    printf(" ok\r\n");//	delay_ms(300);
  }
  else   
  {
    printf("WRG\r\n");//	delay_ms(300);
  }
  free(paq_sent);
  paq_sent=NULL;

}
#endif //BTESTXBEESENDFASTUNICAST

#if EXAMPLEXBEE==BTESTXBEEREC
void setup()
{
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, 1); 
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(200);
}

void loop()
{	
	unsigned char k;
	unsigned char kmax;
	unsigned char i=0;
	

		if( XBee.available() )
		{	
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
			 
					free(xbee802.packet_finished[xbee802.pos-1]);
					xbee802.packet_finished[xbee802.pos-1]=NULL;
					xbee802.pos--;
				} 

			}
			printf("-");
		}

}
#endif //BTESTXBEEREC


//#if EXAMPLEXBEE==	BTESTXBEEOTA
//uint8_t DataFromEeprom[200]={"LIBELIUM0102030405060708090"};
//void setup()
//{
//	uint8_t datai;
//	Utils.initLEDs();
//	Utils.setLED(2, 1);
//	delay_ms(50);
//	Utils.setLED(2, 0);
//	delay_ms(50);
//
//	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
//	beginSerial(115200, 1); chooseuartinterrupt(1);
//	delay_ms(200);
//	serialWrite('5', 1);
//	delay_ms(200);
////	destMacAddagain[16]='\0';
//
//	delay_ms(200);
//	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
//	serialWrite('o', 1);
//	delay_ms(200);
//	xbee802.ON(); 	
//	monitor_offuart3TX();monitor_onuart3RX(); //放到xbee初始化前就是不能捕获数据（看不到收到的数据），也不知道为什么
//	xbee802.getChannel();
//	printf("firstgetchannel=%x ",xbee802.channel);
//	delay_ms(30);
//	xbee802.getChannel();	
//	printf("secondgetchannel=%x ",xbee802.channel);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//	xbee802.setOwnNetAddress(0xff, 0xff);
//	printf("getMY");
//	xbee802.getOwnNetAddress();
//	printf("=%x %x \t\t",xbee802.sourceNA[0],xbee802.sourceNA[1]);
//	Eeprom.ON();	
//	Eeprom.begin();
//	delay_ms(200);	
//	printf("  read eeprom from 107 len=8 ");
//	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
//	for(datai=0;datai<8;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
//	else
//	{	
//		printf("\r\n WRG  or else \r\n");
//		Eeprom.writeEEPROMStr(0xa0,107,(uint8_t *)"LIBELIUM0102030405060708090",16);
//		for(datai=0;datai<200;datai++)
//		{
//			DataFromEeprom[datai]=datai;	
//		}
//	}
//	printf(" 2th read eeprom from 107 len=8 ");
//	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
//	for(datai=0;datai<8;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
//	else 
//	{
//		printf("\r\n\r\n ********************WRONG****************** \r\n\r\n");
//	}
//
//	printf("read eeprom:");
//	Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//	for(datai=0;datai<200;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	uint8_t flag0xff=0;
//	for(datai=0;datai<200;datai++)
//	{
//		if(DataFromEeprom[datai]==0xff)
//		{
//			flag0xff=1;
//			if(datai<=33)DataFromEeprom[datai]=0x31;
//			else if(datai<=97)DataFromEeprom[datai]=0x32;
//			else if(datai<=106)DataFromEeprom[datai]=0x33;
//			else if(datai<=162)DataFromEeprom[datai]=0x34;
//			else DataFromEeprom[datai]=0x39;			 
//		}	
//	}
//	if(flag0xff==1)
//	{
//		Eeprom.writeEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//		printf("read eeprom again:");
//		Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//		for(datai=0;datai<200;datai++)
//		{
//			printf("%2x ",DataFromEeprom[datai]);	
//		}
//	}
//
//	printf("end read eeprom");
//
//		//SD卡电源开
//		printf("\n  hi SD poweron. ");		 
//	 	SD.ON();
//	
//	//SD初始化，成功的话也初始化一下FAT	
//		printf(" SD init. ");    
//		SD.init();
//		if(SD.flag!=NOTHING_FAILED)
//		{
//			printf("  failed!  ");
//			delay_ms(500);
//			while(1);
//		}
//	printf("enter ota choose ");
//	while(1)
//	{
//	  // Check if new data is available
//	  if( XBee.available() )
//	  {
////	  	printf("&");
//	    xbee802.treatData();
//	    // Keep inside this loop while a new program is being received
//	    while( xbee802.programming_ON  && !xbee802.checkOtapTimeout() )
//	    {
//	      if( XBee.available() )
//	      {
//	        xbee802.treatData();
//
//	      }
//	    }
//
//	  }
//	}
//
//}
//
//void loop()
//{
//	delay_ms(100);
//	
//}
//#endif // end OTA
//
//
//#if EXAMPLEXBEE==	BTESTXBEEOTAANDOTHERREC
//uint8_t DataFromEeprom[200]={"LIBELIUM0102030405060708090"};
//void setup()
//{
//	uint8_t datai;
//	Utils.initLEDs();
//	Utils.setLED(2, 1);
//	delay_ms(50);
//	Utils.setLED(2, 0);
//	delay_ms(50);
//	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
//	beginSerial(115200, 1); chooseuartinterrupt(1);
//	delay_ms(200);
//	serialWrite('5', 1);
//	delay_ms(200);
////	destMacAddagain[16]='\0';
//
//	delay_ms(200);
//	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
//	serialWrite('o', 1);
//	delay_ms(200);
//	xbee802.ON(); 	
//	monitor_offuart3TX();monitor_onuart3RX(); //放到xbee初始化前就是不能捕获数据（看不到收到的数据），也不知道为什么
//	xbee802.getChannel();
//	printf("firstgetchannel=%x ",xbee802.channel);
//	delay_ms(30);
//	xbee802.getChannel();	
//	printf("secondgetchannel=%x ",xbee802.channel);
//	xbee802.getMacMode();
//	printf("getMacMode=%x\t\t",xbee802.macMode);
//	xbee802.setOwnNetAddress(0xff, 0xff);
//	printf("getMY");
//	xbee802.getOwnNetAddress();
//	printf("=%x %x \t\t",xbee802.sourceNA[0],xbee802.sourceNA[1]);
//	Eeprom.ON();	
//	Eeprom.begin();
//	delay_ms(200);	
//	printf("  read eeprom from 107 len=8 ");
//	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
//	for(datai=0;datai<8;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
//	else
//	{	
//		printf("\r\n WRG  or else \r\n");
//		Eeprom.writeEEPROMStr(0xa0,107,(uint8_t *)"LIBELIUM0102030405060708090",16);
//		for(datai=0;datai<200;datai++)
//		{
//			DataFromEeprom[datai]=datai;	
//		}
//	}
//	printf(" 2th read eeprom from 107 len=8 ");
//	Eeprom.readEEPROMStr(0xa0,107,DataFromEeprom,8);
//	for(datai=0;datai<8;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	if(strncmp((char *)DataFromEeprom,"LIBELIUM",8)==0);
//	else 
//	{
//		printf("\r\n\r\n ********************WRONG****************** \r\n\r\n");
//	}
//
//	printf("read eeprom:");
//	Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//	for(datai=0;datai<200;datai++)
//	{
//		printf("%2x ",DataFromEeprom[datai]);	
//	}
//	uint8_t flag0xff=0;
//	for(datai=0;datai<200;datai++)
//	{
//		if(DataFromEeprom[datai]==0xff)
//		{
//			flag0xff=1;
//			if(datai<=33)DataFromEeprom[datai]=0x31;
//			else if(datai<=97)DataFromEeprom[datai]=0x32;
//			else if(datai<=106)DataFromEeprom[datai]=0x33;
//			else if(datai<=162)DataFromEeprom[datai]=0x34;
//			else DataFromEeprom[datai]=0x39;			 
//		}	
//	}
//	if(flag0xff==1)
//	{
//		Eeprom.writeEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//		printf("read eeprom again:");
//		Eeprom.readEEPROMStr(0xa0,0X0000,DataFromEeprom,200);
//		for(datai=0;datai<200;datai++)
//		{
//			printf("%2x ",DataFromEeprom[datai]);	
//		}
//	}
//
//	printf("end read eeprom");
//
//		//SD卡电源开
//		printf("\n  hi SD poweron. ");		 
//	 	SD.ON();
//	
//	//SD初始化，成功的话也初始化一下FAT	
//		printf(" SD init. ");    
//		SD.init();
//		if(SD.flag!=NOTHING_FAILED)
//		{
//			printf("  failed!  ");
//			delay_ms(500);
//			while(1);
//		}
//	printf("enter ota choose ");
//	unsigned char k;
//	while(1)
//	{
//		// Check if new data is available
//		if( XBee.available() )
//		{
//		//	  	printf("&");
//		    xbee802.treatData();
//			if( !xbee802.error_RX )
//			{
//				// Sending answer back
//				while(xbee802.pos>0)
//				{				
//					//printf("The whole data is =%s",xbee802.packet_finished[xbee802.pos-1]->data); 
//					//printf("seq %d =%s",NumRec,xbee802.packet_finished[xbee802.pos-1]->data); 
//		
//					printf("RSSI: =%x",xbee802.packet_finished[xbee802.pos-1]->RSSI);
//					printf("len: =%x",xbee802.packet_finished[xbee802.pos-1]->data_length); 
//					for(k=0;k<xbee802.packet_finished[xbee802.pos-1]->data_length;k++)
//						printf(" %2x",xbee802.packet_finished[xbee802.pos-1]->data[k]);
//									 
//					free(xbee802.packet_finished[xbee802.pos-1]);
//					xbee802.packet_finished[xbee802.pos-1]=NULL;
//					xbee802.pos--;
//				} 
//		
//			}
//			printf("-");
//		
//			// Keep inside this loop while a new program is being received
//			while( xbee802.programming_ON  && !xbee802.checkOtapTimeout() )
//			{
//				if( XBee.available() )
//				{
//					xbee802.treatData();
//				}
//			}
//		
//		}
//	}
//
//}
//
//void loop()
//{
//	delay_ms(100);
//	
//}
//#endif // end OTAandrec


