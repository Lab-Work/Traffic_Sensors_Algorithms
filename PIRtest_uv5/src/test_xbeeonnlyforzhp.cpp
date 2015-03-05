#include "allboardinc.h"
#define BTESTXBEESIMPLEONLYFORZHP 1
#define BTESTXBEESENDFASTBROADCASTONLYFORZHP 2
#define BTESTXBEESENDFASTUNICASTONLYFORZHP 3
#define BTESTXBEERECONLYFORZHP 4
#define BTESTXBEERECBYRx80Buffer 5
//#define BTESTXBEERECFASTUNICAST 5
#define BTESTXBEERECTHENFASTBROADCASTONLYFORZHP 6

#define BTESTXBEENULLONLYFORZHP 0

#define EXAMPLEXBEEONLYFORZHP BTESTXBEENULLONLYFORZHP//


extern	int rxprerx80_buffer_tail3 ;
extern int rx_buffer_head3;
extern		int Rx80BufferHead;
extern		int Rx80BufferTail;
extern 	unsigned char Rx80Buffer[];

//send a char
#if EXAMPLEXBEEONLYFORZHP>0

long CntStopOld = 0;
long CntStopCur = 0;
//unsigned char FlagXbeeLinshi =0 ;
unsigned long TimeusLinshi = 0;
//extern uint32_t TimemsArray[];
extern volatile unsigned long timer0_overflow_count;
//extern volatile unsigned long timer0_overflow_count;
char DestMacAdd[17]="0013A200409EDCB6"; // add the mac address for the sink
packetXBee* paq_sent;
uint8_t i=0;
uint8_t destination[8];
char*  data="Test message!";
static char StrPrint[50]; 
 uint8_t broadcastSignalOnlyForZhp(char datastr[],unsigned char lendata)
{
	
   	return xbee802.sendTx64Simple(datastr,lendata,0x00000000,0x0000ffff);

//	//XBee.println("data inside Broadcast ");
//	printf("data inside Broadcast ");
//	//XBee.println(data);
//	//XBee.println(RTCbianliang.getTime());
//	paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
//	paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
//	paq_sent->MY_known=0;
//	paq_sent->packetID=0x52;  //Think about changing it each time you send
//	paq_sent->opt=0;
//	xbee802.hops=0;
//	xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
//	xbee802.setDestinationParams(paq_sent, "000000000000FFFF", data, MAC_TYPE, DATA_ABSOLUTE);
//	xbee802.sendXBee(paq_sent);
//	
//	if( !xbee802.error_TX )
//	{
//	//XBee.println("ok");
//	}
//	free(paq_sent);
//	paq_sent=NULL;
              
             
}


//char teststrzpp[]={0x7e,0x0,0x1e,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0,0x52,0x1,0x23,0x0,0x56,0x78,0x54,0x65,0x73,0x74 ,0x20 ,0x6d ,0x65 ,0x73 ,0x73 ,0x61 ,0x67 ,0x65 ,0x21 ,0x7e};	//OK
//char teststrzpp[]={0x7e, 0x0, 0x0b, 0x0, 0x52, 0x0, 0x13, 0xa2, 0x0, 0x40, 0x9e, 0xdc, 0xb6, 0x0, 0xff};//WRONG
//char teststrzpp[]={0x7e,0x0,0x11,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0,0x52,0x1,0x23,0x0,0x56,0x78, 0x44};//OK

//char teststrzpp[]={0x7e,0x0,0x11,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0,0x52,0x1,0x23,0x0, 0x12};//OK
//char teststrzpp[]={0x7e,0x0,0xb,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0, 0x88};//超时
//char teststrzpp[]={0x7e,0x0,0xd,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0,0x52,0x1, 0x35};//OK

//char teststrzpp[]={0x7e,0x0,0xc,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
//0x0,0x52, 0x36};//OK

char teststrzpp[]={0x7e,0x0,0xc,0x0,0x52,0x0,0x13,0xa2,0x0,0x40,0x9e,0xdc,0xb6,
0x0,0x53, 0x35};//OK

uint8_t sendUnicastSignalOnlyForZhp(char datastr[],unsigned char lendata,unsigned long addressH,unsigned long addressL)
{
	return xbee802.sendTx64Simple(datastr,lendata, addressH, addressL); 
}
static uint8_t RecBuffer[200];
static uint16_t RecLenWant=190;
int  RecLenFact;


static void handle_received_dataOnlyForZhp()
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
			//printf("ok");
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
//				for(i=14;i<(RecLenWant-1);i++)
//					printf("%c",RecBuffer[i]);
				
				//以下这段打印是针对那个100个字节数据包，最后有个cnt = %8d，把那个值取出来的
				sum = 0;
				for(i=3;i<11;i++)
				{
					if(RecBuffer[RecLenWant-i]==0x20)break;
					sum = (RecBuffer[RecLenWant-i]-'0')*k + sum;
					k=k*10;
				}
				printf(" =%d \r\n",sum);

				
				CntStopCur = sum;
				//上面得到了cnt的值，原数据要求发次包cnt++，下面就是通过检测数据包是不是连续的，即这次的cnt和上次的是不是正好相差1
				//如果发现数据包不连续，则打印出Rx80Buffer 最近600,主要是看是因为自己算法照成了漏包，还是在10ms中断处理的时候造成的漏包
				if(CntStopOld!=0)
				{
					if((CntStopCur - CntStopOld)>1)
					{
						//if(Rx80BufferHead>1000)
						printf("H=%d T=%d ",Rx80BufferHead,Rx80BufferTail);
						offset = (Rx80BufferHead + RX80_BUFFER_SIZE -600)%RX80_BUFFER_SIZE;
						for(i=0; i<600;i++)
						{
							printf("%2x ",Rx80Buffer[offset]);	
							offset++;
							offset = offset %RX80_BUFFER_SIZE;
						}
						CntStopOld=0;		
					}
				}
				CntStopOld = CntStopCur;

					
								
			}
										
		}
		else
		{
			//如果没有收到有用的数据包，那打印下出错信息
//			printf("flagerr=%d ",flagerr);
//			if(flagerr==2)printf("H=%dT=%d ",Rx80BufferHead,Rx80BufferTail);

			//printf("n");

		}


//		printf("recnum=%d\r\n",recnum);
//		RecLenFact=serialReadstr(RecBuffer, RecLenWant,3);
//		printf("RecLenFact=%d\r\n",RecLenFact);
//		if((RecLenFact>0)&&(RecLenFact<100))
//		{
//			for(i=0;i<RecLenFact;i++)
//				printf("%2x ",RecBuffer[i]);
//		}


//		xbee802.treatData();
//		if( !xbee802.error_RX )
//		{
//		 // Sending answer back
//			while(xbee802.pos>0)
//			{
//				
//				XBee.println(" The whole data received is ");
//				XBee.println(xbee802.packet_finished[xbee802.pos-1]->data);
//				
//				paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee)); 
//				paq_sent->mode=UNICAST;
//				paq_sent->MY_known=0;
//				paq_sent->packetID=0x52;
//				paq_sent->opt=0; 
//				xbee802.hops=0;
//				xbee802.setOriginParams(paq_sent, "ACK", NI_TYPE);
//				while(i<4)
//				{
//					destination[i]=xbee802.packet_finished[xbee802.pos-1]->macSH[i];
//					i++;
//				}
//				while(i<8)
//				{
//					destination[i]=xbee802.packet_finished[xbee802.pos-1]->macSL[i-4];
//					i++;
//				}
//				xbee802.setDestinationParams(paq_sent, destination, data, MAC_TYPE, DATA_ABSOLUTE);
//				xbee802.sendXBee(paq_sent);
//				if( !xbee802.error_TX )
//				{
//					XBee.println("ok");
//				}
//				free(paq_sent);
//				paq_sent=NULL;
//				
//				free(xbee802.packet_finished[xbee802.pos-1]);   
//				xbee802.packet_finished[xbee802.pos-1]=NULL;
//				xbee802.pos--;
//			} 
//		}
	//}

	//delay_ms(2);    
}


void handle_received_databyRx80Buffer()
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
		flagerr=xbee802.findRxframeinRx80Buffer((char *)RecBuffer, &RecLenWant);
		if(flagerr==0)
		{
			//printf("ok");
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
//				for(i=14;i<(RecLenWant-1);i++)
//					printf("%c",RecBuffer[i]);
				
				//以下这段打印是针对那个100个字节数据包，最后有个cnt = %8d，把那个值取出来的
				sum = 0;
				for(i=3;i<11;i++)
				{
					if(RecBuffer[RecLenWant-i]==0x20)break;
					sum = (RecBuffer[RecLenWant-i]-'0')*k + sum;
					k=k*10;
				}
				printf(" =%d \r\n",sum);

				
				CntStopCur = sum;
				//上面得到了cnt的值，原数据要求发次包cnt++，下面就是通过检测数据包是不是连续的，即这次的cnt和上次的是不是正好相差1
				//如果发现数据包不连续，则打印出Rx80Buffer 最近600,主要是看是因为自己算法照成了漏包，还是在10ms中断处理的时候造成的漏包
				if(CntStopOld!=0)
				{
					if((CntStopCur - CntStopOld)>1)
					{
						//if(Rx80BufferHead>1000)
						printf("H=%d T=%d ",Rx80BufferHead,Rx80BufferTail);
						offset = (Rx80BufferHead + RX80_BUFFER_SIZE -600)%RX80_BUFFER_SIZE;
						for(i=0; i<600;i++)
						{
							printf("%2x ",Rx80Buffer[offset]);	
							offset++;
							offset = offset %RX80_BUFFER_SIZE;
						}
						CntStopOld=0;		
					}
				}
				CntStopOld = CntStopCur;					
			}
										
		}
		else
		{
			//如果没有收到有用的数据包，那打印下出错信息
//			printf("flagerr=%d ",flagerr);
//			if(flagerr==2)printf("H=%dT=%d ",Rx80BufferHead,Rx80BufferTail);

			//printf("n");

		}






	//delay_ms(2);    
}




void handlerecandbroadcastOnlyForZhp(void)
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
			printf("ok");
			printf("RecLenWant=%d ",RecLenWant);
//			for(i=0;i<RecLenWant;i++)
//				printf("%2x ",RecBuffer[i]);


			for(i=0;i<RecLenWant;i++)
				printf("%c",RecBuffer[i]);
//			if(RecLenWant>24)
//			{
//				RecBuffer[RecLenWant-14] = 's';
//				k=0;
//				for(i=14;i<(RecLenWant-1);i++)
//				{
//					//printf("%c",RecBuffer[i]);
//					StrPrint[k]=RecBuffer[i];
//					k++;
//				}
//				StrPrint[k]=0x00;
//				flagerr = broadcastSignal(StrPrint,strlen(StrPrint));
//				if(flagerr==0)
//					printf(" OK ");
//				else 
//					printf("err=%d",flagerr);
//					
//				sum = 0;
//				for(i=3;i<11;i++)
//				{
//					if(RecBuffer[RecLenWant-i]==0x20)break;
//					sum = (RecBuffer[RecLenWant-i]-'0')*k + sum;
//					k=k*10;
//				}
//				printf(" =%d \r\n",sum);
//
//				
//				CntStopCur = sum;
//
//				if(CntStopOld!=0)
//				{
//					if((CntStopCur - CntStopOld)>1)
//					{
//						printf("H=%d T=%d ",Rx80BufferHead,Rx80BufferTail);
//						offset = (Rx80BufferHead + RX80_BUFFER_SIZE -600)%RX80_BUFFER_SIZE;
//						for(i=0; i<600;i++)
//						{
//							printf("%2x ",Rx80Buffer[offset]);	
//							offset++;
//							offset = offset %RX80_BUFFER_SIZE;
//						}
//						CntStopOld=0;		
//					}
//				}
//				CntStopOld = CntStopCur;
//					
//			}
										
		}
		else
		{
//			printf("flagerr=%d ",flagerr);
//			if(flagerr==2)printf("H=%dT=%d ",Rx80BufferHead,Rx80BufferTail);

			//printf("n");

		}
		//delay_ms(2);

  
}




char teststrzpp1[]={0x7e,0x00,0x04,0x08,0x52,0x53,0x4c,0x06};//SL
char teststrzpp2[]={0x7e,0x00,0x04,0x08,0x52,0x53,0x48,0x0A};//SH
char teststrzpp3[]={0x7e,0x00,0x04,0x08,0x52,0x4D,0x59,0xff};//MY
char teststrzpp4[]={0x7e,0x00,0x04,0x08,0x52,0x43,0x48,0x1A};//CH




#endif


#if EXAMPLEXBEEONLYFORZHP==BTESTXBEESIMPLEONLYFORZHP


extern uint8_t  NDMacLength;
extern uint64_t NDMacAddress[10];
extern uint16_t NDMyAddress[10];


void setup()
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	//monitor_on();

	monitor_onuart3TX();
	//monitor_onuart3RX();

	muluart6init();//借用一下电源正，要不然串口3(1)不好测试

	RTCbianliang.ON();//借用一下电源正，要不然串口3(1)不好测试
//	beginSerial(115200, 3);
	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);
		delay_ms(200);
		Utils.setLED(2, 0);
		delay_ms(200);
	}
	printf("zpp!");
//	serialWrite('3', 3);//send a char '3' by uart3
//	serialWrite('c', 3);//send a char 'c'

//	while(1)
//	{
//		serialWrite('3', 3);//send a char '3' by uart3
//		serialWrite('c', 3);//send a char 'c'
//		flagbreak++;
//		if(flagbreak==10000){Utils.setLED(0, 1);}
//		else if(flagbreak==20000){flagbreak=0;Utils.setLED(0, 0);}
//		if(flagbreak>50000)break;
//	}
	DestMacAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(300);
	RTCbianliang.ON(); 
	delay_ms(300);//XBee.println("xbee test");
	printf("xbee test\r\n");
	flagbreak=0;
	//以下轮流发送ucast
	while(1)
	{
		flagbreak++;
		if(flagbreak>=2)flagbreak=0;
		if(flagbreak==1){monitor_onuart3TX();monitor_offuart3RX();}
		else{monitor_offuart3TX();monitor_onuart3RX();}

		printf("get own mac address");
		if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
		);

		else printf("flagxbeeeer=%d ",flagxbeeerr);
		printf("\r\n");

		delay_ms(1000);
//
//
//		printf("get own net address");
//		if((flagxbeeerr=xbee802.getOwnNetAddress())==0)printf("=%2x%2x "\
//		,xbee802.sourceNA[0],xbee802.sourceNA[1]\
//		);
//		else printf("flagxbeeeer=%d ",flagxbeeerr);
//		printf("\r\n");
//		delay_ms(1000);
//	
//		printf("get channel");
//		if((flagxbeeerr=xbee802.getChannel())==0)printf("=%2x"\
//		,xbee802.channel\
//		);
//		else printf("flagxbeeeer=%d ",flagxbeeerr);
//		printf("\r\n");
//		delay_ms(1000);

		printf("scannetwork");
		if((flagxbeeerr=xbee802.scanNetwork())==0)printf("\r\nOK\r\n");
		else printf("flagxbeeeer=%d ",flagxbeeerr);
		printf("\r\n");
		if(NDMacLength>0)
		{	
			for(uint8_t ndi=0; ndi<NDMacLength;ndi++)
				printf(" my=%d mach=%8x L=%8x\t",NDMyAddress[ndi],(uint32_t)(NDMacAddress[ndi]>>32),(uint32_t)(NDMacAddress[ndi]));
		}

		delay_ms(5000);


//		printf("broadcast1:");
//		sprintf(StrPrint,"broadcast1: my Mac H=%2x%2x%2x%2x L=%2x%2x%2x%2x "\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
//		);
//		flagxbeeerr = broadcastSignal(StrPrint,strlen(StrPrint));
//		if(flagxbeeerr==0)
//		{
//			printf(" ok ");
//		}
//		else printf(" wrong ");
//		printf("\r\n");
//		delay_ms(1000);
//
//		printf("ucast1:");
//		sprintf(StrPrint,"unicast1: my Mac=%2x%2x%2x%2x %2x%2x%2x%2x "\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
//		);
//	   	flagxbeeerr = sendUnicastSignal(StrPrint,strlen(StrPrint),0x0013a200,0x409edcb6);
//		if(flagxbeeerr==0)
//		{
//			printf(" ok ");
//		}
//		else printf(" wrong ");
//		printf("\r\n");
//		delay_ms(1000);
//
//
//		printf("broadcast2:");
//		sprintf(StrPrint,"broadcast2: my Mac H=%2x%2x%2x%2x L=%2x%2x%2x%2x "\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
//		);
//		flagxbeeerr = broadcastSignal(StrPrint,strlen(StrPrint));
//		if(flagxbeeerr==0)
//		{
//			printf(" ok ");
//		}
//		else printf(" wrong ");
//		printf("\r\n");
//		delay_ms(1000);
//
//		printf("ucast2:");
//		sprintf(StrPrint,"unicast2: my Mac=%2x%2x%2x%2x %2x%2x%2x%2x "\
//		,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
//		,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
//		);
//	   	flagxbeeerr = sendUnicastSignal(StrPrint,strlen(StrPrint),0x0013a200,0x409c6c6a);
//		if(flagxbeeerr==0)
//		{
//			printf(" ok ");
//		}
//		else printf(" wrong ");
//		printf("\r\n");
//		delay_ms(1000);
//
//
//
//		delay_ms(6000);
	}
	//这个只是简单的把接收的打印出来而已
	while(1)
	handle_received_dataOnlyForZhp();

}


void loop()
{
    delay_ms(3000);
}

#endif


#if EXAMPLEXBEEONLYFORZHP==BTESTXBEESENDFASTBROADCASTONLYFORZHP
//这里就是单纯的发一个广播数据，发送数据的时候都不问MY，直接在之前置为为0xffff，所以这里的测试是最为单纯的发一个包，等待xbee的回馈，然后再发一个包，
//测试的数据包大约为42个字节，在发出去这42个字节等待xbee回馈时，大约为6ms(5-6ms)， 
//串口用时间为 (1/115200)*10*(42+7) = 4.3ms
//一个字节 一般需要10个数据位（开始1位，数据8位，停止1位）  所以(1/115200)*10 .
//发出去数据这里大约是42个字节，接收到的TX的回馈那帧数据为7位，所以串口所有的数据为 42+7个
//串口时间加上xbee回馈时间为 10.3ms
//在供应商提供的xbee板子，在一秒之内收到的包大约6000个,也就是实测大约每秒100个包

//注意：以上的发广播数据时没有问MY，如果正常MY，再改MY，发完数据包之后，再改回MY，根据上面的xbee回馈，可知，正常一套广播下xbee回馈的时间一共为6*4=24ms
//问MY 8个数据 回答11个数据 设置MY10个数据 回应设置9个数据， 这样串口来来回回，不包括真正的广播数据，要57个字符
//刚才那样广播数据为42的话，加上回的7个，在加上MY的那些，一共106个数据
//总体时间为： (1/115200)*10*106 s + 24 ms= 9.2 ms + 24 ms = 34.2 ms
//这个广播正常时间为 1000/34.2 = 29	包 每秒


//若按照以上的计算方法，虽然不知道当100个字节广播时候xbee回馈是多少，假定就是这里的6ms，那么一个包时间为
// (1/115200)*10*(57+107) s + 24 ms= 14.2 ms + 24 ms = 38.2 ms
// 1000/38.2 = 26	包 每秒

//以下实测包数据量为100个字节的

//通过实际测量，不包含那个MY的，60秒大约3300个包，等待xbee回馈的时间为8ms
//核算下： 串口(1/115200)*10*(107)s + 8ms = 9.2ms + 8ms = 17.2ms
//1000/17.2 = 58包每秒，这个和 3300/60 = 55包每秒 是一致的

//把广播里面的两个设置MY（先设为0xffff，然后再设置为原先的值）恢复  （ 没有问MY那个） 实测，
//广播发送的时候等待XBee还是8ms	 设置MY为0xffff时候xbee回馈是2ms
//60秒收到2468个包	即每秒41个包   1/41 = 24.3 ms
//核算 (1/115200)*10*(107 + 19*2)s + (8+2+2)ms = 12.6ms + 12ms = 24.6ms
//1000/24.6= 40	包每秒，是一致的
//

void setup()//
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	unsigned long cntsend=0;
	monitor_onuart3TX();
	muluart6init();//借用一下电源正，要不然串口3(1)不好测试
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据

	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);
		delay_ms(200);
		Utils.setLED(2, 0);
		delay_ms(200);
	}
	printf("zpp!");//
	DestMacAdd[16]='\0';//
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(300);
	RTCbianliang.ON(); 
	delay_ms(300);//XBee.println("xbee test");
	printf("xbee test\r\n");

	printf("get own mac address");
	if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
	);
	else printf("flagxbeeeer=%d ",flagxbeeerr);
	printf("\r\n");
	delay_ms(1000);

	flagbreak=0;

	xbee802.setOwnNetAddress(0x00,0x00);

	//以下轮流发送broadcast
	while(1)
	{

//		sprintf(StrPrint,"0=%8d1=%8d2=%8d3=%8d4=%8d5=%8dtm=%8dcnt=\r\n%8d\t"\
//		,TimemsArray[0],TimemsArray[1],TimemsArray[2],TimemsArray[3],TimemsArray[4],TimemsArray[5]\
//		,TimeusLinshi\
//		,cntsend
//		);


		sprintf(StrPrint,"tm=%8dcnt=\r\n%8d\t"\
		//,TimemsArray[0],TimemsArray[1],TimemsArray[2],TimemsArray[3],TimemsArray[4],TimemsArray[5]\
		
		,TimeusLinshi\
		,cntsend
		);

		flagxbeeerr = broadcastSignalOnlyForZhp(StrPrint,strlen(StrPrint));
		if(flagxbeeerr==0)
		{
			printf(" ok ");
		}
		else printf(" wrong ");
		printf("\r\n");
		cntsend++;
		//XBee.print("hahha");
		//delay_ms(500);
	}
}


void loop()
{
    delay_ms(3000);
}

#endif

#if EXAMPLEXBEEONLYFORZHP==BTESTXBEESENDFASTUNICASTONLYFORZHP

//以下实测包数据量为100个字节的	UNICAST
//把UNICAST里面的两个设置MY是有的，
//一对一发送
//对方xbee是插在供应商提供的模块上面（也就是说，这个xbee比较空闲）,串口发送的时候等待XBee是9ms	 设置MY为0xffff时候xbee回馈是2ms
//60秒收到2422个包	即每秒40个包   1/40 = 25 ms
//核算 (1/115200)*10*(107 + 19*2)s + (9+2+2)ms = 12.6ms + 12ms = 25.6ms
//1000/24.6= 39	包每秒，是一致的
//

//一对二发送，也就是先发到一个xbee模块，再发到另一个xbee模块上，两个模块都是在线的，看起来发送的包数量和上面的一致，也是大约每秒40个包


//一对二发送，也就是先发到一个xbee模块，再发到另一个xbee模块上，两个模块有个不在线的，
//60秒 1840包	大约30包每秒
//对于不在线的那个xbee，当串口发TX给xbee等待xbee回复时，花了15ms的时间
//核算两个包时间，一个是在线xbee，一个不在线xbee ((1/115200)*10*(107 + 19*2)s + (2+2)ms)*2 + 9ms +25ms = 67.2ms
//1000/(67.2/2) = 29包每秒

//三对一发送，也就是三个xbee同时对一个xbee模块发送东西，发送xbee的是速度最快的那种，即不间断的那种。
//对接收的那个xbee模块，只看串口接收口的数据（不用程序处理的那种）,
//当一对一发送的时候，之间测过，没有丢包的情况。这也正常，因为xbee无线速度为250000,串口波特率是115200，大约为无线的一半，所以对于无线来说，串口的速度较低，可以处理过来
//当三对1时，这样空中的包数据量是一对一的三倍，从接受的串口看来，有很多包没有接收到，丢包率大约在1/3的样子，这个1/3不是严格的检查的，就是大概看了3个发送的数据每个大约十几个来个包，有个丢了5个包，有个丢2个包什么的

//当A板发送单播给B板时候（每100ms发一次），B板同时发送单播给C板(不间隔时间发送) ,两组数据没有关联。 
//从C板串口接收口直接看接受口，可以看到，B板给C板的数据包丢包率是很严重的，从第11包到133包，真正到了C板的只有40包收到了
//在发送数据时检测超时定位100ms的
//这个测试也就是意味着，想转包，就别想不漏包或者最大速率之类的

void setup()
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	unsigned long cntsend=0;
	monitor_onuart3TX();
	muluart6init();//借用一下电源正，要不然串口3(1)不好测试
	RTCbianliang.ON();//借用一下电源正，要不然串口3(1)不好测试
	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);
		delay_ms(200);
		Utils.setLED(2, 0);
		delay_ms(200);
	}
	printf("zpp!");
	DestMacAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(300);
	RTCbianliang.ON(); 
	delay_ms(300);//XBee.println("xbee test");
	printf("xbee test\r\n");

	printf("get own mac address");
	if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
	);
	else printf("flagxbeeeer=%d ",flagxbeeerr);
	printf("\r\n");
	delay_ms(1000);

	flagbreak=0;

	xbee802.setOwnNetAddress(0x00,0x00);

	//以下轮流发送ucast
	while(1)
	{

//		sprintf(StrPrint,"0=%8d1=%8d2=%8d3=%8d4=%8d5=%8dtm=%8d cnC=%8d\r\n"\
//		,TimemsArray[0],TimemsArray[1],TimemsArray[2],TimemsArray[3],TimemsArray[4],TimemsArray[5]\
//		,TimeusLinshi\
//		,cntsend
//		);
//		flagxbeeerr = sendUnicastSignal(StrPrint,strlen(StrPrint),0x0013a200,0x409edcb6);;
//		if(flagxbeeerr==0)
//		{
//			printf(" ok ");
//		}
//		else printf(" wrong ");
//		printf("\r\n");
//		cntsend++;


//		sprintf(StrPrint,"0=%8d1=%8d2=%8d3=%8d4=%8d5=%8dtm=%8dcnt=\r\n%8d\t"\
//		,TimemsArray[0],TimemsArray[1],TimemsArray[2],TimemsArray[3],TimemsArray[4],TimemsArray[5]\
//		,TimeusLinshi\
//		,cntsend
//		);

		sprintf(StrPrint,"tm=%8dcnt=\r\n%8d\t"\
		,TimeusLinshi\
		,cntsend
		);
		flagxbeeerr = sendUnicastSignalOnlyForZhp(StrPrint,strlen(StrPrint),0x0013a200,0x40710845);
		if(flagxbeeerr==0)
		{
			printf(" ok ");
		}
		else printf(" wrong ");
		printf("\r\n");
		cntsend++;
		//delay_ms(100);

		//XBee.print("hahha");
	}
}


void loop()
{
    delay_ms(3000);
}

#endif


#if EXAMPLEXBEEONLYFORZHP==BTESTXBEERECONLYFORZHP


void setup()
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	unsigned long cntsend=0;
	//monitor_on();

	monitor_onuart3TX();
	//monitor_onuart3RX();

	muluart6init();//借用一下电源正，要不然串口3(1)不好测试

	RTCbianliang.ON();//借用一下电源正，要不然串口3(1)不好测试
//	beginSerial(115200, 3);
	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);	Utils.setLED(0, 1);
		delay_ms(200);
		Utils.setLED(2, 0);Utils.setLED(0, 0);
		delay_ms(200);
	}
	Timer2_Init(10,1000);//200ms
	printf("zpp!");
	DestMacAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(10);
	RTCbianliang.ON(); 
	delay_ms(10);//XBee.println("xbee test");
	printf("xbee test\r\n");

	printf("get own mac address");
	if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
	);
	else printf("flagxbeeeer=%d ",flagxbeeerr);
	printf("\r\n");
	//delay_ms(1000);

	flagbreak=0;
	//以下轮流发送ucast
	while(1)
	{
		//flagxbeeerr=handle1msprerx80();
//
//				if(flagxbeeerr==0)printf("A ");
//				else printf("%c ",flagxbeeerr+'A');
//
//				if(flagxbeeerr==2)printf("h=%dt=%d ",rx_buffer_head3,rxprerx80_buffer_tail3);
		//delay_ms(100);
		handle_received_dataOnlyForZhp();
	}


}


void loop()
{
    delay_ms(3000);
}

#endif




//一个xbeeA广播不停的发送100个字节（那个有两个设置MY的），速度大约为40包每秒
//另一个xbeeB只管接收，接收到完整的一个0x80RX包时，就打印出来，	打印用%c打印的，
//        注意：如果%2x则占用比较多的时间，等待打印完了，就会发现串口缓冲区里面超过100个字节，也就是一个包多的数据，
//              这样，前几次的包还能正确解析，当多出一定的时候，就解析不成功，然后自我校正，过了一段时间正好校正到了0x7e了，又重新正确解析包，然后循环
//              所以%2x 打印会漏包
//用超级终端保存了大约1分钟的B打印的情况，大约2400个包，看到这些包没有漏包的情况，也就是说，发送 接收丢包率为0%
//
//上面的%2x的情况也说明了，如果A不停的发送数据（40包每秒），B如果要转发的话，很可能话转发不成功或者漏包严重，
//对于如何快速转包编程要好好想想

//对于直接接收处理（#if EXAMPLEXBEE==BTESTXBEEREC） ，即xbee.read()  和 到Rx80Buffer处理接收XBee.Rx80read()，下面这段程序
//当接收到一个包，打印广播发送的带有cnt=%8d那个数据包时，打印出那个cnt值
//发现不管是普通直接接收还是Rx80Buffer处理接收 测了一分钟，都能够收到广播的所有包 也就是说丢包率为0

//上面的处理单播的包，当Rx80Buffer处理接收 测了一分钟，丢包率也是为0
#if EXAMPLEXBEEONLYFORZHP==BTESTXBEERECBYRx80Buffer


void setup()
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	unsigned long cntsend=0;
	//monitor_on();

	monitor_onuart3TX();
	//monitor_onuart3RX();

	muluart6init();//借用一下电源正，要不然串口3(1)不好测试

	RTCbianliang.ON();//借用一下电源正，要不然串口3(1)不好测试
//	beginSerial(115200, 3);
	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);	Utils.setLED(0, 1);
		delay_ms(200);
		Utils.setLED(2, 0);Utils.setLED(0, 0);
		delay_ms(200);
	}
	Timer2_Init(10,1000);//200ms
	printf("zpp!");
	DestMacAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(10);
	RTCbianliang.ON(); 
	delay_ms(10);//XBee.println("xbee test");
	printf("xbee test\r\n");

	printf("get own mac address");
	if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
	);
	else printf("flagxbeeeer=%d ",flagxbeeerr);
	printf("\r\n");
	//delay_ms(1000);

	flagbreak=0;
	//以下轮流发送ucast
	while(1)
	{
		//flagxbeeerr=handle1msprerx80();
//
//				if(flagxbeeerr==0)printf("A ");
//				else printf("%c ",flagxbeeerr+'A');
//
//				if(flagxbeeerr==2)printf("h=%dt=%d ",rx_buffer_head3,rxprerx80_buffer_tail3);
		//delay_ms(100);
		handle_received_databyRx80Buffer();
	}


}


void loop()
{
    delay_ms(3000);
}

#endif












#if EXAMPLEXBEEONLYFORZHP==BTESTXBEERECTHENFASTBROADCASTONLYFORZHP


void setup()
{
	unsigned long flagbreak=0;
	int flagxbeeerr;
	unsigned long cntsend=0;
	//monitor_on();

	monitor_onuart3TX();
	//monitor_onuart3RX();

	muluart6init();//借用一下电源正，要不然串口3(1)不好测试

	RTCbianliang.ON();//借用一下电源正，要不然串口3(1)不好测试
//	beginSerial(115200, 3);
	beginSerial(115200, PRINTFPORT);
	delay_ms(300);
	for(uint16_t i=0;i<5;i++){
		Utils.setLED(2, 1);	Utils.setLED(0, 1);
		delay_ms(200);
		Utils.setLED(2, 0);Utils.setLED(0, 0);
		delay_ms(200);
	}
	Timer2_Init(10,1000);//200ms
	printf("zpp!");
	DestMacAdd[16]='\0';
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	delay_ms(10);
	RTCbianliang.ON(); 
	delay_ms(10);//XBee.println("xbee test");
	printf("xbee test\r\n");

	printf("get own mac address");
	if((flagxbeeerr=xbee802.getOwnMac())==0)printf("machigh=%2x%2x%2x%2x maclow=%2x%2x%2x%2x "\
	,xbee802.sourceMacHigh[0],xbee802.sourceMacHigh[1],xbee802.sourceMacHigh[2],xbee802.sourceMacHigh[3]\
	,xbee802.sourceMacLow[0],xbee802.sourceMacLow[1],xbee802.sourceMacLow[2],xbee802.sourceMacLow[3]\
	);
	else printf("flagxbeeeer=%d ",flagxbeeerr);
	printf("\r\n");
	//delay_ms(1000);

	flagbreak=0;
	//以下轮流发送ucast
	while(1)
	{
		//flagxbeeerr=handle1msprerx80();
//
//				if(flagxbeeerr==0)printf("A ");
//				else printf("%c ",flagxbeeerr+'A');
//
//				if(flagxbeeerr==2)printf("h=%dt=%d ",rx_buffer_head3,rxprerx80_buffer_tail3);
		//delay_ms(100);
		handlerecandbroadcastOnlyForZhp();
	}


}


void loop()
{
    delay_ms(3000);
}

#endif//BTESTXBEERECTHENFASTBROADCAST



























