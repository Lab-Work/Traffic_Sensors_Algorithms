/*
这个是板子的换主频例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：    
#define BTESTSYSCLKWHILECHG 0 //主频不停变化（减少），每次差1M，到了8M在循环到168M
#define BTESTSYSCLKNULL 100	 不选用换主频例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/

#include "allboardinc.h"
#define BTESTSYSCLKWHILECHG 0 //主频不停变化（减少），每次差1M，到了8M在循环到168M
#define BTESTSYSCLKNULL 100

#define EXAMPLESYSCLK BTESTSYSCLKNULL 

#if EXAMPLESYSCLK==BTESTSYSCLKWHILECHG
//粗略延时
static void ZPPDelay(u32 count)
{
	while(count--);
}

//这个是编译文件的年月日
//static 
struct structfiletime
{
	uint8_t year;
	uint8_t month;
	uint8_t date;
	uint8_t hour;
	uint8_t minute;
	uint8_t second;
};
//这个是编译文件时候的年月日
static union unionfiletime
{
	struct structfiletime unstrft;
	uint8_t stringfiletime[sizeof(struct structfiletime)/sizeof(uint8_t)];
}unfltm;	// 

//这个是针对 __DATE__,__TIME__,的
//年份忽略前面的两位。比如2013只采用13
static void getcomplietime(char * strdate,char * strtime)
{
	char *pstr;
	unsigned shi,fen,miao;
	pstr=strdate;
	pstr[3]=0x00;
	if(strncmp(pstr,"Jan",3)==0){unfltm.unstrft.month=1;}
	else if(strncmp(pstr,"Feb",3)==0){unfltm.unstrft.month=2;}
	else if(strncmp(pstr,"Mar",3)==0){unfltm.unstrft.month=3;}
	else if(strncmp(pstr,"Apr",3)==0){unfltm.unstrft.month=4;}
	else if(strncmp(pstr,"May",3)==0){unfltm.unstrft.month=5;}
	else if(strncmp(pstr,"Jun",3)==0){unfltm.unstrft.month=6;}
	else if(strncmp(pstr,"Jul",3)==0){unfltm.unstrft.month=7;}
	else if(strncmp(pstr,"Aug",3)==0){unfltm.unstrft.month=8;}
	else if(strncmp(pstr,"Sep",3)==0){unfltm.unstrft.month=9;}
	else if(strncmp(pstr,"Oct",3)==0){unfltm.unstrft.month=10;}
	else if(strncmp(pstr,"Nov",3)==0){unfltm.unstrft.month=11;}
	else {unfltm.unstrft.month=12;}

	pstr=strdate+4;
	pstr[2]=0x00;
	unfltm.unstrft.date=(uint8_t) atoi(pstr);

	pstr=strdate+9;
	pstr[2]=0x00;
	unfltm.unstrft.year=(uint8_t) atoi(pstr);

	sscanf(strtime,\
	"%d:%d:%d",\
	&shi,&fen,&miao);

	unfltm.unstrft.hour=(uint8_t)shi;
	unfltm.unstrft.minute=(uint8_t)fen;
	unfltm.unstrft.second=(uint8_t)miao;
}

/*
查看RTC时间是否需要改写
返回值0：不需要改写  	  返回值1：需要改写  
用程序编译的时候的时间和RTC时间比较一下，如果RTC时间明显小于编译时间，说明RTC时间肯定有问题，需要改写。
如果RTC时间大于编译时间，这个就不能肯定了，此时返回不需要改写
*/
static uint8_t ischangertc(void)
{
 //	uint8_t flagchangertc=0;//0不该写RTC 1改写RTC
	uint32_t day32file;	
	uint32_t day32rtc;
	uint32_t minute32file;
	uint32_t minute32rtc;

	day32file = unfltm.unstrft.year*10000 +  unfltm.unstrft.month * 100 + unfltm.unstrft.date; 
	day32rtc  =	RTCbianliang.year*10000 + RTCbianliang.month*100 + RTCbianliang.date;

	if(day32rtc>day32file)return 0;//
	else if(day32rtc<day32file)return 1;
	else{
		minute32file = 	unfltm.unstrft.hour*60 +  unfltm.unstrft.minute ;
		minute32rtc  =	RTCbianliang.hour*60 + RTCbianliang.minute;

		if(minute32rtc>minute32file)return 0;
		else if((minute32file-minute32rtc)>20){
			return 1;
		}
		else return 0;
	}
//	return 0;
}

//static uint8_t ReadBuffer[512];


/*
不停换主频测试。
    看串口，I2C，SPI，SD卡是否能正常运行，在正常的时候把数据记录到SD卡里面

	先在setup()里面主频168M下闪灯，定时器，串口，得到RTC值，且打印编译时间，RTC时间，看RTC是否要改写
	然后在loop()里面不停的减少主频值，检测LED（GPIO口），定时器，系统时钟的延时（delay_ms），串口，I2C，SPI，SD卡。在运行正常情况下在SD卡下记录主频和RTC时间
*/

static	unsigned long testsys=168000000;
static	unsigned char flagfangxiang=0;

static	unsigned long zppSysClkWasp;
static  unsigned long zppDelayMsWasp;
static	unsigned long zppDelayUsWasp;
static	unsigned long zppSysPllN,zppSysPllM,zppSysPllP,zppSysPllQ;
static	unsigned long zppSysFlashWait;

static uint8_t AddSDstr[512];//需要新添入到SD卡里面的数据
//主频168M下闪灯，定时器，串口，得到RTC值，且打印编译时间，RTC时间，看RTC是否要改写
void setup()
{
	zppSystemInit();
	
	Utils.initLEDs();//调用灯之前必须初始化一下
	for(int i=0;i<5;i++)
		Utils.blinkLEDs(1000);
				
	Timer3_Init(10,10000);
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, PRINTFPORT); //这个测试老是让串口助手死掉
	printf ("\n\n                  Designed by ZHP,    Built: %s %s ", __DATE__, __TIME__);
	getcomplietime(__DATE__,__TIME__);
	printf(" strtime=%d %d %d  %d %d %d "\
	,unfltm.unstrft.year,unfltm.unstrft.month,unfltm.unstrft.date\
	,unfltm.unstrft.hour,unfltm.unstrft.minute,unfltm.unstrft.second);

	printf("\nfile: %s ,",__FILE__); 
	printf("\nfunction: %s ,",__func__); 
	printf("\nline: %d\n",__LINE__); 


	RTCbianliang.ON();
	RTCbianliang.begin();
	RTCbianliang.getTime();uartsendstr((char *)RTCbianliang.timeStamp, 30,PRINTFPORT);delay_ms(1000);
	if(ischangertc()==1){
				printf(" differnt time ");
				RTCbianliang.setTime(unfltm.unstrft.year, unfltm.unstrft.month, unfltm.unstrft.date, 1\
				, unfltm.unstrft.hour, unfltm.unstrft.minute, unfltm.unstrft.second);
	}
}
//loop()里面不停的减少主频值，检测LED（GPIO口），定时器，系统时钟的延时（delay_ms），串口，I2C，SPI，SD卡。在运行正常情况下在SD卡下记录主频和RTC时间
void loop()
{
	unsigned int i;
	unsigned int cishu=0;
	RCC_ClocksTypeDef RCC_ClocksStatus;
	char filenameorig[20];
	uint8_t eepromdata[30]={"this is test of ic"};
	int32_t filesize=-2;
	uint64_t fsize;

	testsys=168000000;
	for(cishu=0;cishu<1000;cishu++)
	{
		//调用自己做的系统初始化，这个频率是全局变量SysClkWasp，初始默认为168M
		zppSystemInit();	
		Mux_poweron(); //借用一个电源正给串口用的，方便打印数据		
		
		//定时器3初始化，不管系统时钟是什么，这里都会是1ms一个中断的,怎么做到这点，详情看里面定义
		Timer3_Init(10,10000);
		//灯闪烁几下，这里涉及的时间是有系统嘀嗒决定的，不管频率多少，这里的ms us都是自动调整的，详情见ms us里面定义
		for(i=0;i<5;i++)Utils.blinkLEDs(100);
		//开始串口，这里的波特率也是根据系统时钟调配出来的，详情见里面定义
		beginSerial(115200, PRINTFPORT);
		//这里用了最简单的延时，是因为当定时器或者系统嘀嗒那个延时没有配好的话，这个延时仅作简单参考		
		ZPPDelay(0x3456);
		printf(" cishu=%d ",cishu);ZPPDelay(0x34567);
		//得到当前的系统时钟的一些配置，且打印出来
		RCC_GetClocksFreq(&RCC_ClocksStatus);
		printf("sys=%d,hclk=%d,pclk1=%d,pclk2=%d"\
		,RCC_ClocksStatus.SYSCLK_Frequency,RCC_ClocksStatus.HCLK_Frequency\
		,RCC_ClocksStatus.PCLK1_Frequency,RCC_ClocksStatus.PCLK2_Frequency);
		delay_ms(1000);


		


		for(i=0;i<5;i++)
		{		
		  Utils.setLED(0, 1);
		  //没有开定时器	Timer3_Init(10,10000); 不能用delay(40);这个函数，这个函数只是在移植xbee的时候里面有这个函数，因为芯片不一样，为了这个延时达到原来的作用特地开了定时器来做的
		  //不过在开xbee的时候，自动把定时器3也开了，所以假如程序里面开了xbee 就不用单独在初始化一下定时器3了
		  delay(40); 		  printf("i=%d ",i);
		  Utils.setLED(1, 1); 	
		  delay(40); 
		  printf("i=%d ",i);
		}
		for(i=0;i<5;i++)
		{
		  Utils.setLED(0, 1);
		  delay_ms(20); 
		  printf("I=%d ",i);
		  Utils.setLED(1, 1);	
		  delay_ms(20); 
		  printf("I=%d ",i);
		}


		//下面这段换频率的初步测了下，是可以的，因为flash不能檫除次数太多，所以把这段隐掉了
		printf("\r\nspi init");
		Flash.SPI_Flash_Init();
		uint16_t tmpreg = SPI2->CR1;
		uint16_t spibaudrate = (tmpreg>>3)&0x07;
		uint32_t spifclk =	(RCC_ClocksStatus.PCLK1_Frequency >> (spibaudrate+1));
		printf(" spifclk=%d ",spifclk);

//		int readch=0;
//		while(1){
//			readch=0;
//			for(long i100=0;i100<0xf;i100++){
//				readch=Flash.flashreaddata4add(4092);
//				}
//			printf(" 5result of read =%x ",readch);
//
//
//			readch=0;
//			for(long i100=0;i100<0xffff;i100++){
//				readch=Flash.flashreaddata4add(4092);
//				}
//			printf(" 6result of read =%x ",readch);
//
//			readch=0;
//			for(long i100=0;i100<0xf;i100++){
//				readch=Flash.flashreaddata4add(4092);
//				}
//			printf(" 7result of read =%x ",readch);
//
//		}
		 
//		//在地址4092那边写上字符'4'(0x34)
//		readch=Flash.writeFlash(4092,(uint8_t)'4');
//		if(readch==FLASHSECTOEWRITEOK)//如果写成功了
//		{
//			readch=Flash.flashreaddata4add(4092);
//			printf(" result of read =%x ",readch);
//			if(readch!='4')while(1); 
//		}										 
//		else
//		{
//			printf(" the type of writeerr =%d ",readch);
//			while(1);
//		}
//
//		//在地址4092那边写上字符'4'(0x34)
//		readch=Flash.writeFlash(4092,(uint8_t)'3');
//		if(readch==FLASHSECTOEWRITEOK)//如果写成功了
//		{
//			readch=Flash.flashreaddata4add(4092);
//			printf(" result of read =%x ",readch);
//			if(readch!='3')while(1); 
//		}										 
//		else
//		{
//			printf(" the type of writeerr =%d ",readch);
//			while(1);
//		}


		Eeprom.ON();

		Eeprom.begin();

		printf(" eeprom:");delay_ms(500);
		eepromdata[0]='t';
		if(Eeprom.writeEEPROM(0xa0,0X0001,eepromdata,30)==I2CTRUE){printf("write ok ");	}
		else{printf("fail write");}
		
		Eeprom.readEEPROM(0xa0,0X0001,eepromdata,30);
		uartsendstr((char *)eepromdata, 30,PRINTFPORT);delay_ms(500);
	
		eepromdata[0]='u';
		Eeprom.writeEEPROM(0xa0,0X0001,eepromdata,30);
	
		eepromdata[0]=0;
		Eeprom.readEEPROM(0xa0,0X0001,eepromdata,30);
		uartsendstr((char *)eepromdata, 30,PRINTFPORT);delay_ms(500);
		if(eepromdata[0]!='u')
		{
			printf("wrong");
			while(1);
		}

//因为上面已经调用了eeprom.on了，和这里的trc.on是一个口PE10开，所以这里不调用trc.on也是可以的
		RTCbianliang.ON();

		RTCbianliang.begin();

//	RTCbianliang.setTime("Sun, 12/10/04 - 05:59:07");

		RTCbianliang.getTime();
		//printf("%s",RTCbianliang.timeStamp);delay_ms(1000);
		uartsendstr((char *)RTCbianliang.timeStamp, 30,PRINTFPORT);delay_ms(1000);

	sprintf(filenameorig,"m%02d%x%02d%02d.txt"\
	, RTCbianliang.year, RTCbianliang.month, RTCbianliang.date, RTCbianliang.hour);
	printf("%s",filenameorig);
		

//SD卡电源开
	printf("\n  SD poweron. ");		 
 	SD.ON();
//SD初始化，成功的话也初始化一下FAT	
	printf(" SD init. "); //SD卡那个电源开了要等一会会在初始化，要不然有些芯片初始化会有问题   
	SD.init();
	if(SD.flag!=NOTHING_FAILED)
	{
//		printf("err=%d ",flagsderr);
		printf("  failed!  ");
		while(1);
	}
	delay_ms(500);
	printf("sdsize=%dMB ",SDCardInfo.CardCapacity>>20);


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

	sprintf((char *)AddSDstr,"\r\nsys=%10d,hclk=%10d,pclk1=%10d,pclk2=%10d RTC=%s "\
		,RCC_ClocksStatus.SYSCLK_Frequency,RCC_ClocksStatus.HCLK_Frequency\
		,RCC_ClocksStatus.PCLK1_Frequency,RCC_ClocksStatus.PCLK2_Frequency\
		,RTCbianliang.timeStamp);
//		sprintf(filenameorig,"123456789.txt");
		printf("%s",filenameorig);


   	if(SD.isFile(filenameorig)==1)
	{
		printf(" yes.");
	}
	else
	{	
		printf("no.");
		printf("create it");
		if(SD.create(filenameorig))
		{
			printf(" success ");
		}
		else
		{
			printf(" failed  ");
			while(1);
		}
	}


	filesize=SD.getFileSize(filenameorig);
	if(filesize>=0)
	{
		printf("filesize=%d ",filesize);
	}
	else
	{
			printf(" WRONG! ");
			while(1);	
	}
	SD.writeSD(filenameorig, AddSDstr, filesize);


		//初始频率是168M, 然后每次减12M，减到一定程度后就开始加12M，加到一定程序又重复了
		if(flagfangxiang==0)
		{
			testsys=testsys-12000000;
			if(testsys<2000000)
			{
				testsys=168000000;
				flagfangxiang=1;
			}
		}
		else if(flagfangxiang==1)
		{
			testsys=testsys+12000000;
			if(testsys>33600000)
			{
				testsys=168000000;
				flagfangxiang=0;
			}
		}
		delay_ms(3000);
		//把现在的测试频率赋给主频这个全局变量，不过这个时候还没有变频率
		SysClkWasp=testsys;
		//做些其他准备工作，比如改掉PLLM PLLN 这些
		SysPreparePara();

		//下面只是为了打印出改过后的PLLMPLLN这些值
		zppSysClkWasp=SysClkWasp;
		zppSysPllM=SysPllM;
		zppSysPllN=SysPllN;
		zppSysPllP=SysPllP;
		zppSysPllQ=SysPllQ;
		zppSysFlashWait=SysFlashWait;
		zppDelayMsWasp=DelayMsWasp;
		zppDelayUsWasp=DelayUsWasp;
//		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, DISABLE);
//		TIM_Cmd(TIM3, DISABLE);

		//为了防止有些频率换了没法正确在单片机运行，所以先让主频为168M，把刚才要打印PLLM PLLN打印出来
		SysClkWasp=168000000;
		SysPreparePara();
		zppSystemInit();
		Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
		beginSerial(115200, PRINTFPORT); //ZPPDelay(0x3456789);

		printf("\npreclk=%d,pllm=%d,n=%d,p=%d,q=%d,wait=%d delayms=%d delayus=%d "\
		,zppSysClkWasp,zppSysPllM,zppSysPllN,zppSysPllP,zppSysPllQ,zppSysFlashWait,zppDelayMsWasp,zppDelayUsWasp);		
		//delay_ms(2000);

		//这个时候重新变到真正要变的主频来
		SysClkWasp=testsys;
		SysPreparePara();
		//在变主频之前先关掉之前那些功能开关
//		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, DISABLE);
//		TIM_Cmd(TIM3, DISABLE);	ZPPDelay(0x23456);
		//while(1)循环的下一句就是执行自己做的系统初始化，那个就是真正的开始变频率操作了.
	}	
}
#endif // end sendachar




