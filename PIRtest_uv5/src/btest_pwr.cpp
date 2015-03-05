/*
这个是板子的功耗例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：
#define BTESTPWR168SIMPLE 0	 简单的功耗测量.在168M主频下打开RTC（I2C），SD卡，串口2，电池管理芯片，测试电流电压并记录在SD卡里面，这算是最简单的测试功耗了
#define BTESTPWR168SLEEP 1	 168M主频进入睡眠 及唤醒
#define BTESTPWR168STOP 2	 168M主频进入STOP 及唤醒
#define BTESTPWR168STANDBY 3 168M主频进入STANDBY 及唤醒 //这种模式进去就唤不醒了 而且烧程序也烧不进去了  如果一旦进入了这个模式想重新烧写程序可以让boot0=1下用JLINK烧写程序 
#define BTESTPWRNULL 100     不选用功耗例程

不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()


*/
#include "allboardinc.h"
#define BTESTPWR168SIMPLE 0	  //打开RTC（I2C），电池管理芯片，串口1，测试电流电压并打印出来
#define BTESTPWR168SLEEP 1	  //__WFI();
#define BTESTPWR168STOP 2
#define BTESTPWR168STANDBY 3//这种模式进去就唤不醒了 而且烧程序也烧不进去了  和重启单片机没有什么异样	 


#define BTESTPWRCHANGECLK 4	 //换主频测电流
#define BTESTPWRNULL 100

#define EXAMPLEPWR BTESTPWRNULL//

//#define EXAMPLEPWR BTESTPWR168SIMPLE//

//PWR168SIMPLE
#if EXAMPLEPWR==BTESTPWR168SIMPLE
//static uint8_t AddSDstr[512];

//程序开头要打印一下频率
void setup()
{
	Utils.initLEDs();
	Utils.setLED(0, 1);
	Utils.setLED(1, Utils.getLED(0));
	Utils.setLED(2, Utils.getLED(0));
	delay_ms(100);

	//three led cut down
	Utils.setLED(0, 0);
	Utils.setLED(1, Utils.getLED(0));
	Utils.setLED(2, Utils.getLED(0));
	delay_ms(500);


	PWR2745.initPower(PWR_BAT|PWR_RTC);//必须这两个都开读电流数据才能对，这个是因为RTC和读电流电压芯片公用I2C,要是只开一个读数据就会有问题
	PWR2745.switchesON(PWR_BAT|PWR_RTC);
	
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据	
	monitor_onuart3TX();  monitor_offuart3RX();
	beginSerial(115200, PRINTFPORT); //串口
	printf ("\r\n\r\n                  Designed by ZHP,    Built: %s %s ", __DATE__, __TIME__);
	printf("\r\nfile: %s ,",__FILE__); 
	printf("\r\nfunction: %s ,",__func__); 
	printf("\r\nline: %d\r\n",__LINE__); 	

	PWR2745.initBattery();//初始化读电流电压温度那个，实际上用的是I2C的协议	
	PWR2745.initPower(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6);//初始化这些GPIO口为了测开这些口和不开这些口的电流差别
}

void loop()
{
	int16_t tempb;

	PWR2745.switchesOFF(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(3000);//因为电路那边大概的电容的问题，所以置高置底长一点时间，测出来的电流区别才比较大	

	delay_ms(100);	
	tempb=PWR2745.getBatteryVolts();
	printf("\r\n");
	printf("Voltage=%dmv  ",tempb);
	delay_ms(100);

	tempb=PWR2745.getBatterytemperature();
	printf("    temperature=%d'c  " ,tempb);
	delay_ms(100);

	tempb=PWR2745.getBatteryCurrent();
	printf("Current=%dma  ",tempb);
	delay_ms(500);

	PWR2745.switchesON(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(5000);	

	delay_ms(100);	
	tempb=PWR2745.getBatteryVolts();
	printf("\r\n");
	printf("    Voltage=%dmv  ",tempb);
	delay_ms(100);

	tempb=PWR2745.getBatterytemperature();
	printf("temperature=%d'c  " ,tempb);
	delay_ms(100);

	tempb=PWR2745.getBatteryCurrent();
	printf("Current=%dma  ",tempb);
	delay_ms(500);
}
#endif // end PWR168SIMPLE


/*
睡眠模式
    进入：调用__WFI(); 感觉就想程序在那边暂停了
	唤醒：任一一个中断，比如程序里面定义了一个外部中断（PA0口）,比如程序中用了串口PRINTFPORT，初始化里面隐含了打开了接受中断
	一旦来了中断，程序就唤醒。感觉就是程序在暂停的那个地方继续执行了。
	
	****** 注意 ******
	*因为任意中断都可以唤醒，所以假如开启了定时器，比如开xbee的时候就会顺带把定时器3开启，那个是1ms一个中断的， 进入睡眠 1ms就醒了 给人感觉还以为是没有睡觉呢*
	*******************
*/
//这个例程给出的是通过按wakeup按键唤醒 或者通过电池那边的PC1上升沿唤醒
//操作： 1）进入sleep后，按wakeup按键 , 2) 进入sleep后，短路R40来达到模拟电池复位，然后就能唤醒单片机了
//在 GPIO_def();exti_def();NIVC_def();里面还有关于RTC闹钟的唤醒，不过操作有点繁琐，相关程序就不写在这里了，在 btestdata0902.cpp里面有相关例程
#if EXAMPLEPWR==BTESTPWR168SLEEP
void setup()
{
	Utils.initLEDs();
	Utils.setLED(0, 1);
	Utils.setLED(1, Utils.getLED(0));
	Utils.setLED(2, Utils.getLED(0));
	delay_ms(100);

	PWR2745.initPower(PWR_BAT);//必须开这个电池电源，要不然PC1（负责电池电量比较2.8V的）不起作用
	PWR2745.switchesON(PWR_BAT);
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据	
	beginSerial(115200, PRINTFPORT); //串口
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
	
	//three led cut down
	Utils.setLED(0, 0);
	Utils.setLED(1, Utils.getLED(0));
	Utils.setLED(2, Utils.getLED(0));
	delay_ms(500);	
	while(1)
	{
	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(500);
	CNTEXIT++;

	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(500);
	CNTEXIT++;
	if(CNTEXIT>=20)break;

	}

	//enter sleep mode, if any interrupt(such as key PA0 put down ,or uart PRINTFPORT receive datas ) hanppen, MCU can wake
    __WFI();//进入睡眠，假如有任何中断都可以唤醒它，这里的中断比如外部中断，串口接收（接收中断开的话），定时器
	//按按键wakeup就可以唤醒单片机
	//
	//
	printf("I am back! hello\n");

	while(1)
	{
		printf("cnt=%d ",CNTEXIT);
		Utils.blinkLEDs(1000);
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
	while(1);
}
#endif // end BTESTPWR168SLEEP

/*
STOP模式
    进入：调用PWR_EnterSTOPMode(PWR_Regulator_ON, PWR_STOPEntry_WFI); 感觉就想程序在那边暂停了
	唤醒：任一一个外部中断，比如程序里面定义了一个外部中断（PA0口）,但这里串口接收不是
	一旦来了中断，程序就唤醒。感觉就是程序在暂停的那个地方继续执行了，但是此时系统时钟恢复到了内部时钟，所以要系统时钟初始化一下
*/
#if EXAMPLEPWR==BTESTPWR168STOP
void setup()
{
	beginSerial(9600, PRINTFPORT); //串口
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
	while(1)
	{
	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(500);
	CNTEXIT++;

	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(500);
	CNTEXIT++;
	if(CNTEXIT>=20)break;
	}

	printf("%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0x40007004));

//	*(__IO uint32_t *)(0xe000ed10) |=0x02;
//	*(__IO uint32_t *)(0x40007004) |=0x100;

	printf("now=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("sec=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("third=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0x40007004));

	//enter sleep mode, if any ex interrupt(such as key PA0 put down ,but not uart PRINTFPORT receive datas ) hanppen, MCU can wake	
	PWR_EnterSTOPMode(PWR_Regulator_ON, PWR_STOPEntry_WFI);
    //when MCU wake, it sysclk is internal clk. so we must reuse SystemInit();	
	SystemInit();
	printf("I am back!mei hei\n");

	while(1)
	{
		printf("cnt=%d ",CNTEXIT);
		Utils.blinkLEDs(1000);
		CNTEXIT++;
		if(CNTEXIT>=2000)CNTEXIT=0;
		if(CNTEXIT==40)
		{
			PWR_EnterSTOPMode(PWR_Regulator_ON, PWR_STOPEntry_WFI);
		    SystemInit();
			printf("I am back!mei hei\n");		   	
		}
	}
}

void loop()
{
	while(1);
}
#endif // end BTESTPWR168STOP

/*
STANDBY模式
    进入：调用PWR_EnterSTANDBYMode(); 目前感觉就像程序死掉了
	唤醒：固定的几个唤醒
	因为这里没有使用那几个固定的唤醒，所以现在唤不醒，而且重新烧写程序也写不了，此时要烧写可以把 boot0=1烧写.
	
	注意：此模式下寄存器什么的都关掉了，程序的那些数据也不能保存，所以不建议客户使用此模式

*/
#if EXAMPLEPWR==BTESTPWR168STANDBY
void setup()
{
	beginSerial(9600, PRINTFPORT); //串口
	GPIO_def();//PA0
	exti_def();//PA0
	NIVC_def();//PA0
	while(1)
	{
	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(500);
	CNTEXIT++;

	printf("cnt=%d ",CNTEXIT);
	Utils.blinkLEDs(200);
	CNTEXIT++;
	if(CNTEXIT>=20)break;
	}

	printf("%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0x40007004));

//	*(__IO uint32_t *)(0xe000ed10) |=0x02;
//	*(__IO uint32_t *)(0x40007004) |=0x100;

	printf("now=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("sec=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("third=%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0xe000ed10));
	printf("%x ",*(__IO uint32_t *)(0x40007004));


	PWR_EnterSTANDBYMode();
	//enter sleep mode, if any ex interrupt(such as key PA0 put down ,but not uart PRINTFPORT receive datas ) hanppen, MCU can wake	
    //when MCU wake, it sysclk is internal clk. so we must reuse SystemInit();	
	SystemInit();
	printf("I am back!mei hei\n");

	while(1)
	{
		printf("cnt=%d ",CNTEXIT);
		Utils.blinkLEDs(1000);
		CNTEXIT++;
		if(CNTEXIT>=2000)CNTEXIT=0;
		if(CNTEXIT==40)
		{
			PWR_EnterSTOPMode(PWR_Regulator_ON, PWR_STOPEntry_WFI);
		    SystemInit();
			printf("I am back!mei hei\n");		   	
		}
	}
}

void loop()
{
	while(1);
}
#endif // end BTESTPWR168STANDBY





//BTESTPWRCHANGECLK
#if EXAMPLEPWR==BTESTPWRCHANGECLK


//程序开头要打印一下频率

void setup()
{
	unsigned long testsys=168000000;
	int16_t tempb;	
	RCC_ClocksTypeDef RCC_ClocksStatus;
	
	while(1)
	{
		zppSystemInit();
		Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
			delay_ms(10);	
		beginSerial(115200, PRINTFPORT); //串口	
		RCC_GetClocksFreq(&RCC_ClocksStatus);
		printf("\r\n\r\nsys=%d,hclk=%d,pclk1=%d,pclk2=%d"\
		,RCC_ClocksStatus.SYSCLK_Frequency,RCC_ClocksStatus.HCLK_Frequency\
		,RCC_ClocksStatus.PCLK1_Frequency,RCC_ClocksStatus.PCLK2_Frequency);
		delay_ms(10);	
		Utils.initLEDs();
		Utils.setLED(0, 1);
		Utils.setLED(1, Utils.getLED(0));
		Utils.setLED(2, Utils.getLED(0));
		delay_ms(100);
	
		//three led cut down
		Utils.setLED(0, 0);
		Utils.setLED(1, Utils.getLED(0));
		Utils.setLED(2, Utils.getLED(0));
		delay_ms(500);

		PWR2745.initPower(PWR_BAT|PWR_RTC);//必须这两个都开读电流数据才能对，这个是因为RTC和读电流电压芯片公用I2C,要是只开一个读数据就会有问题
		PWR2745.switchesON(PWR_BAT|PWR_RTC);//开启这个检测电池电流的才能起作用 
		PWR2745.initBattery();//初始化读电流电压温度那个，实际上用的是I2C的协议		
		PWR2745.initPower(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6);//初始化这些GPIO口为了测开这些口和不开这些口的电流差别	
	
	
		RTCbianliang.ON(); delay_ms(10);   //因为上面已经开了RTC电源了，所以这里的是可以省掉的
		RTCbianliang.begin();delay_ms(10); //因为上面已经开了I2C初始化了，所以这里的是可以省掉的

		RTCbianliang.getTime();
		printf("\r\n  %s",RTCbianliang.timeStamp);
		delay_ms(20);

		PWR2745.switchesOFF(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(5000);//因为电路那边大概的电容的问题，所以置高置底长一点时间，测出来的电流区别才比较大	
	
		delay_ms(100);	
		tempb=PWR2745.getBatteryVolts();
		printf("\r\n");
		printf("  Voltage=%dmv  ",tempb);
		delay_ms(100);
	
		tempb=PWR2745.getBatterytemperature();
		printf("    temperature=%d'c  " ,tempb);
		delay_ms(100);
	
		tempb=PWR2745.getBatteryCurrent();
		printf("Current=%dma  ",tempb);
		delay_ms(500);
	
		PWR2745.switchesON(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(5000);	
	
		delay_ms(100);	
		tempb=PWR2745.getBatteryVolts();
		printf("\r\n");
		printf("      Voltage %dmv  ",tempb);
		delay_ms(100);
	
		tempb=PWR2745.getBatterytemperature();
		printf("temperature %d'c  " ,tempb);
		delay_ms(100);
	
		tempb=PWR2745.getBatteryCurrent();
		printf("Current %dma  ",tempb);
		delay_ms(500);


			testsys=testsys-12000000;
			if(testsys<24000000)
			{
				testsys=168000000;
			}
		//把现在的测试频率赋给主频这个全局变量，不过这个时候还没有变频率
		SysClkWasp=testsys;
		//做些其他准备工作，比如改掉PLLM PLLN 这些
		SysPreparePara();
	}

}

void loop()
{
while(1);
}
#endif // end BTESTPWRCHANGECLK



