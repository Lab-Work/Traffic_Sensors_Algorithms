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
#define BTESTPWRNULL 100
#define EXAMPLEPWR BTESTPWRNULL 


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
	
	printf("\r\n SD poweron. \r\n");		 
 	SD.ON();//intilise SD 

	printf("SD init. \r\n");    
	SD.init();
	
		if(SD.isFile("Power.txt")==1)
	{
		printf(" Power.txt yes. ");
	}
	else
	{
		printf("  There is no folder dir0, now create the folder. ");
		if(SD.create("Power.txt"))
		{
			printf(" success ");
		}
		else
		{
			printf(" failed  ");
		}
	}			
	
}

uint8_t StrFileName[20]="Power.txt";// Define your txt file name, here it is GPS
static uint8_t  StrWriteSDPower[200];// Define a string to be writen into the SD
static long  OffsetWriteSDPower=0;  //Define the offset length of the current writing 
static long  LenWriteSDPower=0;  //length of the string to be written 
static uint8_t  StrWriteSD[200];
static long  OffsetWriteSD=0;
static long  LenWriteSD=0;
//uint8_t StrFileName[20]="test.txt";//名字会随着时间改的


void loop()
{
	int16_t temp_voltage;
	int16_t temp_current;
	int16_t temp_temp;

	PWR2745.switchesOFF(PWR_XBEE|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(3000);//因为电路那边大概的电容的问题，所以置高置底长一点时间，测出来的电流区别才比较大	

	delay_ms(100);	
	temp_voltage=PWR2745.getBatteryVolts();
	printf("\r\n");
	printf("Voltage=%dmv  ",temp_voltage);
	delay_ms(100);

	temp_temp=PWR2745.getBatterytemperature();
	printf("    temperature=%d'c  " ,temp_temp);
	delay_ms(100);

	temp_current=PWR2745.getBatteryCurrent();
	printf("Current=%dma  ",temp_current);
	delay_ms(100);

sprintf((char *)StrWriteSD,"\r\n Voltage=%d  Temperature=%d  Current=%d",temp_voltage, temp_temp, temp_current);
			
		OffsetWriteSD=SD.getFileSize((const char *)StrFileName);
		LenWriteSD= strlen((const char *)StrWriteSD);
		SD.writeSD((const char *)StrFileName,StrWriteSD, OffsetWriteSD);
		OffsetWriteSD += LenWriteSD;
		if(OffsetWriteSD>0xfffffff)OffsetWriteSD=0;
		
		
	

	PWR2745.switchesON(PWR_XBEE|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6); delay_ms(5000);	

	delay_ms(100);	
	temp_voltage=PWR2745.getBatteryVolts();
	printf("\r\n");
	printf("    Voltage=%dmv  ",temp_voltage);
	delay_ms(100);

	temp_temp=PWR2745.getBatterytemperature();
	printf("temperature=%d'c  " ,temp_temp);
	delay_ms(100);

	temp_current=PWR2745.getBatteryCurrent();
	printf("Current=%dma  ",temp_current);
	delay_ms(500);
}
#endif // end PWR168SIMPLE

