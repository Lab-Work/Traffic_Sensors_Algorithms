/*
这个是板子的I2C例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：   
#define BTESTEEPROM 0		 EEPROM	  读写
#define BTESTI2CRTC 1		 外部RTC 读时间 写时间 读温度
#define BTESTMLX90614 2		 MLX90614 读温度 用的SMBUS协议 //此程序仅适合总线上就一个红外传感器的情况
#define BTESTI2CNULL 100	 不选用I2C例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTXIAOMO 0

#define BTESTXIAOMONULL 100

#define EXAMPLEXIAOMO BTESTXIAOMONULL
/*
写数据，读数据
写一个数据然后读出来检测对不对
*/
//EEPROM
#if EXAMPLEXIAOMO==BTESTXIAOMO
void setup()
{

	  Mux_poweron();
    monitor_onuart3TX();
	  beginSerial(115200, 3);
		
/*	
	monitor_offuart3TX();monitor_onuart3RX();//用USB监控一下xbee接收	
	beginSerial(9600, 1); chooseuartinterrupt(1);
	printf("1");
	printf("2");	
*/
	
	
	//beginSerial(9600, 3);

	serialWrite('3', 3);//send a char '3' by uart3 //查询方式
	serialWrite('C', 3);//send a char 'c'
	delay_ms(1000);
	
	serialWritestr("\r\nHello world!\n", 15,3);	
	
	
	beginMb7076();
	
	SMBus_Init();
	SMBus_Apply();
	
}

char value_str[10];
char sl_add_char[20];
char temps[40];

void loop()
{
	int valuesonar;
	int flagsonar;	
	
	unsigned int slave_add;
	float tempobject;
	float tempambient;

	
	flagsonar=readMb7076(&valuesonar);
	
	sprintf(value_str,"Dist: %d",valuesonar);
	
	serialWritestr(value_str,10,3);	
	serialWrite('\n', 3);
	delay_ms(1000);
	
	slave_add= readMlx90614Subadd(); 
	sprintf(sl_add_char,"Slave address:%x",slave_add);
	serialWritestr(sl_add_char,20,3);	
	serialWrite('\n', 3);
	
	delay_ms(200);
	
	tempobject=readMlx90614ObjectTemp(slave_add);
	tempambient=readMlx90614AmbientTemp(slave_add);	
	sprintf(temps,"temp obj: %f'C, temp amb: %f'C",tempobject, tempambient);
	serialWritestr(temps,40,3);	
	serialWrite('\n', 3);

	
	
}
#endif //end EEPROM



