/*
这个是板子的LED例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：    
#define BTESTUTILSLED 0		 LED例程
#define BTESTUTILSNULL 100	 不选用Util例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTUTILSLED1 0
#define BTESTUTILSNULL1 100

//#define EXAMPLEUTILS1  BTESTUTILSLED1
#define EXAMPLEUTILS1  BTESTUTILSNULL1

//LED
#if EXAMPLEUTILS1==	BTESTUTILSLED1
void setup()
{
	  Mux_poweron(); 
    beginSerial(9600, 1);	
Utils.initLEDs();
}

void loop()
{
	uint16_t i=0;	


	for(i=0;i<5;i++){
		Utils.blinkLEDs(1000);
	}
	
	printByte('a',1);
	
	delay_ms(1000);
	
}
#endif // end LED




