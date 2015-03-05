/*
这个是板子的LED例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：    
#define BTESTUTILSLED 0		 LED例程
#define BTESTUTILSNULL 100	 不选用Util例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTUTILSLED 0
#define BTESTUTILSNULL 100

#define EXAMPLEUTILS  BTESTUTILSNULL
//#define EXAMPLEUTILS  BTESTUTILSLED

//LED
#if EXAMPLEUTILS==	BTESTUTILSLED
void setup()
{
Utils.initLEDs();
}

void loop()
{
	uint16_t i=0;	

	for(i=0;i<10;i++)
	{
		//three led light up
		Utils.setLED(0, 1);
		Utils.setLED(1, Utils.getLED(0));
		Utils.setLED(2, Utils.getLED(0));
		delay_ms(100);

		//three led cut down
		Utils.setLED(0, 0);
		Utils.setLED(1, Utils.getLED(0));
		Utils.setLED(2, Utils.getLED(0));
		delay_ms(500);
	}

	for(i=0;i<5;i++)
		Utils.blinkLEDs(1000);
	
}
#endif // end LED




