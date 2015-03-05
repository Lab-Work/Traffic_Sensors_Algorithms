/*
这个是板子的LED例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：    
#define BTESTWDG 0		 看门狗例程
#define BTESTWDGNULL 100	 不选用
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()

*/
#include "allboardinc.h"
#define BTESTWDG 0
#define BTESTWDGNULL 100

#define EXAMPLEWDG  BTESTWDGNULL

//LED
#if EXAMPLEWDG==	BTESTWDG
void setup()
{
	uint16_t i=0;	
	Utils.initLEDs();
	for(i=0;i<5;i++)
		Utils.blinkLEDs(100);
	delay_ms(2000);
    //stm32f4的看门狗频率为32KHZ，下面选择的预分频64 是频率/64的意思 也就是频率变为0.5KHZ，即2毫秒为一个基准单位
	//在这样的频率下，例子选择了装在1000，也就是2秒需要喂一次狗，超时就单片机复位
	IWDG_Init(IWDG_Prescaler_64,1000);//2s为超出时间
	IWDG_Feed();
	//下面是想分别1.7秒 1.9秒 2.1秒 ... 喂一次狗，当执行了i=800, i=900 之后，执行i=1000时，先灯亮了1秒，然后灭一秒，然后再延时100毫秒，显然这个时候超过了喂狗的时间，所以的单片机会复位
	//看到的现象是这里的灯应该亮了三次，单片机复位了。其中最后一次是灯灭的状态没有走完，超时复位了。
	for(i=800;i<2000;i=i+100)
	{
		Utils.setLED(1, 1);
		delay_ms(i);
		Utils.setLED(1, 0);
		delay_ms(i);
		delay_ms(100);
		IWDG_Feed();			
	}
//	while(1)
//	IWDG_Feed();
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




