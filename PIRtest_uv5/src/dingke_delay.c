#include "dingke_delay.h"
#include "dingke_sysclkchange.h"
////////////////////////////////////////////
//1.这里的两个延时都是死等待的方式，
//比如当调用delay_ms了，就会SysTick_Config(168000); 在stm32f4xx_it.c中有个SysTick_Handler(void)就会1ms产生一个中断（原理我还不知道，看现象是这样）
//当中断了就 ntime--;这样在delay_ms(u16 nms)只要读一下ntime有没有变成0，到0了就说明时间的到了,然后禁止这种中断
//这样退出delay_ms(u16 nms)。所以说这里延时就是简单的死等待
//2.千万不要忘了和	stm32f4xx_it.c 配合使用
//3.这里不需要delay初始化之类的函数
//4.当然这里的延时实际上是通过设置系统嘀嗒时钟中断做到的
////////////////////////////////////////////
extern __IO u32 ntime;								    
//使用SysTick进行精确延时

//ms延时
void delay_ms(u16 nms)
{	 		  	  
	ntime=nms;
	
	if(SysClkWasp>CLKCHANGEDELAY1);
	else if(SysClkWasp>CLKCHANGEDELAY2) ntime = ntime<<1;
	else  ntime = ntime<<2;

	SysTick_Config(DelayMsWasp);
	//SysTick_Config(168000);//1ms产生一次中断,并对ntime减1 //针对系统时钟为168M的情况
	while(ntime);//等待时间到达

	SysTick->CTRL=0x00;			  	    
}   
//us延时		    								   
void delay_us(u32 nus)	//这里暂时用10us来代替，主要是在RTC里面用了us来延时，但是这个时间太频繁了，进系统时钟中断后没有出来去又进去
{		
	ntime=nus;

	if(SysClkWasp>CLKCHANGEDELAY1);
	else if(SysClkWasp>CLKCHANGEDELAY2) ntime = ntime<<1;
	else  ntime = ntime<<2;

//	SysTick_Config(672);//1us产生一次中断,并对ntime减1
	SysTick_Config(DelayUsWasp*4);//1us产生一次中断,并对ntime减1
	
//	while(1)
//	{
//		if((ntime==0)||(ntime>0xffff0000))break;
//	}
	while(ntime);//等待时间到达


	SysTick->CTRL=0x00;
}




























