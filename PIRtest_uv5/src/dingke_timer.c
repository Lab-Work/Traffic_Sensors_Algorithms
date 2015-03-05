
//这里主要是定时器3 和定时器2 以及和定时器3有关的那个delay()
//#include "timer.h"
#include "dingke_timer.h"

//通用定时器中断初始化

//arr：自动重装值。
//psc：时钟预分频数
//这里使用的是定时器3!
//目前这个是1ms的样子
void Timer3_Init(u16 cnt,u32 hz)
{
    TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
  RCC_ClocksTypeDef RCC_ClocksStatus;
  uint32_t flagapb=1;
  uint32_t apbtmclk;


  if(cnt==0)cnt=2;
  else if(cnt==1){cnt=10;hz=hz*10; }

  if(hz==0)hz=1;
  else if(hz>1000000)hz=1000000;



  RCC_GetClocksFreq(&RCC_ClocksStatus);
  if(RCC_ClocksStatus.SYSCLK_Frequency==16000000)//内部晶振
  {
	TIM_TimeBaseStructure.TIM_Prescaler =(16000000/hz-1); //设置用来作为TIMx时钟频率除数的预分频值  10Khz的计数频率   
  }
  else
  {
  	if(RCC_ClocksStatus.PCLK1_Frequency==RCC_ClocksStatus.SYSCLK_Frequency);
	else flagapb=2;
	apbtmclk=RCC_ClocksStatus.PCLK1_Frequency*flagapb;
	TIM_TimeBaseStructure.TIM_Prescaler =(apbtmclk/hz-1); //设置用来作为TIMx时钟频率除数的预分频值  10Khz的计数频率  
  }

	TIM_TimeBaseStructure.TIM_Period = cnt-1; //设置在下一个更新事件装入活动的自动重装载寄存器周期的值	 计数到5000为500ms

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE); //时钟使能

	TIM_TimeBaseStructure.TIM_ClockDivision = 0; //设置时钟分割:TDTS = Tck_tim
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //TIM向上计数模式
	TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure); //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
 
	TIM_ITConfig(  //使能或者失能指定的TIM中断
		TIM3, //TIM2
		TIM_IT_Update  |  //TIM 中断源
		TIM_IT_Trigger,   //TIM 触发中断源 
		ENABLE  //使能
		);
	NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;  //TIM3中断
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //先占优先级0级
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;  //从优先级3级
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //IRQ通道被使能
	NVIC_Init(&NVIC_InitStructure);  //根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器

	TIM_Cmd(TIM3, ENABLE);  //使能TIMx外设							 
}
volatile unsigned long timer0_overflow_count=0;
uint32_t Flag1ms;
void TIM3_IRQHandler(void)   //TIM3中断
{
	int flagerr;
	if (TIM_GetITStatus(TIM3, TIM_IT_Update) != RESET) //检查指定的TIM中断发生与否:TIM 中断源 
		{
			TIM_ClearITPendingBit(TIM3, TIM_IT_Update  );  //清除TIMx的中断待处理位:TIM 中断源 

	//GPIOC->ODR^=GPIO_Pin_13;//监控用的，以后要删掉
			timer0_overflow_count++;
			
			Flag1ms++;


//				flagerr=handle1msprerx80();
//				if(flagerr==0)printf("A ");
//				else printf("%c ",flagerr+'A');	



			/*
			if(timer0_overflow_count>10000)
			{  
				// Jump to user application //
				JumpAddress = *(__IO uint32_t*) (USER_FLASH_STARTADDRESS + 4);
				Jump_To_Application = (pFunction) JumpAddress;
		      	// Initialize user application's Stack Pointer //
				__set_MSP(*(__IO uint32_t*) USER_FLASH_STARTADDRESS);
				Jump_To_Application();			

				
			}*/
		}
}


//和以前一样功能
//这个里面的值变化是靠定时器3的1ms中断来做到的
unsigned long millis()
{
	return timer0_overflow_count ;

}


//和以前一样功能
//这个函数只有开启了tim3才有用，否则程序就死在这边了
void delay(unsigned long ms)
{
	unsigned long start = millis();
	
	while (millis() - start < ms);
}

unsigned char  Timer2_Init(u16 cnt,u32 hz)
{
	TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	RCC_ClocksTypeDef RCC_ClocksStatus;
	uint32_t flagapb=1;
	uint32_t apbtmclk;
	
	uint32_t prescaler;
	uint32_t period;

  	RCC_GetClocksFreq(&RCC_ClocksStatus);

	while(1)
	{
		if(cnt==0)cnt=2;
		else if(cnt==1){cnt=10;hz=hz*10; }
		
		if(hz==0)hz=1;
		else if(hz>1000000)hz=1000000;
		
		if(RCC_ClocksStatus.SYSCLK_Frequency==16000000)//内部晶振
		{
			prescaler =(16000000/hz-1); //设置用来作为TIMx时钟频率除数的预分频值  10Khz的计数频率   
		}
		else
		{
			if(RCC_ClocksStatus.PCLK1_Frequency==RCC_ClocksStatus.SYSCLK_Frequency);
			else flagapb=2;
			apbtmclk=RCC_ClocksStatus.PCLK1_Frequency*flagapb;
			prescaler =(apbtmclk/hz-1); //设置用来作为TIMx时钟频率除数的预分频值  10Khz的计数频率  
		}
	
		period = cnt-1; //设置在下一个更新事件装入活动的自动重装载寄存器周期的值	 计数到5000为500ms
		

		if(period>0xffff)return 1;
		else 
		{
			if(prescaler>0xffff)
			{
				hz=hz*10;
				cnt=cnt*10;
				if(hz>10000000)return 2;
			}
			else break;		
		}
		
	}

	TIM_TimeBaseStructure.TIM_Prescaler =prescaler;
	TIM_TimeBaseStructure.TIM_Period = period;


	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE); //时钟使能

	TIM_TimeBaseStructure.TIM_ClockDivision = 0; //设置时钟分割:TDTS = Tck_tim
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //TIM向上计数模式
	TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure); //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
 
	TIM_ITConfig(  //使能或者失能指定的TIM中断
		TIM2, //TIM2
		TIM_IT_Update  |  //TIM 中断源
		TIM_IT_Trigger,   //TIM 触发中断源 
		ENABLE  //使能
		);
	NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;  //TIM3中断
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //先占优先级0级
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;  //从优先级3级
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //IRQ通道被使能
	NVIC_Init(&NVIC_InitStructure);  //根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器

	TIM_Cmd(TIM2, ENABLE);  //使能TIMx外设	
	return 0;						 
}

void TIM2_IRQHandler(void)   //TIM2中断
{
	
	if (TIM_GetITStatus(TIM2, TIM_IT_Update) != RESET) //检查指定的TIM中断发生与否:TIM 中断源 
	{
		TIM_ClearITPendingBit(TIM2, TIM_IT_Update  );  //清除TIMx的中断待处理位:TIM 中断源 
		GPIO_ToggleBits(GPIOE, GPIO_Pin_1);
//			handle1msprerx80();
	}
}

