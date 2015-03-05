#include "dingke_exti.h"

//外部中断初始化使能
uint16_t CNTEXIT=0;
//uint8_t  FlagExit12=0;
void GPIO_def(void)
{
	  GPIO_InitTypeDef  GPIO_InitStructured;

  	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);	*/
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);

  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_0;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructured);

  	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);	*/
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_1;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
//  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
//  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOC, &GPIO_InitStructured);


  	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);	*/
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_12;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
//  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
//  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOE, &GPIO_InitStructured);

}
//定义中断线
void exti_def(void)
{
	EXTI_InitTypeDef EXTI_InitStruct;
	SYSCFG_EXTILineConfig(EXTI_PortSourceGPIOA ,EXTI_PinSource0 );
//	
	EXTI_InitStruct.EXTI_Line = EXTI_Line0;
	EXTI_InitStruct.EXTI_Mode = EXTI_Mode_Interrupt;
	EXTI_InitStruct.EXTI_Trigger = EXTI_Trigger_Falling;
	EXTI_InitStruct.EXTI_LineCmd = ENABLE;
	
	EXTI_Init(&EXTI_InitStruct);

  /* Enable SYSCFG clock */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);

	SYSCFG_EXTILineConfig(EXTI_PortSourceGPIOC ,EXTI_PinSource1 );
	
	EXTI_InitStruct.EXTI_Line = EXTI_Line1;
	EXTI_InitStruct.EXTI_Mode = EXTI_Mode_Interrupt;
	EXTI_InitStruct.EXTI_Trigger = EXTI_Trigger_Rising;
	EXTI_InitStruct.EXTI_LineCmd = ENABLE;
	
	EXTI_Init(&EXTI_InitStruct);



	SYSCFG_EXTILineConfig(EXTI_PortSourceGPIOE ,EXTI_PinSource12 );
	
	EXTI_InitStruct.EXTI_Line = EXTI_Line12;
	EXTI_InitStruct.EXTI_Mode = EXTI_Mode_Interrupt;
	EXTI_InitStruct.EXTI_Trigger = EXTI_Trigger_Falling;
	EXTI_InitStruct.EXTI_LineCmd = ENABLE;
	
	EXTI_Init(&EXTI_InitStruct);

	
}

void NIVC_def(void)
{
	NVIC_InitTypeDef NVIC_InitStruct;
	NVIC_InitStruct.NVIC_IRQChannel=EXTI0_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 0;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 0;
	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
	NVIC_PriorityGroupConfig( NVIC_PriorityGroup_0);
	NVIC_Init(&NVIC_InitStruct);


	NVIC_InitStruct.NVIC_IRQChannel=EXTI1_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 1;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 1;
	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
//	NVIC_PriorityGroupConfig( NVIC_PriorityGroup_0);
	NVIC_Init(&NVIC_InitStruct);


	NVIC_InitStruct.NVIC_IRQChannel=EXTI15_10_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 12;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 12;
	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
//	NVIC_PriorityGroupConfig( NVIC_PriorityGroup_0);
	NVIC_Init(&NVIC_InitStruct);

}

void setPowerEnoughAsAwake(void)
{
	GPIO_InitTypeDef  GPIO_InitStructured;
	EXTI_InitTypeDef EXTI_InitStruct;
	NVIC_InitTypeDef NVIC_InitStruct;
	/*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);	*/
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);
	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_1;
	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_IN;
	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOC, &GPIO_InitStructured);		
	

	/* Enable SYSCFG clock */
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);
	
	SYSCFG_EXTILineConfig(EXTI_PortSourceGPIOC ,EXTI_PinSource1 );	
	EXTI_InitStruct.EXTI_Line = EXTI_Line1;
	EXTI_InitStruct.EXTI_Mode = EXTI_Mode_Interrupt;
	EXTI_InitStruct.EXTI_Trigger = EXTI_Trigger_Rising;
	EXTI_InitStruct.EXTI_LineCmd = ENABLE;	
	EXTI_Init(&EXTI_InitStruct);
				
	NVIC_InitStruct.NVIC_IRQChannel=EXTI1_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority= 0;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority= 0;
	NVIC_InitStruct.NVIC_IRQChannelCmd= ENABLE;
	NVIC_Init(&NVIC_InitStruct);

}
void EXTI0_IRQHandler(void) 
{	
	CNTEXIT=100;  
	EXTI_ClearITPendingBit(EXTI_Line0); //清除LINE10上的中断标志位


}
void EXTI1_IRQHandler(void) 
{	
	CNTEXIT=200;  
	EXTI_ClearITPendingBit(EXTI_Line1); //清除LINE10上的中断标志位


}
void EXTI15_10_IRQHandler(void) 
{
	uint8_t temp;
	int flagerr;	
	CNTEXIT=300;  
	EXTI_ClearITPendingBit(EXTI_Line12); //清除LINE10上的中断标志位
//	FlagExit12=1;
}

