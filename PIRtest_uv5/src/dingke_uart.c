//这里串口1 2 3 6 ，  相关的xbee 声纳 串口6多路选择
//#define ZPPTEMP 0xff 
#include "stm32f4xx_conf.h"
#include "dingke_uart.h"
#include <string.h>

//#include <string.h>
//#ifndef __WASPCONSTANTS_H__
//	#include "WaspConstants.h"
//#endif


#define	UART_LIMIT (RX_BUFFER_SIZE+100)

	unsigned char Rx80Buffer[RX80_BUFFER_SIZE];

	unsigned char rx_buffer3[RX_BUFFER_SIZE_3];
	unsigned char rx_buffer2[RX_BUFFER_SIZE_2];
	unsigned char rx_buffer1[RX_BUFFER_SIZE_1];
	unsigned char rx_buffer6[RX_BUFFER_SIZE_6];
#if USEUSART4==1
	unsigned char rx_buffer4[RX_BUFFER_SIZE_4];
#endif
#if USEUSART5==1
	unsigned char rx_buffer5[RX_BUFFER_SIZE_5];
#endif


	int rxprerx80_buffer_tail3 = 0;

	int Rx80BufferHead = 0;
	int Rx80BufferTail = 0;

	int rx_buffer_head3 = 0;
	int rx_buffer_tail3 = 0;
	int rx_buffer_head2 = 0;
	int rx_buffer_tail2 = 0;
	int rx_buffer_head1 = 0;
	int rx_buffer_tail1 = 0;
	int rx_buffer_head6 = 0;
	int rx_buffer_tail6 = 0;
#if USEUSART4==1
	int rx_buffer_head4 = 0;
	int rx_buffer_tail4 = 0;
#endif
#if USEUSART5==1
	int rx_buffer_head5 = 0;
	int rx_buffer_tail5 = 0;
#endif



static char StrRx80ForMove[300];//和上面的RecAPIFrameStr不同，RecAPIFrameStr处理所有接受到的xbee对单片机的回复，StrRx80只处理xbee从别的xbee接受到的数据
static uint16_t LenRx80ForMove=0;








//目前支持串口1 2 3
void beginSerial(long baud, uint8_t portNum)
{
	    GPIO_InitTypeDef GPIO_InitStructure;
		USART_InitTypeDef USART_InitStructure;
		NVIC_InitTypeDef NVIC_InitStructure;

	//开启串口3，接收为中断，发送到发送字符串时再开启中断，发送单个字节时用查询的方式
	if(portNum==3)
	{
	    //GPIO端口设置	

	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 
	  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOD, GPIO_PinSource8, GPIO_AF_USART3);  
	  	GPIO_PinAFConfig(GPIOD, GPIO_PinSource9, GPIO_AF_USART3);
	
	     //USART3_TX   PD.8
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOD, &GPIO_InitStructure);
	   
	    //USART3_RX	  PD.9
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOD, &GPIO_InitStructure);  
	
	   //Usart3 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART3
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;	 //没有校验位

		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(USART3, &USART_InitStructure);
	   
		USART_Cmd(USART3, ENABLE);                    //使能串口
	}
	else if(portNum==1)
	{
	    //GPIO端口设置	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
	  	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_USART1);  
	  	GPIO_PinAFConfig(GPIOB, GPIO_PinSource7, GPIO_AF_USART1);
	
	     //USART1_TX   PB.6
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOB, &GPIO_InitStructure);
	   
	    //USART1_RX	  PB.7
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOB, &GPIO_InitStructure);  
	
	   //Usart1 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART1
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;
		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(USART1, &USART_InitStructure);
	   
		USART_Cmd(USART1, ENABLE);                    //使能串口

	}
	else if(portNum==2)
	{	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE); 
	  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOA, GPIO_PinSource2, GPIO_AF_USART2);  
	  	GPIO_PinAFConfig(GPIOA, GPIO_PinSource3, GPIO_AF_USART2);
	
	     //USART2_TX   PA.2
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOA, &GPIO_InitStructure);
	   
	    //USART2_RX	  PA.3
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOA, &GPIO_InitStructure);  
	
	   //Usart2 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART2
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		   
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;//没有校验位

		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(USART2, &USART_InitStructure);
	   
		USART_Cmd(USART2, ENABLE);                    //使能串口	
	}
	else if(portNum==6)
	{
	    //GPIO端口设置	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE); 
	  	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART6, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOC, GPIO_PinSource6, GPIO_AF_USART6);  
	  	GPIO_PinAFConfig(GPIOC, GPIO_PinSource7, GPIO_AF_USART6);
	
	     //USART6_TX   PC.6
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOC, &GPIO_InitStructure);
	   
	    //USART6_RX	  PC.7
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOC, &GPIO_InitStructure);  
	
	   //Usart1 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = USART6_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART1
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;
		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(USART6, &USART_InitStructure);
	   
		USART_Cmd(USART6, ENABLE);                    //使能串口

	}
#if USEUSART4==1
	else if(portNum==4)
	{	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE); 
	  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_UART4, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOA, GPIO_PinSource0, GPIO_AF_UART4);  
	  	GPIO_PinAFConfig(GPIOA, GPIO_PinSource1, GPIO_AF_UART4);
	
	     //USART4_TX   PA.0
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOA, &GPIO_InitStructure);
	   
	    //USART4_RX	  PA.1
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOA, &GPIO_InitStructure);  
	
	   //Usart4 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = UART4_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART4
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;
		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(UART4, &USART_InitStructure);
	   
	    USART_ITConfig(UART4, USART_IT_RXNE, ENABLE);//开启中断 
	    UART4->CR1&=(~(0x01<<7));//禁止发送中断
		USART_Cmd(UART4, ENABLE);                    //使能串口	
	}
#endif

#if USEUSART5==1
	else if(portNum==5)
	{	
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);
	  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);		 
	  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_UART5, ENABLE);
	
	  	GPIO_PinAFConfig(GPIOC, GPIO_PinSource12, GPIO_AF_UART4);  
	  	GPIO_PinAFConfig(GPIOD, GPIO_PinSource2, GPIO_AF_UART4);
	
	     //USART5_TX   PC.12
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12;
	    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOC, &GPIO_InitStructure);
	   
	    //USART5_RX	  PD.2
	    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
	    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	    GPIO_Init(GPIOD, &GPIO_InitStructure);  
	
	   //Usart5 NVIC 配置
	    NVIC_InitStructure.NVIC_IRQChannel = UART5_IRQn;
		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
	
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
		NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART5
	  
	   //USART 初始化设置  
		USART_InitStructure.USART_BaudRate = baud;//一般设置为9600;
		USART_InitStructure.USART_WordLength = USART_WordLength_8b;
		USART_InitStructure.USART_StopBits = USART_StopBits_1;
		USART_InitStructure.USART_Parity = USART_Parity_No;
		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	    USART_Init(UART5, &USART_InitStructure);
	   
	    USART_ITConfig(UART5, USART_IT_RXNE, ENABLE);//开启中断 
	    UART5->CR1&=(~(0x01<<7));//禁止发送中断
		USART_Cmd(UART5, ENABLE);                    //使能串口	
	}
#endif
}

void chooseuartinterrupt(uint8_t portNum)
{
	if(portNum==3)
	{	   
	    USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);//开启中断 
	    USART3->CR1&=(~(0x01<<7));//禁止发送中断
	}
	else if(portNum==1)
	{	   
	    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);//开启中断 
	    USART1->CR1&=(~(0x01<<7));//禁止发送中断
	}
	else if(portNum==2)
	{		   
	    USART_ITConfig(USART2, USART_IT_RXNE, ENABLE);//开启中断 
	    USART2->CR1&=(~(0x01<<7));//禁止发送中断
	}
	else if(portNum==6)
	{	   
	    USART_ITConfig(USART6, USART_IT_RXNE, ENABLE);//开启中断 
	    USART6->CR1&=(~(0x01<<7));//禁止发送中断
	}

}

//开串口3发送的监控，即串口3发送的数据可以通过usb232可以看到，
void monitor_onuart3TX(void)
{
// PD15管PD8(uart3 TX)  PA8 管PD9(usart3 RX)  
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_15;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOD, &GPIO_InitStructured);

    GPIO_SetBits(GPIOD,GPIO_Pin_15);
}
//开串口3接收的监控，即串口3接收的数据可以通过usb232可以看到，
void monitor_onuart3RX(void)
{
// PD15管PD8(uart3 TX)  PA8 管PD9(usart3 RX)  
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_8;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructured);

    GPIO_SetBits(GPIOA,GPIO_Pin_8);
}

void monitor_offuart3TX(void)
{
// PD15管PD8(uart3 TX)  PA8 管PD9(usart3 RX)  
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_15;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOD, &GPIO_InitStructured);

    GPIO_ResetBits(GPIOD,GPIO_Pin_15);
}
void monitor_offuart3RX(void)
{
// PD15管PD8(uart3 TX)  PA8 管PD9(usart3 RX)  
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_8;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructured);

    GPIO_ResetBits(GPIOA,GPIO_Pin_8);
}

//要用USB232就得开这个
//同时开是不可以的，这个要看电路图了
void monitor_on(void)
{
// PD15管PD8(uart3 TX)  PA8 管PD9(usart3 RX)  
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_8;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOA, &GPIO_InitStructured);

    GPIO_SetBits(GPIOA,GPIO_Pin_8);


  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_15;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOD, &GPIO_InitStructured);

    GPIO_SetBits(GPIOD,GPIO_Pin_15);

}

//xbee电源开启
void Xbee_poweron(void)
{

// 让PTD11置高，也就是让Xbee 模块供电，否则Xbee不能工作//新板子是PD11
  	GPIO_InitTypeDef  GPIO_InitStructured;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_11;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOD, &GPIO_InitStructured);

   GPIOD->BSRRL=GPIO_Pin_11; //目前这块板子不知道为什么GPIO程序是低，现象却是高。这个问题暂时搁置

}

void Xbee_poweroff(void)
{
// 新板子是PD11低
  	GPIO_InitTypeDef  GPIO_InitStructured;
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_11;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOD, &GPIO_InitStructured);
	
	GPIO_ResetBits(GPIOD,GPIO_Pin_11);
}
//多路选择开启
void Mux_poweron(void)
{
	GPIO_InitTypeDef  GPIO_InitStructured;
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_8;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOE, &GPIO_InitStructured);

	GPIO_SetBits(GPIOE, GPIO_Pin_8);
}

void Mux_poweroff(void)
{
	GPIO_InitTypeDef  GPIO_InitStructured;
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_8;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOE, &GPIO_InitStructured);

	GPIO_ResetBits(GPIOE, GPIO_Pin_8);
}
//串口6多路选择初始化，注意这个不是串口6初始化
void muluart6init(void)
{
  	GPIO_InitTypeDef  GPIO_InitStructured;

	Mux_poweron();

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
  	GPIO_InitStructured.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1;
  	GPIO_InitStructured.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructured.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructured.GPIO_Speed = GPIO_Speed_100MHz;
  	GPIO_InitStructured.GPIO_PuPd = GPIO_PuPd_NOPULL;
  	GPIO_Init(GPIOB, &GPIO_InitStructured);	
}

void muluart6choose(unsigned char choose)
{
	if(choose==1)
	{
		GPIO_ResetBits(GPIOB, GPIO_Pin_1);
		GPIO_ResetBits(GPIOB, GPIO_Pin_0);	
	}
	else if(choose==2)
	{
		GPIO_ResetBits(GPIOB, GPIO_Pin_1);
		GPIO_SetBits(GPIOB, GPIO_Pin_0);	
	}
	else if(choose==3)
	{
		GPIO_SetBits(GPIOB, GPIO_Pin_1);
		GPIO_ResetBits(GPIOB, GPIO_Pin_0);	
	}
	else if(choose==4)
	{
		GPIO_SetBits(GPIOB, GPIO_Pin_1);
		GPIO_SetBits(GPIOB, GPIO_Pin_0);	
	}
}




void closexbeepwr(void)
{
		//这样Xbee模块就不会工作了
	   GPIOD->BSRRH=GPIO_Pin_15; //目前这块板子不知道为什么GPIO程序是低，现象却是高。这个问题暂时搁置
}

//开启串口3
//开Xbee供电
//开启TIM3,1ms中断一次，为了计数多少ms，这个是为了支持millis(),这个函数在WaspXBeeCore.cpp调用到
// connects the internal peripheral in the processor and configures it
void beginxbee(long baud, uint8_t portNum)
{	
	//开启串口3，接收为中断，发送到发送字符串时再开启中断，发送单个字节时用查询的方式
	beginSerial(baud, portNum);
	chooseuartinterrupt(portNum);//开启接收中断，发送中断暂时是禁止的
	Xbee_poweron();
	//开启1ms 中断一次
	Timer3_Init(10,10000); //第二个参数是time的频率，第一个参数是cnt
}


//这里关掉XBee的电源，这样XBee就不会工作了，
//这里并没有关掉定时器3，如果需要关掉，请客户提出
// disconnects the internal peripheral in the processor
void closeSerial(uint8_t portNum)
{
	if(portNum==3)
	{
	   closexbeepwr();
	   USART_Cmd(USART3, DISABLE);
	}
	else if(portNum==1)
	{
	   USART_Cmd(USART1, DISABLE);
	}
	else if(portNum==2)
	{
	   USART_Cmd(USART2, DISABLE);
	}
	else if(portNum==6)
	{
	   USART_Cmd(USART6, DISABLE);
	}
#if USEUSART4==1
	else if(portNum==4)
	{
	   USART_Cmd(UART4, DISABLE);
	}
#endif

#if USEUSART5==1
	else if(portNum==5)
	{
	   USART_Cmd(UART5, DISABLE);
	}
#endif

}
//用查询的方式发送串口3数据
void serialWrite(unsigned char c, uint8_t portNum)
{
	if(portNum==3)
	{
	while((USART3->SR&0X40)==0);//循环发送,直到发送完毕   
    USART3->DR = (uint8_t ) c;
	}
	else if(portNum==1)
	{
	while((USART1->SR&0X40)==0);//循环发送,直到发送完毕   
    USART1->DR = (uint8_t ) c;
	}
	else if(portNum==2)
	{
	while((USART2->SR&0X40)==0);//循环发送,直到发送完毕   
    USART2->DR = (uint8_t ) c;
	}
	else if(portNum==6)
	{
	while((USART6->SR&0X40)==0);//循环发送,直到发送完毕   
    USART6->DR = (uint8_t ) c;
	}

#if USEUSART4==1
	else if(portNum==4)
	{
	while((UART4->SR&0X40)==0);//循环发送,直到发送完毕   
    UART4->DR = (uint8_t ) c;
	}
#endif

#if USEUSART5==1
	else if(portNum==5)
	{
	while((UART5->SR&0X40)==0);//循环发送,直到发送完毕   
    UART5->DR = (uint8_t ) c;
	}
#endif


}
//和以前保持一样
int serialAvailable(uint8_t portNum)
{
	if (portNum == 3)
		return (RX_BUFFER_SIZE_3 + rx_buffer_head3 - rx_buffer_tail3) % RX_BUFFER_SIZE_3;
	else if (portNum == 1)
		return (RX_BUFFER_SIZE_1 + rx_buffer_head1 - rx_buffer_tail1) % RX_BUFFER_SIZE_1;
	else if (portNum == 2)
		return (RX_BUFFER_SIZE_2 + rx_buffer_head2 - rx_buffer_tail2) % RX_BUFFER_SIZE_2;
	else if (portNum == 6)
		return (RX_BUFFER_SIZE_6 + rx_buffer_head6 - rx_buffer_tail6) % RX_BUFFER_SIZE_6;
#if USEUSART4==1
	else if (portNum == 4)
		return (RX_BUFFER_SIZE_4 + rx_buffer_head4 - rx_buffer_tail4) % RX_BUFFER_SIZE_4;
#endif

#if USEUSART5==1
	else if (portNum == 5)
		return (RX_BUFFER_SIZE_5 + rx_buffer_head5 - rx_buffer_tail5) % RX_BUFFER_SIZE_5;
#endif
	else return 0;
	
}
//和以前保持一样
int serialRead(uint8_t portNum)
{
	if (portNum == 3) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head3 == rx_buffer_tail3) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer3[rx_buffer_tail3];
			rx_buffer_tail3 = (rx_buffer_tail3 + 1) % RX_BUFFER_SIZE_3;
			return c;
		}
	}
	else if (portNum == 1) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head1 == rx_buffer_tail1) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer1[rx_buffer_tail1];
			rx_buffer_tail1 = (rx_buffer_tail1 + 1) % RX_BUFFER_SIZE_1;
			return c;
		}
	}
	else if (portNum == 2) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head2 == rx_buffer_tail2) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer2[rx_buffer_tail2];
			rx_buffer_tail2 = (rx_buffer_tail2 + 1) % RX_BUFFER_SIZE_2;
			return c;
		}
	}
	else if (portNum == 6) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head6 == rx_buffer_tail6) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer6[rx_buffer_tail6];
			rx_buffer_tail6 = (rx_buffer_tail6 + 1) % RX_BUFFER_SIZE_6;
			return c;
		}
	}
#if USEUSART4==1
	else if (portNum == 4) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head4 == rx_buffer_tail4) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer4[rx_buffer_tail4];
			rx_buffer_tail4 = (rx_buffer_tail4 + 1) % RX_BUFFER_SIZE_4;
			return c;
		}
	}
#endif

#if USEUSART5==1
	else if (portNum == 5) 
	{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head5 == rx_buffer_tail5) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer5[rx_buffer_tail5];
			rx_buffer_tail5 = (rx_buffer_tail5 + 1) % RX_BUFFER_SIZE_5;
			return c;
		}
	}
#endif


	else return UARTRUNSUPPORT;
}


//
int serialPreRx80Available(void)
{
		return (RX_BUFFER_SIZE_3 + rx_buffer_head3 - rxprerx80_buffer_tail3) % RX_BUFFER_SIZE_3;	
}
//
int serialPreRx80Read(void)
{
		// if the head isn't ahead of the tail, we don't have any characters
		if (rx_buffer_head3 == rxprerx80_buffer_tail3) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = rx_buffer3[rxprerx80_buffer_tail3];
			rxprerx80_buffer_tail3 = (rxprerx80_buffer_tail3 + 1) % RX_BUFFER_SIZE_3;
			return c;
		}
}

uint8_t handle1msprerx80(void)
{
	int recnum=0;
	uint8_t ch;
//	uint8_t lenrecapi=0;
	uint8_t check;
	uint8_t icheck=3;
	long i=0;

loopstart:
			if( (recnum=serialPreRx80Available())==0 )
			{
				return 2;//还没有收到数据				
			}
			else
			{
				while(recnum--)
				{
					ch=serialPreRx80Read();
//#if ZPPDEBUGXBEEPRINT==1
					//printf("c%2x",ch);
					//printf("%c",ch);
//#endif
					if(LenRx80ForMove==0)
					{
						if(ch==0x7e)
						{
							StrRx80ForMove[LenRx80ForMove]=ch;
							LenRx80ForMove++;
							continue;
						}
						else
							continue;
					}
					else
					{ 
						if(LenRx80ForMove==2)
						{
//							lenrecapi= (uint16_t)((((uint16_t)StrRx80ForMove[1])<<8)|ch); //这里的ch不能写成 RecAPIFrameStr[2],因为[2]还没有赋值过来
//#if ZPPDEBUGXBEEPRINT==1
//							printf("lenapi=%2x ",lenrecapi);
//#endif						
							//我们假定包都是小于256个字节的，这样，有如下判断
							if(StrRx80ForMove[1]!=0)
							{
								LenRx80ForMove=0;
								return 4;
								//continue;
							}
						}
						else if((LenRx80ForMove==3)&&(ch!=0x80))
						{
							LenRx80ForMove=0;
							return 3;
							//continue;
						}
						else if((LenRx80ForMove>3)&&(LenRx80ForMove==(StrRx80ForMove[2]+3)))//这里不能使用lenrecapi这个值
						{
							
							check=0;
							for(icheck=3;icheck<LenRx80ForMove;icheck++)
								check = check+StrRx80ForMove[icheck];
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x ",check);
#endif
							check = 0xff -check;
#if ZPPDEBUGXBEEPRINT==1
							printf("chk:%2x <>",check);
							for(uint8_t icheck=0;icheck<LenRx80ForMove;icheck++)
								printf("%2x ",StrRx80ForMove[icheck]);
#endif
							if(check!=ch)
							{
								LenRx80ForMove=0;
								return 5;
								//continue;
							}
							else 
							{
#if ZPPDEBUGXBEEPRINT==1
								printf("find a APIframe ");
#endif
//								if(StrRx80[3]==0x80)

		 						StrRx80ForMove[LenRx80ForMove]=ch;
								LenRx80ForMove++;
								goto end;
//								else return 3;
								//return 0;
							}
						}

						StrRx80ForMove[LenRx80ForMove]=ch;
						LenRx80ForMove++;
					}
#if ZPPDEBUGXBEEPRINT==1
					printf("ofs=%2x ",LenRx80ForMove);	
#endif									
				}
			}
			
	return 1;

//loop4:
	

end:

////	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
//	*len = (*len)<LenRx80ForMove ? (*len):LenRx80ForMove ;
////	printf("*len=%d lenrx80=%d  ",*len,LenRx80);
	for(icheck=0; icheck<LenRx80ForMove; icheck++)
	{
		i = (Rx80BufferHead + 1) % RX80_BUFFER_SIZE;
		//if (i != Rx80BufferTail)
		{
			Rx80Buffer[Rx80BufferHead] = StrRx80ForMove[icheck];
			Rx80BufferHead = i;
			if(Rx80BufferHead==UART_LIMIT)
			{
				Rx80BufferHead=0;
				Rx80BufferTail=0;
			}
		}

	}


	LenRx80ForMove = 0;	
	if(recnum>0)
		goto loopstart;	
	return 0;





}
//
int serialRx80Available(void)
{
		return (RX80_BUFFER_SIZE + Rx80BufferHead - Rx80BufferTail) % RX80_BUFFER_SIZE;	
}
//
int serialRx80Read(void)
{
		// if the head isn't ahead of the tail, we don't have any characters
		if (Rx80BufferHead == Rx80BufferTail) 
		{
			return UARTRECEMPTY;
		} 
		else 
		{
			unsigned char c = Rx80Buffer[Rx80BufferTail];
			Rx80BufferTail = (Rx80BufferTail + 1) % RX80_BUFFER_SIZE;
			return c;
		}
}





//当要读的数据比最大接收区大则返回-1，
//把当期接收区数据都读出来（最大到len个），赋给str这个数组里面,接收区剩下的数据等待下次再读
//如果接收区没有数据则返回0，有数据返回赋给str数据个数
//如果接收区数据少，str的len大，则有多少读多少
int serialReadstr(uint8_t *str, uint8_t len,uint8_t portNum)
{
	int i;
	int recvzhi;
	if(len>RX_BUFFER_SIZE)return UARTRECERRELSE;

	for(i=0;i<len;i++)
	{
		recvzhi=serialRead(portNum);
		if(recvzhi>=0)//接收到数据，如果没有接收到数据则这个数据肯定是负数（因为错误在这里都用负数来定义的）
		{
			str[i]=recvzhi;
		}
		else return i;
	}
	return i;
}


//和以前保持一样
void serialFlush(uint8_t portNum)
{
	// don't reverse this or there may be problems if the RX interrupt
	// occurs after reading the value of rx_buffer_head but before writing
	// the value to rx_buffer_tail; the previous value of rx_buffer_head
	// may be written to rx_buffer_tail, making it appear as if the buffer
	// were full, not empty.
	if (portNum == 3){
		rx_buffer_tail3=0;
		rx_buffer_head3 = rx_buffer_tail3;
	}
	else if (portNum == 1)
	{
		rx_buffer_tail1=0;
		rx_buffer_head1 = rx_buffer_tail1;
	}
	else if (portNum == 2)
	{
		rx_buffer_tail2=0;
		rx_buffer_head2 = rx_buffer_tail2;
	}
	else if (portNum == 6)
	{
		rx_buffer_tail6=0;
		rx_buffer_head6 = rx_buffer_tail6;
	}
#if USEUSART4==1
	else if (portNum == 4)
	{
		rx_buffer_tail4=0;
		rx_buffer_head4 = rx_buffer_tail4;
	}
#endif

#if USEUSART5==1
	else if (portNum == 5)
	{
		rx_buffer_tail5=0;
		rx_buffer_head5 = rx_buffer_tail5;
	}
#endif

}



//这里和以前一样，调用的函数串口3是查询方式发送一个字节
void printByte(unsigned char c, uint8_t portNum)
{
	serialWrite(c, portNum);
}

//void printNewline(uint8_t portNum)
//{
//	printByte('\n', portNum);
//}

//uint8_t USART_RX_BUF3[64];     //接收缓冲,最大64个字节.
//uint8_t USART_RX_STA3=0;       //接收状态标记

//uint8_t USART_RX_BUF1[64];     //接收缓冲,最大64个字节.
//uint8_t USART_RX_STA1=0;       //接收状态标记

//uint8_t USART_RX_BUF2[64];     //接收缓冲,最大64个字节.
//uint8_t USART_RX_STA2=0;       //接收状态标记

uint32_t USARTTCnt3=0;
char *pUSART_TX_BUF3; 

uint32_t USARTTCnt1=0;
char *pUSART_TX_BUF1; 

uint32_t USARTTCnt2=0;
char *pUSART_TX_BUF2; 

uint32_t USARTTCnt6=0;
char *pUSART_TX_BUF6;
 
#if USEUSART4==1
uint32_t USARTTCnt4=0;
char *pUSART_TX_BUF4;
#endif

#if USEUSART5==1
uint32_t USARTTCnt5=0;
char *pUSART_TX_BUF5;
#endif

////初始化串口3
////开启接收中断，但没有开启发送中断，
////当要发一个字节时采用的是查询的方式，见 serialWrite(unsigned char c, uint8_t portNum)
////当要发字符串时采用的是中断方式，见 printString(const char *s, uint8_t portNum) 和	USART3_IRQHandler(void)
//void COM3Init(uint32_t BaudRate)
//{
//      //GPIO端口设置
//    GPIO_InitTypeDef GPIO_InitStructure;
//	USART_InitTypeDef USART_InitStructure;
//	NVIC_InitTypeDef NVIC_InitStructure;
//	 
////	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1|RCC_AHB1Periph_GPIOA , ENABLE);
//  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 
//  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);
//
//  	GPIO_PinAFConfig(GPIOD, GPIO_PinSource8, GPIO_AF_USART3);  
//  	GPIO_PinAFConfig(GPIOD, GPIO_PinSource9, GPIO_AF_USART3);
//
//     //USART1_TX   PD.8
//    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
//    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
//    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
//    GPIO_Init(GPIOD, &GPIO_InitStructure);
//   
//    //USART1_RX	  PD.9
//    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
//    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
//    GPIO_Init(GPIOD, &GPIO_InitStructure);  
//
//   //Usart1 NVIC 配置
//
//    NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
//	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=3 ;
//	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;		//
//
//	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
//	NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART1
//  
//   //USART 初始化设置  
//	USART_InitStructure.USART_BaudRate = BaudRate;//一般设置为9600;
//	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
//	USART_InitStructure.USART_StopBits = USART_StopBits_1;
//	USART_InitStructure.USART_Parity = USART_Parity_No;
//	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
//	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
//
//    USART_Init(USART3, &USART_InitStructure);
//   
//    USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);//开启中断
//    //USART_ITConfigzpp(USART3, USART_IT_TXE, ENABLE);//开启中断
//    //USART_ITConfigzpp(USART3, USART_IT_RXNE|USART_IT_TC , ENABLE);//开启中断  
//    //USART_ITConfigzpp(USART3, USART_IT_RXNE|USART_IT_TXE , ENABLE);//开启中断  
//    USART3->CR1&=(~(0x01<<7));//禁止发送中断
//	USART_Cmd(USART3, ENABLE);                    //使能串口
//}
//
//
//




//串口3中断服务程序
void USART3_IRQHandler(void)                	
{
	uint8_t Res;	
	int  i;
	if(USART_GetITStatus(USART3, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
	{
		Res =USART_ReceiveData(USART3);//(USART3->DR);	//读取接收到的数据
		
		i = (rx_buffer_head3 + 1) % RX_BUFFER_SIZE_3;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		//if (i != rx_buffer_tail3)
		 {
			rx_buffer3[rx_buffer_head3] = Res;
			rx_buffer_head3 = i;
			if(rx_buffer_head3==UART_LIMIT)
			{
				rx_buffer_head3=0;
				rx_buffer_tail3=0;
			}
		}
		//if(Res==0xe5)serialWrite(0xd5, 3);

		//serialWrite(Res, 2);
		if((Res>='A')&&(Res<='Z'))
		//if(Res=='i')
		{
		  //USART_Cmd(USART3, DISABLE);
			//;
			//serialWrite(Res, 3);
			if(Res=='I')
			{

			   //NVIC_SystemReset();
			}
			return;


		}


  		 
     } 
	 else if(USART_GetITStatus(USART3, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt3>0)
		{
			USART_SendData(USART3, (uint8_t ) pUSART_TX_BUF3[0]);
			   pUSART_TX_BUF3++;
			USARTTCnt3--;		
		}
		else
		{
			USART3->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }
} 







//目前用查询的方式发送字符串
//这里开启中断，当发送结束（也就是说USARTTCnt3=0）时，则关掉中断
void printString(const char *s, uint8_t portNum)
{
uint32_t len,ilen;

	//uartsendstr((char *)s, strlen(s),portNum);
	//USARTTCnt3=strlen(s);
	//pUSART_TX_BUF3=(char *)s;
	//USART3->CR1|=((0x01<<7));//开启发送中断

//假如客户不喜欢用中断方式发送，用查询方式也可以，下面是查询方式 ,当然把上面的代码屏蔽掉
len=strlen(s);
for(ilen=0;ilen<len;ilen++)	
	serialWrite(s[ilen], portNum);


}
void serialWritestr(char *s, uint8_t len,uint8_t portNum)
{
	uartsendstr(s, len,portNum);
}

//中断方式发送数据
void uartsendstr(char *s, uint8_t len,uint8_t portNum)
{
	if(portNum==3)
	{
		USARTTCnt3=len;
		pUSART_TX_BUF3=(char *)s;
		USART3->CR1|=((0x01<<7));//开启发送中//
	}
	else if(portNum==1)
	{
		USARTTCnt1=len;
		pUSART_TX_BUF1=(char *)s;
		USART1->CR1|=((0x01<<7));//开启发送中
	}
	else if(portNum==2)
	{
		USARTTCnt2=len;
		pUSART_TX_BUF2=(char *)s;
		USART2->CR1|=((0x01<<7));//开启发送中//
	}
	else if(portNum==6)
	{
		USARTTCnt6=len;
		pUSART_TX_BUF6=(char *)s;
		USART6->CR1|=((0x01<<7));//开启发送中//
	}
#if USEUSART4==1
	else if(portNum==4)
	{
		USARTTCnt4=len;
		pUSART_TX_BUF4=(char *)s;
		UART4->CR1|=((0x01<<7));//开启发送中//

	}
#endif

#if USEUSART5==1
	else if(portNum==5)
	{
		USARTTCnt5=len;
		pUSART_TX_BUF5=(char *)s;
		UART5->CR1|=((0x01<<7));//开启发送中//

	}
#endif


}





void USART1_IRQHandler(void)                	//串口1中断服务程序
{
	uint8_t Res;
	int  i;
	if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
	{
		Res =USART_ReceiveData(USART1);//(USART1->DR);	//读取接收到的数据

		i = (rx_buffer_head1 + 1) % RX_BUFFER_SIZE_1;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		if (i != rx_buffer_tail1)
		 {
			rx_buffer1[rx_buffer_head1] = Res;
			rx_buffer_head1 = i;
			if(rx_buffer_head1==UART_LIMIT)
			{
				rx_buffer_head1=0;
				rx_buffer_tail1=0;
			}
		}
		//serialWrite(Res, 1);   		 
     }
	 else if(USART_GetITStatus(USART1, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt1>0)
		{
			USART_SendData(USART1, (uint8_t ) pUSART_TX_BUF1[0]);
			   pUSART_TX_BUF1++;
			USARTTCnt1--;		
		}
		else
		{
			USART1->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }	  
} 


void USART2_IRQHandler(void)                	//串口1中断服务程序
{
	uint8_t Res;
	int  i;
	if(USART_GetITStatus(USART2, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
		{
		Res =USART_ReceiveData(USART2);//(USART2->DR);	//读取接收到的数据
		
		i = (rx_buffer_head2 + 1) % RX_BUFFER_SIZE_2;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		if (i != rx_buffer_tail2)
		 {
			rx_buffer2[rx_buffer_head2] = Res;
			rx_buffer_head2 = i;
			if(rx_buffer_head2==UART_LIMIT)
			{
				rx_buffer_head2=0;
				rx_buffer_tail2=0;
			}
		}

  		 
     }
	 else if(USART_GetITStatus(USART2, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt2>0)
		{
			USART_SendData(USART2, (uint8_t ) pUSART_TX_BUF2[0]);
			   pUSART_TX_BUF2++;
			USARTTCnt2--;		
		}
		else
		{
			USART2->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }	  
} 
void USART6_IRQHandler(void)                	//串口6中断服务程序
{
	uint8_t Res;
	int  i;
	if(USART_GetITStatus(USART6, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
		{
		Res =USART_ReceiveData(USART6);//(USART6->DR);	//读取接收到的数据

		i = (rx_buffer_head6 + 1) % RX_BUFFER_SIZE_6;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		if (i != rx_buffer_tail6)
		 {
			rx_buffer6[rx_buffer_head6] = Res;
			rx_buffer_head6 = i;
			if(rx_buffer_head6==UART_LIMIT)
			{
				rx_buffer_head6=0;
				rx_buffer_tail6=0;
			}
		}

		 
     }
	 else if(USART_GetITStatus(USART6, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt6>0)
		{
			USART_SendData(USART6, (uint8_t ) pUSART_TX_BUF6[0]);
			   pUSART_TX_BUF6++;
			USARTTCnt6--;		
		}
		else
		{
			USART6->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }	  
} 

#if USEUSART4==1
void UART4_IRQHandler(void)                	//串口4中断服务程序
{
	uint8_t Res;
	int  i;
	if(USART_GetITStatus(UART4, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
		{
		Res =USART_ReceiveData(UART4);//(UART4->DR);	//读取接收到的数据
		
		i = (rx_buffer_head4 + 1) % RX_BUFFER_SIZE_4;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		if (i != rx_buffer_tail4)
		 {
			rx_buffer4[rx_buffer_head4] = Res;
			rx_buffer_head4 = i;
			if(rx_buffer_head4==UART_LIMIT)
			{
				rx_buffer_head4=0;
				rx_buffer_tail4=0;
			}
		}
		 
     }
	 else if(USART_GetITStatus(UART4, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt4>0)
		{
			USART_SendData(UART4, (uint8_t ) pUSART_TX_BUF4[0]);
			   pUSART_TX_BUF4++;
			USARTTCnt4--;		
		}
		else
		{
			UART4->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }	  
} 
#endif


#if USEUSART5==1
void UART5_IRQHandler(void)                	//串口4中断服务程序
{
	uint8_t Res;
	int  i;
	if(USART_GetITStatus(UART5, USART_IT_RXNE) != RESET)  //接收中断(接收到的数据必须是0x0d 0x0a结尾)
		{
		Res =USART_ReceiveData(UART5);//(UART4->DR);	//读取接收到的数据
		
		i = (rx_buffer_head5 + 1) % RX_BUFFER_SIZE_5;
		// if we should be storing the received character into the location
		// just before the tail (meaning that the head would advance to the
		// current location of the tail), we're about to overflow the buffer
		// and so we don't write the character or advance the head.
		if (i != rx_buffer_tail5)
		 {
			rx_buffer5[rx_buffer_head5] = Res;
			rx_buffer_head5 = i;
			if(rx_buffer_head5==UART_LIMIT)
			{
				rx_buffer_head5=0;
				rx_buffer_tail5=0;
			}
		}
		 
     }
	 else if(USART_GetITStatus(UART5, USART_IT_TXE ) != RESET)
	 {
	 	if(USARTTCnt5>0)
		{
			USART_SendData(UART5, (uint8_t ) pUSART_TX_BUF5[0]);
			   pUSART_TX_BUF5++;
			USARTTCnt5--;		
		}
		else
		{
			UART5->CR1&=(~(0x01<<7));//禁止发送中断
		}

	 }	  
} 
#endif



//这里和以前的有区别：1.以前是查询方式一个个发送字节，这里改成中断方式发送字符串，2.为了用中断方式，对字符串进行了前期处理
//这里调用的是用查询方式发送字符串的
//目前base>16认为是非法，这个客户可以修改的
//其实作者觉得这个函数根本就没有存在性，这个完全可以用sprintf 和 uartsendstr 代替
void printIntegerInBase(unsigned long n, unsigned long base, uint8_t portNum)
{ 
	unsigned char buf[8 * sizeof(long)]; // Assumes 8-bit chars. 
	unsigned long i = 0;
	unsigned long len;
	unsigned char temp;

	if (n == 0) {
		buf[0]='0';
		buf[1]=0x00;
		printString((const char *)buf, portNum);
		return;
	} 

	if(base>16)
	{
		sprintf((char *) buf,"error: base>16! ");
		printString((const char *)buf, portNum);
		return;			
	}
	//得到[0]=n%base [1]=(n/base)%base ...
	while (n > 0) {
		buf[i++] = n % base;
		n /= base;
	}
	buf[i]=0x00;
	len=i;
	//倒序一下 使得[0]为最高位
	for(i=0;i<(len+1)/2;i++)
	{
		temp=buf[i];
		buf[i]=buf[len-1-i];
		buf[len-1-i]=temp;
	}
	//把数字转换成ASICC码，目前只支持16进制及以下，所以当数字大于9则显示A~F，如果客户要求支持超出16进制，需要客户改写这一段
	for(i=0;i<len;i++)
	{
		if(buf[i]<10)buf[i]+='0';
		else buf[i]+='A'-10;
	}
	//中断方式把字符串发出去
	printString((const char *)buf, portNum);

}


void beginMb7076(void)
{
  	GPIO_InitTypeDef  GPIO_InitStructure;//GPIO初始化结构体
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);//外设时钟使能
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);
	GPIO_SetBits(GPIOE, GPIO_Pin_7);
	beginSerial(9600, 2);
	chooseuartinterrupt(2);
	
}
//return  2 // //没有任何新数据	 //rx_buffer_head2==rx_buffer_tail2
//        0	// 有了一个完整的新数据	, 有两种情况，一种是正好收到完整的一个数据包，
//                                                     另一种,当前的收到好几个包，前面的数据包都是完整的，最后一个不完整，此时反馈前面 数据包内容
//        1 // 有数据，但是目前数据不足(只有半包数据,或者最近的包都不是完整包),需要等待一会儿,此时数据反馈值为上次的数据值
// 如果收到的数据正确，则距离值有value带出来
int readMb7076( int* value)
{
	int len;
	int strbegain;
	int cengci=0;
	uint8_t zhi;
	uint8_t zhia[6];
	static int LastMb7076Value=0;
	int counterr=0;
	int i;
	int head2 ;
	i= rx_buffer_tail2;	head2= rx_buffer_head2;
//	if(i!=head2)
//	{
		
//		printf(" i=%d head2=%d ",i,head2);
//		while(1)
//		{
//			i++;
//			i %= RX_BUFFER_SIZE_2;	
//			printf(" %2x",rx_buffer2[i]);
//
//			if(i== head2)
//			{
//				break;
//			}
//			
//		}
//
//	}



	if(head2 !=rx_buffer_tail2)
	{
		len = (head2 +RX_BUFFER_SIZE_2-rx_buffer_tail2)%RX_BUFFER_SIZE_2;
//		printf("len=%d",len);
		if(len<6) //// 有数据，但是目前数据不足
		{
			*value=LastMb7076Value ;
			return 1; 
		}
//		zhi= rx_buffer2[rx_buffer_head2];
//		if(zhi=='\n'){* value = 11; }
//		else if( (zhi>='0')&&(zhi<='9')){* value = zhi-'0'; }
//		else if(zhi=='R'){* value = 12; }
//		else  {* value = 13; }
//		rx_buffer_tail2 = rx_buffer_head2 ;
//
//		return 0;
		strbegain = (head2 +RX_BUFFER_SIZE_2-5)%RX_BUFFER_SIZE_2;
//		printf(" start:%d ",strbegain);
		while(1)
		{
			for(cengci=0;cengci<6;cengci++)
			{
				zhi= rx_buffer2[(strbegain+cengci)%RX_BUFFER_SIZE_2];
//				printf("zhi=%d ",zhi);
//				printf(" cen=%d ",cengci);
				if(cengci==0)
				{
//					printf(" 000 ");
					if(zhi!='R')break;
				}
				else if((cengci>=1)&&(cengci<=4))
				{
//					printf(" 111444 ");
					if((zhi<'0')||(zhi>'9'))
					{
						printf(" false num cebgci=%d ",cengci);
						break;
					}
					zhia[cengci-1]=zhi-'0';
//					printf(" ce-1 =%d []=%d ",cengci-1,zhia[cengci-1]);
				}
				else if(cengci==5)
				{
//					printf(" 555 ");
					if(zhi!=0x0d)
					{ 
						//printf("no end");
						break; 
					}
				}
				else
				{
					printf("   WRONG    ");
				}
			}
			if(cengci==6)
			{
//				printf(" zhia %d %d %d %d ",zhia[0],zhia[1],zhia[2],zhia[3]);
				*value = ((int)zhia[0])*1000+((int)zhia[1])*100+ ((int)zhia[2])*10+ ((int)zhia[3]);
//				printf(" *value %d ,",*value );
				LastMb7076Value=*value ;
				//rx_buffer_tail2 = (head2 +RX_BUFFER_SIZE_2-5)%RX_BUFFER_SIZE_2;
				rx_buffer_tail2 = head2;
				return 0;
			}
			else
			{
				;
//				printf(" cengci=%d ",cengci);
			}
			
			

			strbegain = (strbegain +RX_BUFFER_SIZE_2-1)%RX_BUFFER_SIZE_2;
//			printf(" start:%d ",strbegain);

			if(strbegain == rx_buffer_tail2)
			{
				rx_buffer_tail2 = (head2 +RX_BUFFER_SIZE_2-6)%RX_BUFFER_SIZE_2;
				*value=LastMb7076Value ;

				return 1;
			}
			delay_ms(1);
			counterr++;
			if(counterr>1000)return 3;
		}

		
	}
	else
	{ 
		
		//*value = 0;
		*value=LastMb7076Value ;
		return 2;//没有任何新数据
	}
}


