
#include "dingke_pwr.h"
void initPwr(uint32_t choose)
{
  	GPIO_InitTypeDef  GPIO_InitStructure;//GPIO初始化结构体
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);//外设时钟使能
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);//外设时钟使能
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);//外设时钟使能
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);

	//3.3VsensorPC5 	
	if(choose&0x0001)
	{
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOC, &GPIO_InitStructure);
	}	
	
	//5Vsensor1  PE7
	if(choose&0x0002)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOE, &GPIO_InitStructure);	
	}

	//5Vsensor2 PA4	
	if(choose&0x0004)
	{
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOA, &GPIO_InitStructure);	
	}

	//5Vsensor3 PD6	
	if(choose&0x0008)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOD, &GPIO_InitStructure);
	}

	//Muxuart6 PE8	
	if(choose&0x0010)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOE, &GPIO_InitStructure);
	}


	//SD PD7	
	if(choose&0x0020)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOD, &GPIO_InitStructure);
	}

	//XBEE PD11	
	if(choose&0x0040)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOD, &GPIO_InitStructure);
	}

	//RTC PE10	
	if(choose&0x0080)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOE, &GPIO_InitStructure);
	}

	//BAT PA7	
	if(choose&0x0100)
	{	
		GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;	
		GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
		GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
		GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
		GPIO_Init(GPIOA, &GPIO_InitStructure);
	}		
}
void 	initAllPwr(void)
{
	initPwr(0xffff);
}

void setPwr(uint32_t choose, uint16_t mode)
{
	//3.3VsensorPC5 	
	if(choose&0x0001) //#define	PWR_SENS_3V3	0x0001
	{
		if(mode==PWR_ON) GPIO_SetBits(GPIOC, GPIO_Pin_5);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOC, GPIO_Pin_5);
	}	
	
	//5Vsensor1  PE7
	if(choose&0x0002) //#define	PWR_SENS1_5V	0x0002
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOE, GPIO_Pin_7);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOE, GPIO_Pin_7);	
	}

	//5Vsensor2 PA4	
	if(choose&0x0004)//#define	PWR_SENS2_5V	0x0004
	{
		if(mode==PWR_ON)  GPIO_SetBits(GPIOA, GPIO_Pin_4);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOA, GPIO_Pin_4);	
	}

	//5Vsensor3 PD6	
	if(choose&0x0008)//#define	PWR_SENS3_5V	0x0008
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOD, GPIO_Pin_6);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOD, GPIO_Pin_6);
	}

	//Muxuart6 PE8	
	if(choose&0x0010)//#define	PWR_MUX_UART6	0x0010
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOE, GPIO_Pin_8);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOE, GPIO_Pin_8);
	}


	//SD PD7	
	if(choose&0x0020)//#define	PWR_SD			0x0020
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOD, GPIO_Pin_7);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOD, GPIO_Pin_7);
	}

	//XBEE PD11	
	if(choose&0x0040)//#define	PWR_XBEE		0x0040
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOD, GPIO_Pin_11);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOD, GPIO_Pin_11);
	}

	//RTC PE10	
	if(choose&0x0080)//#define	PWR_RTC			0x0080
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOE, GPIO_Pin_10);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOE, GPIO_Pin_10);
	}

	//BAT PA7	
	if(choose&0x0100)//#define	PWR_BAT			0x0100
	{	
		if(mode==PWR_ON) GPIO_SetBits(GPIOA, GPIO_Pin_7);
		else if(mode==PWR_OFF) GPIO_ResetBits(GPIOA, GPIO_Pin_7);
	}

}


void	offAllPwr(void) //电源关 
{
	setPwr(0xffff, PWR_ON);		
}


void	onAllPwr(void)
{		
	setPwr(0xffff, PWR_OFF);
}

