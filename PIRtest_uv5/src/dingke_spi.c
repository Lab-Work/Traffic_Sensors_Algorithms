#include <stm32f4xx.h>
#include "dingke_spi.h"

//#include "delay.h"
static SPI_InitTypeDef  SPI_InitStructure;
static GPIO_InitTypeDef GPIO_InitStructure;
void SPI_poweron(void)//PE15	//这个口是和IIC公用的
{
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE); 
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);
	
	GPIO_SetBits(GPIOE,GPIO_Pin_10);
}

//初始化SPI2接口
void SPI2_Init(void)
{
	SPI_poweron();

  	RCC_APB1PeriphClockCmd(RCC_APB1Periph_SPI2, ENABLE);
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB ,ENABLE);
  	GPIO_PinAFConfig(GPIOB, GPIO_PinSource13, GPIO_AF_SPI2);
  	GPIO_PinAFConfig(GPIOB, GPIO_PinSource14, GPIO_AF_SPI2);
  	GPIO_PinAFConfig(GPIOB, GPIO_PinSource15, GPIO_AF_SPI2);
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_DOWN;
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15;
  	GPIO_Init(GPIOB, &GPIO_InitStructure);
}


//将SPI2初始化为串行FLASH用途
void FLASH_SPI_Init(void)
{
	SPI_I2S_DeInit(SPI2);
 	SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;//全双工
  	SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;//8位数据模式
  	SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;//空闲模式下SCK为0
  	SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;//数据采样从第1个时间边沿开始
  	SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;//NSS软件管理
  	//SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_2;//波特率
	SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_8;
  	SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;//大端模式
  	SPI_InitStructure.SPI_CRCPolynomial = 7;//CRC多项式
  	SPI_InitStructure.SPI_Mode = SPI_Mode_Master;//主机模式
  	SPI_Init(SPI2, &SPI_InitStructure);
  	SPI_Cmd(SPI2, ENABLE);
}


//初始化所有SPI器件片选引脚并置高不选中
void CSPin_init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;

  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12;//PB12
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_Init(GPIOB, &GPIO_InitStructure);
  	GPIO_SetBits(GPIOB, GPIO_Pin_12);//不选中	//片选置高


  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_14;//PB12
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_Init(GPIOE, &GPIO_InitStructure);
  	GPIO_SetBits(GPIOE, GPIO_Pin_14);//不选中 //写保护置高为了不保护
	//GPIO_ResetBits (GPIOE, GPIO_Pin_3);



  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;//PB12
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  	GPIO_Init(GPIOE, &GPIO_InitStructure);
  	GPIO_SetBits(GPIOE, GPIO_Pin_15);//不选中 //	HOLD置高，就是为了不HOLD
	//GPIO_ResetBits (GPIOE, GPIO_Pin_3);



}

u8 SPI2_RWByte(u8 byte)
{
 	while((SPI2->SR&SPI_I2S_FLAG_TXE)==RESET);
 	SPI2->DR = byte;
 	while((SPI2->SR&SPI_I2S_FLAG_RXNE)==RESET);
 	return(SPI2->DR);
}


