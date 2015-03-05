#include <stm32f4xx.h>
#include <stdio.h>
#include "stm32f4xx_conf.h"
#include "dingke_delay.h"	
#include "usbd_hid_core.h"
#include "usbd_usr.h"
#include "usbd_desc.h"

#include "dingke_timer.h"
#include "dingke_uart.h"
#include "dingke_sysclkchange.h"//为了测试主频变化的
/****************************************************************************

****************************************************************************/

/****************************************************************************
程序说明
	虚拟鼠标
	连接USB OTG接口至电脑,通过触摸屏操作模拟鼠标功能
****************************************************************************/

USB_OTG_CORE_HANDLE  USB_OTG_dev;
void Send_Report(u8 byte0,u8 byte1,u8 byte2,u8 byte3);
	u8 zppshow[30];

void testusbhid(void)
{			
	//硬件初始化	
	USBD_Init(&USB_OTG_dev,USB_OTG_FS_CORE_ID,&USR_desc,&USBD_HID_cb,&USR_cb);//USB从机初始化

	while(1)
	{
		//光标上移
		Send_Report(0,0,250,0);//发送报文	
		delay_ms(1000);

		//光标左移
		Send_Report(0,250,0,0);//发送报文	
		delay_ms(1000);

		//光标下移
		Send_Report(0,0,6,0);//发送报文	
		delay_ms(1000);

		//光标右移
		Send_Report(0,6,0,0);//发送报文	
		delay_ms(1000);	
	}
}

void testclkusbhid(void)
{

	unsigned long testsys=84000000;
	unsigned char flagfangxiang=0;

//	unsigned long zppSysClkWasp;
//    unsigned long zppDelayMsWasp;
//	unsigned long zppDelayUsWasp;
//	unsigned long zppSysPllN,zppSysPllM,zppSysPllP,zppSysPllQ;
//	unsigned long zppSysFlashWait;
	
	unsigned int hidci=0;
	unsigned int cishu=0;
	RCC_ClocksTypeDef RCC_ClocksStatus;
  //uint32_t tmp = 0, presc = 0, pllvco = 0, pllp = 2, pllsource = 0, pllm = 2,plln=0;
  uint32_t pllp = 2, pllm = 2,plln=0, pllvco = 0,pllsource = 0,pllq=0;

//   	GPIO_InitTypeDef  GPIO_InitStructure;//GPIO初始化结构体



	for(cishu=0;cishu<1000;cishu++)
	{
		SysClkWasp=testsys;
		SysPreparePara();

		zppSystemInit();SysChsngeDelay(0x123456);			
		Timer3_Init(10,10000);
		beginSerial(9600, 3);
		//硬件初始化	
		USBD_Init(&USB_OTG_dev,USB_OTG_FS_CORE_ID,&USR_desc,&USBD_HID_cb,&USR_cb);//USB从机初始化






		printf(" cishu=%d ",cishu);
		RCC_GetClocksFreq(&RCC_ClocksStatus);
		printf("sys=%d,hclk=%d,pclk1=%d,pclk2=%d"\
		,RCC_ClocksStatus.SYSCLK_Frequency,RCC_ClocksStatus.HCLK_Frequency\
		,RCC_ClocksStatus.PCLK1_Frequency,RCC_ClocksStatus.PCLK2_Frequency);
		SysChsngeDelay(0x1234567);
		printf("\npreclk=%d,pllm=%d,n=%d,p=%d,q=%d,wait=%d delayms=%d delayus=%d "\
		,SysClkWasp,SysPllM,SysPllN,SysPllP,SysPllQ,SysFlashWait,DelayMsWasp,DelayUsWasp);
		SysChsngeDelay(0x1234567);



		pllm = RCC->PLLCFGR & RCC_PLLCFGR_PLLM;
		plln=((RCC->PLLCFGR & RCC_PLLCFGR_PLLN) >> 6);
		pllp = (((RCC->PLLCFGR & RCC_PLLCFGR_PLLP) >>16) + 1 ) *2;
		pllq = ((RCC->PLLCFGR & RCC_PLLCFGR_PLLQ) >> 24);
      pllsource = (RCC->PLLCFGR & RCC_PLLCFGR_PLLSRC) >> 22;
      if (pllsource != 0)
      {
        /* HSE used as PLL clock source */
        pllvco = (HSE_VALUE / pllm) * ((RCC->PLLCFGR & RCC_PLLCFGR_PLLN) >> 6);
      }
      else
      {
        /* HSI used as PLL clock source */
        pllvco = (HSI_VALUE / pllm) * ((RCC->PLLCFGR & RCC_PLLCFGR_PLLN) >> 6);      
      }

		printf("m=%d n=%d p=%d ps=%d,pv=%d,pq=%d,pv/pq=%d ",pllm,plln,pllp,pllsource,pllvco,pllq,pllvco/pllq);
			
		delay_ms(5000);
		for(hidci=0;hidci<5;hidci++)
		{
			//光标上移
			Send_Report(0,0,250,0);//发送报文
			printf("shang");	
			delay_ms(300);
	
			//光标左移
			Send_Report(0,250,0,0);//发送报文	
			delay_ms(300);
	
			//光标下移
			Send_Report(0,0,6,0);//发送报文	
			delay_ms(300);
	
			//光标右移
			Send_Report(0,6,0,0);//发送报文	
			delay_ms(1000);	

//		USB_OTG_dev.dev.usr_cb->DeviceDisconnected();
//  		USB_OTG_dev.dev.class_cb->DeInit(&USB_OTG_dev, 0);
//			delay_ms(2000);
		}
		*((__IO uint32_t 	*)0x50000804 ) |=0x02; //软件上使得设备与电脑断开
//		USB_OTG_dev.dev.usr_cb->DeviceDisconnected();
//  		USB_OTG_dev.dev.class_cb->DeInit(&USB_OTG_dev, 0);

//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);//外设时钟使能
//  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9|GPIO_Pin_11|GPIO_Pin_12|GPIO_Pin_10;	
//  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
//  	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
//  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
//  	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
//  	GPIO_Init(GPIOA, &GPIO_InitStructure);

//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);//外设时钟使能
//  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9|GPIO_Pin_11|GPIO_Pin_12|GPIO_Pin_10;	
//  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
//  	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
//  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
//  	GPIO_Init(GPIOA, &GPIO_InitStructure);
//
//	GPIO_SetBits(GPIOA, GPIO_Pin_9);
//	GPIO_SetBits(GPIOA, GPIO_Pin_10);
//	GPIO_SetBits(GPIOA, GPIO_Pin_11);
//	GPIO_SetBits(GPIOA, GPIO_Pin_12);
//
//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);//外设时钟使能
//  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;	
//  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
//  	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
//  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
//  	GPIO_Init(GPIOC, &GPIO_InitStructure);
//	GPIO_SetBits(GPIOC, GPIO_Pin_0);





				delay_ms(5000);
				printf("OD");


//		for(hidci=0;hidci<100;hidci++)
//		  delay_ms(5000);
		if(flagfangxiang==0)
		{
			testsys=testsys-6000000;
			if(testsys<60000000)
			{
				testsys=168000000;
				flagfangxiang=1;
			}
		}
		else if(flagfangxiang==1)
		{
			testsys=testsys+6000000;
			if(testsys>18000000)
			{
				testsys=168000000;
				flagfangxiang=0;
			}
		}

//		USB_OTG_dev.dev.usr_cb->DeviceDisconnected();
//  		USB_OTG_dev.dev.class_cb->DeInit(&USB_OTG_dev, 0);
////		USBD_USR_DeviceReset(USB_OTG_SPEED_FULL);SysChsngeDelay(0x123456);
////		//
////			USBD_USR_DeviceDisconnected ();
////		USBD_DevDisconnected(&USB_OTG_dev);
//		USB_OTG_StopDevice(&USB_OTG_dev);
////		SysChsngeDelay(0x123456);
//		//RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_OTG_FS, DISABLE) ;
//
////		delay_ms(5000);
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, DISABLE);
		TIM_Cmd(TIM3, DISABLE);

		SysChsngeDelay(0x123456);
		SysClkWasp=testsys;
		SysPreparePara();
	}
}


//发送HID报文
void Send_Report(u8 byte0,u8 byte1,u8 byte2,u8 byte3)
{
	u8 HID_Buffer[4];
	
	HID_Buffer[0]=byte0;
	HID_Buffer[1]=byte1;
	HID_Buffer[2]=byte2;
	HID_Buffer[3]=byte3;
	USBD_HID_SendReport(&USB_OTG_dev,HID_Buffer,4);
}


