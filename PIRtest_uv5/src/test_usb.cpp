#include <stm32f4xx.h>
#include <stdio.h>
#include "stm32f4xx_conf.h"
#include "delay.h"

	
#include "usbd_hid_core.h"
#include "usbd_usr.h"
#include "usbd_desc.h"


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
int testusbhid(void)
{	




		
	//硬件初始化
 	//Key_Init();//按键初始化
	//   beginSerial(9600, 3);

	
	USBD_Init(&USB_OTG_dev,USB_OTG_FS_CORE_ID,&USR_desc,&USBD_HID_cb,&USR_cb);//USB从机初始化

	while(1)
	{
		//光标上移
		Send_Report(0,0,250,0);//发送报文	
		delay_ms(1000);
		//光标下移
		Send_Report(0,0,6,0);//发送报文	
		delay_ms(1000);
		//光标左移
		Send_Report(0,250,0,0);//发送报文	
		delay_ms(1000);
		//光标右移
		Send_Report(0,6,0,0);//发送报文	
		delay_ms(1000);	
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


