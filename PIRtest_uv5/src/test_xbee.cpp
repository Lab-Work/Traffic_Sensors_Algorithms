#include <stm32f4xx.h>

#include "WaspXBee.h"
//#include "wiring_private.h"
#include "dingke_delay.h"

int testxbee(void)
{
	uint8_t str1[30]={"this is test of xbee\n"};
	uint8_t str2[30]={" test of xbee end \n"};
		
	uint8_t explain1[]={"if there is print (const char c/[/]) in the program, this is interrupt send by UART \n"};
	uint8_t explain2[]={"send a integer: "};
	uint8_t explain3[]={"send a hex: "};
	uint8_t explain4[]={"if there is print (char c) in the program, this is inquire send by UART .example:"};

	long testnum=-12345;
	uint32_t numhex=0x897a;
	uint8_t testch1='A',testch2='B',testch3='C';

	int recflag;
	uint8_t recbuffer[20];
	uint8_t reclenwant=10;
	int reclenfact;	

	XBee.begin();

	while(1)
	{
		XBee.print((const char *)str1);
				delay_ms(3000);
		XBee.print((const char *)explain1);
				delay_ms(3000);

		XBee.print((const char *)explain2);
				delay_ms(3000);
		XBee.print(testnum);
				delay_ms(3000);

		XBee.print((const char *)explain3);
				delay_ms(3000);
		XBee.print(numhex,16);
				delay_ms(3000);

		XBee.print((const char *)explain4);
				delay_ms(3000);
		XBee.print((char)testch1);XBee.print((char)testch2);XBee.print((char)testch3);
				delay_ms(3000);


		XBee.print((const char*)"example read a char \n");
		delay_ms(300);
		recflag=XBee.read();
//		if(recflag==UARTRECEMPTY)
//		{
//			XBee.print((const char*)"no data in rec buffer defined by author(rx_buffer3) ");
//		}
//		else if(recflag>=0)//接收正确的数据都是char型，所以肯定为正，
//		{
//			XBee.print((const char*)" the data is ");
//			delay_ms(3000);
//			XBee.print((char)recflag);				
//		}
//		else
//		{
//			XBee.print((const char*)" else err ");	
//		}
		delay_ms(3000);


		XBee.print((const char*)"example read string \n");
		delay_ms(300);
		reclenfact=XBee.readstr(recbuffer,reclenwant);
		if(reclenfact<0)
		{
			XBee.print((const char*)" err! reclenwant>RX_BUFFER_SIZE ");
		}
		else if(recflag==0)//接收正确的数据都是char型，所以肯定为正，
		{
			XBee.print((const char*)" no data ");				
		}
		else
		{
			recbuffer[reclenfact]=0;
			XBee.print((const char*)recbuffer);	
		}
		delay_ms(3000);



 		XBee.print((const char *)str2);
				delay_ms(3000);

	}
}

