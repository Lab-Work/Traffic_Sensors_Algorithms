/*
这个是板子的功耗例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：
#define SPIFLASH 0			 FLASH例程
#define SPINULL 100			 不选用SPI例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()
*/
#include "allboardinc.h"
#define SPIFLASH 0
#define SPINULL 100

#define EXAMPLESPI SPINULL

//SPIFLASH
#if EXAMPLESPI==SPIFLASH
void setup()
{
	int readch;
	unsigned long k;
	uint8_t testflashbuf[4096];
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, PRINTFPORT);
	printf("test SPI\n");

	printf("init SPI_FLash");
	Flash.SPI_Flash_Init();//SPI FLASH初始化
	
	//在地址4092那边写上字符'4'(0x34)
	printf("\n  Write a char '4' to address=4092.");
	readch=Flash.writeFlash(4092,(uint8_t)'4');
	if(readch==FLASHSECTOEWRITEOK)//如果写成功了
	{
		printf("ok");
		//读地址4092处的数据,如果读到数据把数据给readch,若读不到数据或者其他错误，则错误信息（都是负数）给readch
		//此函数不符合客户提供的读数据写法，根据客户提供也有uint8_t readFlash1byte(uint32_t add)，uint16_t WaspFLASH::readFlash2byte(uint32_t add),uint32_t WaspFLASH::readFlash4byte(uint32_t add)
		// 但是根据客户那样函数，如果遇到芯片忙之类的就只能返回0而不能反应错误信息了
		printf("  Read a char from address=4092:");
		readch=Flash.flashreaddata4add(4092);
		if(readch<0)
		{
			printf(" err=%d ",readch);
			while(1);
		}
		else
		{
			//把读出来的数据（或错误类型）用串口3打印出来，其实这里的printf只针对串口3，关于这个printf是在retarget.c中把串口3的serialWrite((unsigned char) ch, 3)映射到fputc(int ch, FILE *f)了。
			// 如果客户不喜欢这个可以把retarget.c删掉，串口打印就用wiring_serial.c里面的void serialWrite(unsigned char c, uint8_t portNum)
			printf("%c\n",readch); 		
		}
	}										 
	else
	{
		printf(" the type of writeerr =%d ",readch);
		while(1);
		//delay_ms(30000);
	}

	printf("\n  Read datas with a sector size. if it work, the Array Flashbuf get these datas. ");
	//读从地址0开始的一个扇区数据，并把数据赋给Flashbuf[k]
	readch=Flash.readFlashsector(0);
	//如果读成功，打印一些数据 
	if(readch==FLASHSECTOEREADOK)
	{
		//打印地址0-9的数据
		printf(" show the datas with address=0 to 10:");
		for(k=0;k<10;k++)
		{
			printf("%x ",Flash.Flashbuf[k]);
		}

		//打印地址4090-4095的数据
		printf(" show the datas with address=4090 to 4096:");
		for(k=4090;k<4096;k++)
		{
			printf(" %x",Flash.Flashbuf[k]);delay_ms(500);
		}
		delay_ms(3000); 
	}
	else
	{
		printf(" the type of readerr=%d ",readch);
		while(1);
	}

	testflashbuf[0]='A';testflashbuf[1]='B'; testflashbuf[2]='C';testflashbuf[3]='D';
	
	testflashbuf[4091]=0x61;testflashbuf[4092]=0x62;testflashbuf[4093]=0x63;testflashbuf[4094]=0x64;
	//把字符串testflashbuf数据写到地址0所在的扇区，写4096个数据
	printf("\n  Write 4096 datas to sector 0(the first sector)");
	readch=Flash.writeFlash(0,testflashbuf,4096);
	//如果写成功，
	if(readch==FLASHSECTOEWRITEOK)
	{	
	    printf("  Read a data from address=0:");
		readch=Flash.flashreaddata4add(0);
		printf("%d\n",readch);
		  
	}
	//如果写不成功
	else
	{
		printf(" str writeerr yuanyin=%d ",readch);
		while(1);
	}

	//读一个扇区的内容，成功就打印部分数据
	printf("\n  Read again the sector 0.");
	readch=Flash.readFlashsector(0); 
	if(readch==FLASHSECTOEREADOK)
	{
		printf(" show the datas with address=0 to 10:");
		for(k=0;k<10;k++)
		{
			printf("%x ",Flash.Flashbuf[k]);
		}

		printf(" show the datas with address=4090 to 4096:");
		for(k=4090;k<4096;k++)
		{
			printf(" %x",Flash.Flashbuf[k]);delay_ms(300);
		}
		
	}
	else
	{
		printf(" the type of readerr=%d ",readch);
		while(1);
	}

	printf("\n  Read a data from address=0: ");
	printf(" %x ",Flash.readFlash1byte(0));

	printf("\n  Read a short data from address=0: ");
	printf(" %x ",Flash.readFlash2byte(0));

	printf("\n  Read a word data from address=0: ");
	printf(" %x ",Flash.readFlash4byte(0));

	printf("\nOK end");
}

void loop()
{
	while(1); 	
}
#endif //end SPIFLASH



