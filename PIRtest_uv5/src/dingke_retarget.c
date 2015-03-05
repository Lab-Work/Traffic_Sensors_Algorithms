#include <stdio.h>
#include <rt_misc.h>
#include "dingke_retarget.h"

#pragma import(__use_no_semihosting_swi)
//有了这个文件就可以用串口打印的东西写成printf的形式，很方便，
//当然要指定好哪个串口，或者其他
//#define PRINTFPORT 2
extern void serialWrite(unsigned char c, unsigned char portNum);  /* in serial.c */
extern int serialRead( unsigned char );

struct __FILE { int handle; /* Add whatever you need here */ };
FILE __stdout;
FILE __stdin;
FILE __stderr;

static int readachar=0;
//为了方便打印串口数据为printf
int fputc(int ch, FILE *f) 
{
  serialWrite((unsigned char) ch, PRINTFPORT);
  return 0;
}
//为了方便接收串口数据为scanf格式
int fgetc (FILE *fp)  {
	while(1)
	{
	readachar=serialRead(PRINTFPORT);
	if(readachar<0);
	else return readachar;
	}
//  return (0);
}


int fclose(FILE* f) {
return 0;
}

int ferror(FILE *f) {
  /* Your implementation of ferror */
  return EOF;
}
int fseek (FILE *fp, long nPos, int nMode)  {
  return (0);
}


int fflush (FILE *pStream)  {
  return (0);
}


void _ttywrch(int ch) 
{
  serialWrite((unsigned char) ch, PRINTFPORT);
}


void _sys_exit(int return_code) {
label:  goto label;  /* endless loop */
}

