
#ifndef __DINGKE_USART_H
#define __DINGKE_USART_H

#include <stdio.h>
#include <inttypes.h>

//#include "delay.h"	
//#include "retarget.h"
#include "dingke_timer.h" //因为这里有个开启xbee的函数，顺便把定时器3开启了
#ifdef __cplusplus
extern "C"{
#endif

#define UARTRECEMPTY  -1//接收串口的数据，发现没有数据在自己建立的缓冲区（数组）里面
#define UARTRUNSUPPORT  -2
#define UARTRECERRELSE  -3//接收的其他错误类型



//下面串口4 5 是可以用的，测试通过的。 不过不建议使用原因如下
#define USEUSART4 0//串口4对应PA0 PA1口，而PA0口被ds3231的（/INT SQW）占用，所以要用串口4就不能用ds3231
#define USEUSART5 0//串口5对应PC2 PD2 这两个口分别被SD卡clk cmd占用，所以要用串口5就不能用SD卡


// Define constants and variables for buffering incoming serial data.  We're
// using a ring buffer (I think), in which rx_buffer_head is the index of the
// location to which to write the next incoming character and rx_buffer_tail
// is the index of the location from which to read.

//#define RX_BUFFER_SIZE 612
#define RX_BUFFER_SIZE 9000
//#define RX_BUFFER_SIZE_3 4000

#define RX80_BUFFER_SIZE 8000
#define RX_BUFFER_SIZE_3 RX_BUFFER_SIZE
#define RX_BUFFER_SIZE_1 100
#define RX_BUFFER_SIZE_2 500
#define RX_BUFFER_SIZE_6 100
#if USEUSART4==1
#define RX_BUFFER_SIZE_4 RX_BUFFER_SIZE
#endif

#if USEUSART5==1
#define RX_BUFFER_SIZE_5 RX_BUFFER_SIZE
#endif

void Mux_poweron(void);
void Mux_poweroff(void);
void muluart6init(void)	;
void muluart6choose(unsigned char choose);

void monitor_on(void);
void monitor_onuart3TX(void);
void monitor_onuart3RX(void);
void monitor_offuart3TX(void);
void monitor_offuart3RX(void);

void Xbee_poweron(void);
void Xbee_poweroff(void);
void beginxbee(long baud, uint8_t portNum);
void beginSerial(long, uint8_t);
void chooseuartinterrupt(uint8_t portNum);
void closeSerial(uint8_t);
void serialWrite(unsigned char, uint8_t);
int serialAvailable(uint8_t);
int serialRead(uint8_t);
void serialFlush(uint8_t);

int serialPreRx80Available(void) ;
int serialPreRx80Read(void);
uint8_t  handle1msprerx80(void);
int serialRx80Available(void);
int serialRx80Read(void);

//void printMode(int, uint8_t);
void printByte(unsigned char c, uint8_t);
//void printNewline(uint8_t);
void printString(const char *s, uint8_t);
//void printInteger(long n, uint8_t);
//void printHex(unsigned long n, uint8_t);
//void printOctal(unsigned long n, uint8_t);
//void printBinary(unsigned long n, uint8_t);
void printIntegerInBase(unsigned long n, unsigned long base, uint8_t);

void uartsendstr(char *s, uint8_t len,uint8_t portNum);
int serialReadstr(uint8_t *str, uint8_t len,uint8_t portNum);
void serialWritestr(char *s, uint8_t len,uint8_t portNum);

//void setup(void);
//void loop(void);

void beginMb7076(void);
int readMb7076( int* value);




#ifdef __cplusplus
} // extern "C"
#endif




#endif
