#ifndef __DINGKE_SMBUS_H
#define __DINGKE_SMBUS_H
#include <stm32f4xx.h>
//////////////////////////////////////////////////////////////////////////////////	 

////////////////////////////////////////////////////////////////////////////////// 	  


#ifdef __cplusplus
extern "C"{
#endif

void SMBus_Init(void);
void SMBus_Apply(void);

// Yanning, delare function
void SMBus_poweron(void);
	
void MEM_WRITE1(unsigned char slave_addR, unsigned char cmdR,unsigned int data); 
unsigned int MEM_READ1(unsigned char slave_addR, unsigned char cmdR);
unsigned char ACKaddress(unsigned char ADDRESS);
int readMlx90614Subadd(void);
void changeMlx90614Subadd(unsigned char INIT_addR, unsigned int CURR_addR);

//读环境温度
float readMlx90614AmbientTemp(unsigned char slave_addR);
//读目标温度
float readMlx90614ObjectTemp(unsigned char slave_addR)	;
 
#ifdef __cplusplus
} // extern "C"
#endif 
 
  
#endif
















