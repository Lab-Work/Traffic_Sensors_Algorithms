#ifndef __DINGKE_WDG_H
#define __DINGKE_WDG_H			   
#include <stm32f4xx.h>
//////////////////////////////////////////////////////////////////////////////////	 


////////////////////////////////////////////////////////////////////////////////// 
#ifdef __cplusplus
extern "C"{
#endif


void IWDG_Init(uint8_t prer,uint16_t rlr);
void IWDG_Feed(void);

#ifdef __cplusplus
} // extern "C"
#endif 



#endif





























