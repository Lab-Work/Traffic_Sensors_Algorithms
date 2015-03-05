#ifndef __DINGKE_EXTI_H
#define __DINGKE_EXTI_H

#include "stm32f4xx_conf.h"

//#include "led.h"
#ifdef __cplusplus
extern "C"{
#endif
extern uint16_t CNTEXIT;
//extern uint8_t  FlagExit12;
void GPIO_def(void);

void exti_def(void);

void NIVC_def(void);

void EXTI0_IRQHandler(void);
void setPowerEnoughAsAwake(void);
#ifdef __cplusplus
} // extern "C"
#endif

static u8 EXTFLAG=0;


#endif

