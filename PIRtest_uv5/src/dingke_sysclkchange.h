#ifndef __DINGKE_SYSCLKCHANGE_H
#define __DINGKE_SYSCLKCHANGE_H
#include "stm32f4xx_conf.h"

#define CLKCHANGEDELAY1 29000000
#define CLKCHANGEDELAY2 27000000

extern unsigned long SysClkWasp;
extern unsigned long DelayMsWasp;
extern unsigned long DelayUsWasp;
extern unsigned long SysPllN,SysPllM,SysPllP,SysPllQ;
extern unsigned long SysFlashWait;
#ifdef __cplusplus
extern "C"{
#endif
void SysChsngeDelay(u32 count);
void SysPreparePara(void);
void zppSystemInit(void);

#ifdef __cplusplus
} // extern "C"
#endif
#endif
