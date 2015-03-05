#ifndef __DINGKE_PWR_H
#define __DINGKE_PWR_H
#define	PWR_ON		1
#define	PWR_OFF	0
#define	PWR_SENS_3V3	0x0001
#define	PWR_SENS1_5V	0x0002
#define	PWR_SENS2_5V	0x0004
#define	PWR_SENS3_5V	0x0008
#define	PWR_MUX_UART6	0x0010
#define	PWR_SD			0x0020
#define	PWR_XBEE		0x0040
#define	PWR_RTC			0x0080
#define	PWR_BAT			0x0100
		   
#include <stm32f4xx.h>

#ifdef __cplusplus
extern "C"{
#endif


void initPwr(uint32_t choose);
void 	initAllPwr(void);
void setPwr(uint32_t choose, uint16_t mode);
void	offAllPwr(void);
void	onAllPwr(void);
#ifdef __cplusplus
} // extern "C"
#endif 


#endif
