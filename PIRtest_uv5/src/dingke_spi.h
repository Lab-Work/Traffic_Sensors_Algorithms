#ifndef __DINGKE_SPI_H
#define __DINGKE_SPI_H

#ifdef __cplusplus
extern "C"{
#endif

void SPI2_Init(void);


//将SPI2初始化为串行FLASH用途
void FLASH_SPI_Init(void);

//初始化所有SPI器件片选引脚并置高不选中
void CSPin_init(void);
u8 SPI2_RWByte(u8 byte);

#ifdef __cplusplus
}
#endif


#endif

