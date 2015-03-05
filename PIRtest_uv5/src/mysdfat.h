#ifndef MySdFat_h
#define MySdFat_h

//#include <stdio.h>
//#include <stdarg.h>
#include <string.h>

#include "sdio_sd.h"
  #include "ff.h"	
#ifdef __cplusplus
extern "C"{
#endif

extern uint16_t FlagMySdFatc;
#define MAXPATHLENGTHSD 128
uint8_t openFilec (
	FIL *fp,			/* Pointer to the blank file object */
	const char *path	/* Pointer to the file name */
);
int32_t getFileSizec(const char* path);
int8_t isDirc(const char* path)	;
int8_t isFilec(const char* path) ;
uint8_t delc(const char* path);
uint8_t delFilec(const char* path);
uint8_t deldirallc(const char* path);

FRESULT createfilec (const char *path);
FRESULT createdirc (const char *path);
uint8_t createc(const char* path);
int8_t numFilesc(const char *path);
uint8_t readSDc(const TCHAR *path, void *str, uint32_t offset,uint32_t length);
uint8_t writeSD3c(const char* path, const char* str, int32_t offset);
uint8_t writeSD2c(const char *path, uint8_t* str, int32_t offset);
uint8_t writeSDc(const char* path, const char* str, int32_t offset, int16_t length)	;
uint16_t cleanFlagsc(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif

