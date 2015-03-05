#ifndef __FSEARCH_H__
#define __FSEARCH_H__

#include "stm32f4xx_conf.h"
#include "ff.h"


#define FILMAXCNT  	128//定义最大扫描文件数(受缓冲大小限制)
#define PATHLEN	    50//定义路径缓冲大小


//支持的文件类型
typedef enum { 
	T_ANY=0,//任何文件
	T_DIR=1,//文件夹
	T_AUDIO=2,//音频文件
	T_IMAGE=3,//图片文件
	T_VIDO=4,//视频文件
	T_BOOK=5,//文本文件
	T_GAME=6,//游戏文件
	T_SYS=7,//系统文件	
	//在此增加新类型
	TYPENUM=8,//文件类型数					
}ftype;
//播放按钮	
#define T_PLAY 	TYPENUM	
#define T_PREV	TYPENUM+1	
#define T_NEXT	TYPENUM+2	
#define T_PAUSE	TYPENUM+3


//各个文件类型所包含的后缀名,逗号隔开
static const char* MYTYPE[]={
 	"...",
	"...",
 	"MP3,WMA,WAV",    	
 	"JPG,BMP",
	"AVI,MP4",
	"TXT,c,h",	        
 	"NES",		
 	"BIN,sys",									
};




extern u8   namebuf[FILMAXCNT*13];//能存放50个短文件名  
extern u32  sizebuf[FILMAXCNT];//每个文件对应的文件大小
extern u8   tribbuf[FILMAXCNT];//每个文件对应的文件属性
extern u8   path_curr[PATHLEN];//用于存放当前目录
extern u8   readbuf[20480];//读取缓冲





u16  FileandFolder_search(u8* path,ftype type);
u16 File_search(u8* path);
FRESULT File_searchindir(u8* path,unsigned char *num);


u8   Get_type(u8* fname,u8 fattrib);
u8   Get_suff(u8* fname);
void Add_path(u8* src,u8* dest);
void Cut_path(u8* src);
void My_chdir(u8* dir);

#endif





