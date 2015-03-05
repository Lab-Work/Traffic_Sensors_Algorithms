

//#ifndef __WPROGRAM_H__
//#include "WaspClasses.h"
//#endif
////#include "mytest.h"
//#include "sdio_sd.h"
  #include "mysdfat.h"	
  uint16_t FlagMySdFatc;
 //PD15
//void SD_poweronc(void)  
//{
//	GPIO_InitTypeDef GPIO_InitStructure;	
//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 	
//	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7  ;
//	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
//	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
//	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
//	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
//	GPIO_Init(GPIOD, &GPIO_InitStructure);
//	
//	GPIO_SetBits(GPIOD,GPIO_Pin_7);
//}
//
//void SD_poweroffc(void)  
//{
//	GPIO_InitTypeDef GPIO_InitStructure;	
//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 	
//	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7  ;
//	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
//	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
//	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
//	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
//	GPIO_Init(GPIOD, &GPIO_InitStructure);
//	
//	GPIO_ResetBits(GPIOD,GPIO_Pin_7);
//}


	
//char initc()
//{
//	if(SD_Init()!= SD_OK)//初始化SD卡
//	{
//	    FlagMySdFatc = INIT_FAILED;
//	    return INIT_FAILED_em;
//	}
//	else
//	{
//		 f_mount(0,&fs1);//初始化磁盘
//	}
//  FlagMySdFatc = NOTHING_FAILED;   
//  return NOTHING_FAILED_em;  
//}



////得到SD的fat文件系统对应的整个盘(或者说"0"盘)的大小
//FRESULT getUfatsizec(const TCHAR *path,uint64_t *size)
//{
//	FRESULT res;
// 	FATFS fs; 
//	FATFS *pfs =&fs;	
//
//	DWORD nclst;
//
// 	res=f_getfree("0",&nclst,&pfs);
//	if(res!=FR_OK)
//	{
//		return res;	 
//	}
//	//一共多少簇  每簇多少扇区  每扇区多少字节
//	*size = (uint64_t)(pfs->n_fatent-2) * (uint64_t)(pfs->csize) * 512;
//	return res;
//} 


////得到SD的fat文件系统对应的整个盘(或者说"0"盘)的剩余空间大小
//FRESULT getUfatfreesizec(const TCHAR *path,uint64_t *size)
//{
//	FRESULT res;
// 	FATFS fs; 
//	FATFS *pfs =&fs;	
//
//	DWORD nclst;
//
// 	res=f_getfree("0",&nclst,&pfs);
//	if(res!=FR_OK)
//	{
//		return res;	 
//	}
//	//一共多少簇  每簇多少扇区  每扇区多少字节
//	*size = (uint64_t)(pfs->free_clust) * (uint64_t)(pfs->csize) * 512;
//	return res;
//} 



//uint64_t getDiskSizec()
//{
//	uint64_t fsize;
//
//	FlagMySdFatc=getUfatsizec("0",&fsize);
//    return fsize;
//}

//uint64_t getDiskFreec()
//{
//	uint64_t fsize;
//	FlagMySdFatc=getUfatfreesizec("0",&fsize);
//    return fsize;
//}


//char* print_disk_infoc()
//{
////    struct fat_fs_struct* _fs;
////    _fs=fs;
//    
//    // check if the card is there or not
////  if (!isSD())
////  {
////    flag = CARD_NOT_PRESENT;
////    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
////    return buffer;
////  }
//
////    if(!fs)
////        return 0;
//
////    struct sd_raw_info disk_info;
////    if(!sd_raw_get_info(&disk_info))
////        return 0;
//
//    // update the publicly available size-related variables
///*下面两句话暂时屏蔽掉*/
////    diskFree = fat_get_fs_free(_fs);
////    diskSize = fat_get_fs_size(_fs);
// 
////    sprintf(buffer, "" \
////    "manuf:  0x%x\n" \   
////    "oem:    %s\n" \     
////    "prod:   %s\n" \     
////    "rev:    %x\n" \     
////    "serial: 0x%lx\n" \   
////    "date:   %u/%u\n" \  
////    "size:   %u MB\n" \
////    "free:   %lu/%lu\n" \
////    "copy:   %u\n" \     
////    "wr.pr.: %u/%u\n" \  
////    "format: %u\n",   
////    disk_info.manufacturer,          
////    (char*) disk_info.oem,           
////    (char*) disk_info.product,       
////    disk_info.revision,              
////    disk_info.serial,                
////    disk_info.manufacturing_month,    
////    disk_info.manufacturing_year,    
////    disk_info.capacity / 1024 / 1024,                
////    diskFree>>16,
////    diskSize>>16,
////    disk_info.flag_copy,             
////    disk_info.flag_write_protect_temp,
////    disk_info.flag_write_protect,    
////    disk_info.format); 
//
//    //sprintf(buffer,"size:   %u MB\n free:   %u\n",SDCardInfo.CardCapacity>>20,diskFree>>16);            
//    sprintf(buffer,"size:   %u MB ",SDCardInfo.CardCapacity>>20); 
//    return buffer;
//}




#define MAXPATHLENGTHSD 128
static char CommonPathNewSD[MAXPATHLENGTHSD];	//公共路径存放处
static
char * codelinkc( char *path)
{
	unsigned char i;
	if((path[0]=='0')&&(path[1]==':'))
	{
		for(i=0;i<MAXPATHLENGTHSD;i++)
		{
			if(path[i]=='\\')	// 是反斜线 \	//  特别的，在路径里面也是双反斜杠才行，例"0/07/070/0707\\51.TXT"
				CommonPathNewSD[i]='/';
			else
				CommonPathNewSD[i]=path[i];
	
			if(path[i]==0x00)break;	
		}
	}
	else
	{
		CommonPathNewSD[0]='0';
		CommonPathNewSD[1]=':';
		for(i=0;i<MAXPATHLENGTHSD;i++)
		{
			if(path[i]=='\\')	// 是反斜线 \	//  特别的，在路径里面也是双反斜杠才行，例"0/07/070/0707\\51.TXT"
				CommonPathNewSD[i+2]='/';
			else
				CommonPathNewSD[i+2]=path[i];
	
			if(path[i]==0x00)break;	
		}
	}	
	return CommonPathNewSD;	
}

static
void pathcuttailc( char *path)
{
	unsigned char i;
	unsigned char len;

	len=strlen(path);
	for(i=len-1;i>0;i--)
	{
		if((path[i]=='/')||(path[i]=='\\'))
		{
			path[i]=0x00;
			break;
		}
	}
}


uint8_t openFilec (
	FIL *fp,			/* Pointer to the blank file object */
	const char *path	/* Pointer to the file name */
)
{
	char *pathnew;
	FRESULT res;

	pathnew=codelinkc( (char *)path);
	res=f_open(fp,pathnew,FA_WRITE|FA_READ) ;
	FlagMySdFatc=res;
	if(res==FR_OK)return 1;
	else return 0;		
}





static
FRESULT   getFileInfc(
	const TCHAR *path,	/* Pointer to the file path */
	FILINFO *fno
)
{
	char *pathnew;

	pathnew=codelinkc( (char *)path);
	
	return f_stat (
	pathnew,	/* Pointer to the file path */
	fno		/* Pointer to file information to return */
	);	
}


/*
 * getFileSize (name) - answers the file size for filename in current folder
 *
 * answers a longint with the file "name" size in the current folder
 *
 * If the file is not available in the folder, it will answer -1, it will also
 * update the DOS.flag to FILE_OPEN_ERROR
 */
int32_t getFileSizec(const char* path)
{
	int32_t size;
	FILINFO fno;
	FRESULT res;

	res=getFileInfc(path,&fno);
	if(res==FR_OK)
	{
		size =fno.fsize;
		//FlagMySdFatc=NOTHING_FAILED;
		FlagMySdFatc=0;
		return size;
	}
	else
	{
		FlagMySdFatc=res;
		return -1;
	}
}

/*
 * getAttributes (name) - returns the attributes for a directory or file entry 
 *
 * returns the attributes for a directory or file entry in the current directory. The attributes
 * are encoded with two characters:
 *
 * - char #1: it is either "d" for a directory or "-" for a file entry
 * - char #2: is either "r" for read only, and "w" if the file/directory is also writeable
 *
 * If the file or directory is not available in the folder, it will answer "--", it will also
 * update the DOS.flag to FILE_OPEN_ERROR
 */
//uint8_t getAttributesc(const char* path)
//{
//	FILINFO fno;
//	FRESULT res;
//	
//	res=getFileInfc(path,&fno);
//	FlagMySdFatc=res;
//	if(res==FR_OK)
//	{
//		FlagMySdFatc=res;
//		return fno.fattrib;
//	}
//	else
//		return 0;
//}

//#define MAXLNLENR 1024
//unsigned char BufferStrLnRc[MAXLNLENR];

//#define MAXLNSCOPELENR DOS_BUFFER_SIZE
//#define MAXLNSCOPELENR  256
//unsigned char BufferStrLnScopeR[MAXLNSCOPELENR];
//static 
//FRESULT f_readlnc (
//	FIL *fp, 		/* Pointer to the file object */
//	unsigned char *str,		/* Pointer to data buffer */
//	UINT *size		/* Number of bytes to read */
//)
//{
//	FRESULT res;
//	//uint32_t len=1;
//	UINT  pbw;
//	unsigned int i=0;
//
//	*size=0;
//	while(1)
//	{
//		res=f_read (fp,str,1,&pbw);
//		if(res!=FR_OK)
//		{
//			return res;	 
//		}
//		BufferStrLnR[*size]=str[0];
//		(*size)++;
//
//		if((str[0]=='\n')||((*size)==MAXLNLENR))
//		{	
//			for(i=0;i<(*size);i++)
//			{
//				str[i]=BufferStrLnR[i];
//			}		
//			return res;
//		}		
//	}
////	return res;
//}
//static
//FRESULT readfirstln (
//	const TCHAR *path, 		/* Pointer to the file object */
//	unsigned char *str,		/* Pointer to data buffer */
//	UINT *size		/* Number of bytes to read */
//)
//{
//
//	FRESULT res;
//	FIL fp;
//	char *pathnew;
//
//	pathnew=codelink( (char *)path);
//	res=f_open(&fp,pathnew,FA_WRITE|FA_READ) ;
//	if(res!=FR_OK)
//	{
//		return res;	 
//	}
//	else
//	{
//		res=f_lseek (&fp, 0);
//		if(res!=FR_OK)
//		{
//			f_close (&fp);
//			return res;	
//		}
//		else
//		{
//			res=f_readln (&fp,str,size);
//			if(res!=FR_OK)
//			{
//				f_close (&fp);
//				return res;	
//			}
//			else
//			{
//				return f_close (&fp);
//			}			
//		}
//	}
//
//}



/*
 * cat (filename, offset, scope) 
 *
 * dumps into the DOS.buffer the amount of bytes in scope after offset 
 * coming from filename, it will also return it as a string
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the data read
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * If there was an error opening the file, the returned string will say so and
 * the DOS.flag will show the FILE_OPEN_ERROR bit active
 *
 * An example of calls to cat(filename, offset, scope) is:
 *
 * - DOS.cat("hola.txt", 3, 17): will show the 17 characters after jumping over 3 in the file "hola.txt"
 *
 * The information is sent back as a string where each one of the characters are 
 * printed one after the next, EOL ('\n') will be encoded as EOL, and will be
 * accounted as one byte, an example of this would be:
 *
 * un lugar
 * de la man
 *
 */
//char* catc (const char* path, int32_t offset, uint16_t scope)
//{
//	readSDc(path, buffer, offset,scope);
//	return buffer;
//}
//
//uint8_t* catBinc (const char* path, int32_t offset, uint16_t scope)
//{
//	readSDc(path, bufferBin, offset,scope);
//	return bufferBin;
//}

/*
 * cat (filename, offset, scope) 
 *
 * dumps into the DOS.buffer the amount of bytes in scope after offset 
 * coming from filename, it will also return it as a string
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the data read
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * If there was an error opening the file, the returned string will say so and
 * the DOS.flag will show the FILE_OPEN_ERROR bit active
 *
 * An example of calls to cat(filename, offset, scope) is:
 *
 * - DOS.cat("hola.txt", 3, 17): will show the 17 characters after jumping over 3 in the file "hola.txt"
 *
 * The information is sent back as a string where each one of the characters are 
 * printed one after the next, EOL ('\n') will be encoded as EOL, and will be
 * accounted as one byte, an example of this would be:
 *
 * un lugar
 * de la man
 *
 */
//uint8_t* WaspSD::catBin (const char* filename, int32_t offset, uint16_t scope)
//{
//	struct fat_file_struct* _fd;
//	_fd=fd;
//  // check if the card is there or not
//	if (!isSD())
//	{
//		flag = CARD_NOT_PRESENT;
//		flag |= FILE_OPEN_ERROR;
//		sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//		return bufferBin;
//	}
//    
//  // if scope is zero, then we should make it 
//  // ridiculously big, so that it goes on reading forever
//	if (scope <= 0) scope = 1000;
//  
//	flag &= ~(TRUNCATED_DATA | FILE_OPEN_ERROR);
//
//  // search file in current directory and open it 
//  // assign the file pointer to the general file pointer "fp"
//  // exit if error and modify the general flag with FILE_OPEN_ERROR
//	_fd = openFile(filename);
//	if(!_fd)
//	{
//		sprintf(buffer, "error opening %s", filename);
//		flag |= FILE_OPEN_ERROR;
//		return bufferBin;
//	}
//
//  // iterate through the file
//	byte maxBuffer = 1;  // size of the buffer to use when reading
//	uint8_t bufferSD[maxBuffer];
//	uint32_t cont = 0;
//  
//  // first jump over the offset
//	if(!fat_seek_file(_fd, &offset, FAT_SEEK_SET))
//	{
//		sprintf(buffer, "error seeking on: %s\n", filename);
//		fat_close_file(_fd);
//		return bufferBin;
//	}
//  
//	uint8_t readRet = fat_read_file(_fd, bufferSD, sizeof(bufferSD));
//
//  // second, read the data and store it in the DOS.buffer
//  // as long as there is room in it
//	while(readRet > 0 && scope > 0 && cont < BIN_BUFFER_SIZE)
//	{
//		for(uint8_t i = 0; i < maxBuffer; ++i)
//		{
//			bufferBin[cont++] = bufferSD[i];
//			scope--;
//			readRet = fat_read_file(_fd, bufferSD, sizeof(bufferSD));
//		}
//	}
//
//	fat_close_file(_fd);
//
//	return bufferBin;
//
//}

/*
 * catln (filename, offset, scope) 
 *
 * dumps into the DOS.buffer the amount of lines in scope after offset 
 * lines coming from filename, it will also return it as a string
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the data read
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * If there was an error opening the file, the returned string will say so and
 * the DOS.flag will show the FILE_OPEN_ERROR bit active
 *
 * An example of calls to catln(filename, offset, scope) is:
 *
 * - DOS.catln("hola.txt", 10, 5): will show the 5 lines following line number 10 in the file "hola.txt"
 *
 * The information is sent back as a string where each one of the file's lines are 
 * printed in one line, an example of this would be:
 *
 * en un lugar
 * de la mancha de
 * cuyo nombre no qui>>
 *
 * Lines end with EOF ('\n'), this is the symbol used to count the amount of lines.
 */
//char* catlnc (const char* path, uint32_t offset, uint16_t scope)
//{
//	FRESULT res;
//	FIL fp;
//	unsigned int size;
//	unsigned long count=0;
//	unsigned int strsizemax=0;
//
//
//	if(openFilec (&fp,path))
//	{		
//		res=f_lseek (&fp, 0);
//		if(res!=FR_OK)
//		{
//			f_close (&fp);
//			FlagMySdFatc=res;
//			return buffer;	
//		}
//	
//		count=0;
//		while(count<offset)
//		{	
//			res=f_readln (&fp,(unsigned char *)buffer,&size);
//			if(res!=FR_OK)
//			{
//				f_close (&fp);
//				FlagMySdFatc=res;
//				return buffer;	
//			}
//			count++;
//		}
//	
//		count=0;
//		BufferStrLnScopeR[0]=0;
//		while(1)
//		{	
//			if(scope==0)
//			{
//				res=f_close (&fp);
//				flag=res;
//				if(res!=FR_OK)
//				{
//					//???
//					return buffer;	
//				}
//			}
//			res=f_readln (&fp,(unsigned char *)buffer,&size);
//			if(res!=FR_OK)
//			{
//				f_close (&fp);
//				flag=res;
//				return buffer;	
//			}
//			strsizemax +=size;
//			if(strsizemax>=MAXLNSCOPELENR)
//			{
//				//flag=??
//				return buffer;
//			}
//			count++;
//			strncat((char *)BufferStrLnScopeR,(char *)buffer,size);
//			if(count==scope)
//			{
//				size=0;
//				while(1)
//				{
//					buffer[size]=BufferStrLnScopeR[size];
//					if(BufferStrLnScopeR[size]==0x00)
//					{
//						res=f_close (&fp);
//						flag=res;
//						if(res!=FR_OK)
//						{
//							//???
//							return buffer;	
//						}						
//					}
//					size++;
//				}
//			}
//		} 
//	}	
//	 return buffer;
//
//}

/*
 * indexOf ( filename, pattern, offset ) - search for first occurrence of a string in a file
 *
 * looks into filename for the first occurrence of the pattern after a certain offset. The
 * algorythm will jump over offset bytes before starting the search for the pattern. It will
 * returns the amount of bytes (as a longint) to the pattern from the offset
 *
 * Example, file hola.txt contains:
 *
 * hola caracola\nhej hej\n   hola la[EOF]
 *
 * The following table shows the results from searching different patterns:
 *
 * Command                            Answer
 * indexOf("hola.txt", "hola", 0)       0
 * indexOf("hola.txt", "hola", 1)       23
 * indexOf("hola.txt", "hej", 3)        11
 *
 * Notes:
 *
 * - the special characters like '\n' (EOL) are accounted as one byte
 * - files are indexed from 0 
 *
 * If there was an error opening the file, the buffer string will say so and
 * the DOS.flag will show the FILE_OPEN_ERROR bit active
 */
//int32_t indexOfc (const char* path, const char* pattern, uint32_t offset)
//{
//	FRESULT res;
//	FIL fp;
//	uint32_t len;
//	UINT  pbw;
//	uint32_t offsetnow;
//	int count;
//
//	char* str=NULL;
//
//	len=strlen(pattern);
////
//	str=(char *)calloc(6,sizeof(char));
//
//	if(str==NULL)
//	{
//		flag=0;
//		return -1;
//		//return FR_OK;
//	}
//	flag=openFile (&fp,path);
//	if(flag!=0)
//	{	
//		free(str);		
//		return -1;	 
//	}
//	offsetnow=offset;
//
//	while(1)
//	{
//		res=f_lseek (&fp, offsetnow);
//		if(res!=FR_OK)
//		{
//			free(str);
//			f_close (&fp);
//			flag=res;
//			return -1;	
//		}
//
//		res=f_read (&fp,str,len,&pbw);
//		if(res!=FR_OK)
//		{
//			free(str);
//			f_close (&fp);
//			flag=res;
//			return -1;	
//		}
//
//		if(strncmp(str,pattern,len)==0)
//		{
//			count = offsetnow-offset;
//			free(str);
//			res=f_close (&fp);
//			flag=res;
//			return count;		
//		}
//
//		else 
//		{
//			offsetnow++;	
//		}
//	}				
//}
//
//
//
//int32_t numlnc(const TCHAR *path)
//{
//	FRESULT res;
//	FIL fp;
//	uint32_t i;
//	UINT  pbw;
//	char str[1];
//	int32_t count;
//	int fsize;
//	
//	fsize=getFileSize(path);
//	if(fsize==-1)
//	{
//		return -1;	 
//	}
//
//	flag=openFile (&fp,path);
//	if(flag!=0)
//	{
//		return -1;	 
//	}
//
//	res=f_lseek (&fp, 0);
//	if(res!=FR_OK)
//	{
//		f_close (&fp);
//		flag=res;
//		return -1;	
//	}
//	
//	count=0;
//	for(i=0;i<fsize;i++)
//	{
//		res=f_read (&fp,str,1,&pbw);
//		if(res!=FR_OK)
//		{
//			f_close (&fp);
//			flag=res;
//			return -1;	
//		}
//		if(str[0]=='\n')
//		{
//			(count)++;
//		}
//	}
//	f_close (&fp);
//	return count;			
//
//
//}


int8_t isDirc(const char* path)
{
	FILINFO fno;
	FRESULT res;
	
	res=getFileInfc(path,&fno);
	if(res==FR_OK)
	{
		if(fno.fattrib==0x10)
			return 1;
		else return 0;
	}
	return -1;
}

//1: exist the file, 0:exist but not file maybe dir, -1 error
int8_t isFilec(const char* path)
{
	FILINFO fno;
	FRESULT res;
	
	res=getFileInfc(path,&fno);
	if(res==FR_OK)
	{
		if(fno.fattrib==0x20)
			return 1;
		else return 0;
	}
	return -1;
}

//能删除指定的路径  文件和文件夹都可以删除 例如 del ("0/07/070/0704"); del ("0/07/070/55.txt");
//Delete a File or folder , example: del ("0/07/070/0704"); del ("0/07/070/55.txt");

 //能删除指定的路径  文件和文件夹都可以删除 例如 del ("0/07/070/0704"); del ("0/07/070/55.txt");
//Delete a File or folder , example: del ("0/07/070/0704"); del ("0/07/070/55.txt");
//但是如果这个文件里面有东西则不能删除
uint8_t delc(const char* path)
{
	FRESULT res;
	char *pathnew;

	pathnew=codelinkc( (char *)path);
	//printf("%s",pathnew);
	res= f_unlink(pathnew);
	FlagMySdFatc=res;
	//printf(" res=%d ",res);
	if(res==FR_OK)return 1;
	else return 0;
}

uint8_t delFilec(const char* path)
{
  return delc(path);
}

uint8_t delDirc(const char* path)
{
  return delc(path);
}



//unsigned char CountDelDepth=3;
//删除此目录下的所有文件，如果是文件夹，则深入此文件夹把这个文件夹里面的文件删掉在返回
//如果删除了就返回1 ，其他返回0
uint8_t deldirallc(const char* path)
{
	
	FRESULT res=FR_OK;
	char *pathnew;
	DIR  direct;
	FILINFO finfo;
	unsigned char returnflag;
	
	pathnew=codelinkc( (char *)path);
	res=f_opendir(&direct,pathnew);
	if(res!=FR_OK)return 0;

	while(1)//读取目录下所有类型符合的文件
	{
		//printf("entwh");
		f_readdir(&direct,&finfo);
		//printf("read");
		if(!finfo.fname[0])//没有文件了
		{
			break;
		}
		//printf(" !fname ");
		if((finfo.fname[0]&0xf0)==0xe0)
			continue;
		//printf(" delf:fszie=%d fattrib=%x  %s ",finfo.fsize,finfo.fattrib,finfo.fname);		
		//if((finfo.fattrib!=AM_DIR))
		//如果是文件 或者是文件夹
		if((finfo.fattrib&AM_ARC)||(finfo.fattrib==AM_DIR))
		{	
			//把路径编程这个文件或者文件夹的路径	
			strncat(pathnew,"/",1);
			strncat(pathnew,finfo.fname,13);
			//printf(" pathnew=%s ",pathnew);

			if((finfo.fattrib&AM_ARC))//如果是文件，就删除此文件	
			{	
				res= f_unlink(pathnew);
				FlagMySdFatc=res;
				if(res!=FR_OK)return 0;
				//printf(" res=%d ",res);
				pathcuttailc(pathnew);//注意结束了要把路径还原成原来的路径				
			}
			else if((finfo.fattrib==AM_DIR))//如果是文件夹，则再次调用本函数.为了删除子文件夹里面的文件（如果是再是文件夹就再次调用这个文件）
			{								//当文件夹里面的文件都删掉的话，通过跳出前删除本文件夹来实现的
				returnflag=	deldirallc(pathnew);
				pathcuttailc(pathnew); //把 文件路径还原成原来的路径

				if(returnflag==1);
				else return returnflag;									
			}
		}		
	}

	res= f_unlink(pathnew);//跳出前删除本文件夹
	FlagMySdFatc=res;
	//printf(" res=%d ",res);
	if(res==FR_OK)
	{
		return 1;	
	}
	else return 0;
}



//也是有条件的，必须他的上级目录存在的才可以生成这个文件
////Create a file.  The upper level directory must exit.
//if path dont include '.',it is ok.  example:	createFile ("0/07/07A"); there is a file named 07A not a folder.
FRESULT createfilec (
	const char *path	/* Pointer to the file name */
)
{
	FRESULT res;
	char *pathnew;
	FIL fp;

	pathnew=codelinkc( (char *)path);  //FA_CREATE_NEW|FA__WRITTEN|
	res=f_open (&fp,pathnew, FA_CREATE_ALWAYS) ;
	if(res!=FR_OK)		
		return res;
	else
	{
		res=f_close (&fp) ;
		return res;		
	}
}

//也是有条件的，必须他的上级目录存在的才可以生成这个文件夹  , example: createDir ("0/07/070/0789");
////Create a folder.  The upper level directory must exit.	example: createDir ("0/07/070/0789");
//if path include '.', it is work, example: createDir ("0/07/074555.txt"); there will a folder named 074555.txt not a file
FRESULT createdirc (
	const char *path	/* Pointer to the directory path */
)
{
	FRESULT res;
	char *pathnew;

	pathnew=codelinkc( (char *)path);
	res=f_mkdir (pathnew) ;	
	return res;
}

//uint8_t mkdirc(const char* path)
//{
//	FlagMySdFatc= createdirc(path);
//	return FlagMySdFatc;
//}


//不清楚新建的是文件还是文件夹，按照path里面最后一个\那边有没有.出现，出现则认为是文件，否则按照文件夹处理
//the funtion is not good function, so, when the last level of path  include '.', the path will be thought as a file path, else, a folder path.
//example: 	"0/07/074/741.txt" is a file path, "0:0/07/073/0737" is a folder path


uint8_t createc(const char* path)
{
	FRESULT res;
	char len;
	char i;

	len=strlen(path);
	for(i=len-1;;i--)
	{
		if(path[i]=='.')
		{
			res=createfilec (path);
			FlagMySdFatc=res;
			if(res==FR_OK)
				return 1;
			else return 0;
		}
		if(path[i]=='/')break;
		if(path[i]==0)break;
	}
	//res=createfile(path);
	res=createdirc(path);
	FlagMySdFatc=res;
	if(res==FR_OK)
		return 1;
	else return 0;

}


 //如果遇到文件打不开之类的错误返回-1，其他返回文件数目个数，当里面没有文件时返回0
int8_t numFilesc(const char *path)//
{//

	int8_t count=0;
	FRESULT res=FR_OK;
	char *pathnew;
	DIR  direct;
	FILINFO finfo;

	pathnew=codelinkc( (char *)path);
	res=f_opendir(&direct,pathnew);
	if(res!=FR_OK)return -1;
	
	while(1)//读取目录下所有类型符合的文件
	{
		f_readdir(&direct,&finfo);
		if(!finfo.fname[0])//没有文件了
		{
			break;
		}
		if((finfo.fname[0]&0xf0)==0xe0)
			continue;
		//printf("   fszie=%d fattrib=%x  %s ",finfo.fsize,finfo.fattrib,finfo.fname);		
		if((finfo.fattrib&AM_ARC))
		//其实应该是等于0x20才是文件，但是我直接生成的文件属性是0x00，文件夹没有问题是0x10，
		//所以这里计算文件简单的认为不是文件夹即可	
		//if((finfo.fattrib!=AM_DIR))	
		{							
			(count)++;				
		}		
	}
	return count;//


}




uint8_t readSDc(const TCHAR *path, void *str, uint32_t offset,uint32_t length)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;
	len=length;

	if(openFilec (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			FlagMySdFatc=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_read (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				FlagMySdFatc=res;
				return 0;
				//return res;	
			}
			else
			{
				res=f_close (&fp);
				FlagMySdFatc=res;
				if(res!=FR_OK)
					{return 0;}
				else 
				  return 1;
			}			
		}
	}
	else return 0;
}



uint8_t writeSD3c(const char* path, const char* str, int32_t offset)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=strlen((const char *)str);
	if(len==0)
	{	
		FlagMySdFatc=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFilec (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			FlagMySdFatc=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				FlagMySdFatc=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				FlagMySdFatc=res;
				return 1;
				//return f_close (&fp);
			}			
		}
	}
	else
	{
		return 0;
	}
}


uint8_t writeSD2c(const char *path, uint8_t* str, int32_t offset)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=strlen((const char *)str);
	if(len==0)
	{	
		FlagMySdFatc=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFilec (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			FlagMySdFatc=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				FlagMySdFatc=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				FlagMySdFatc=res;
				return 1;
				//return f_close (&fp);
			}			
		}
	}
	else
	{
		return 0;
	}
}

uint8_t writeSDc(const char* path, const char* str, int32_t offset, int16_t length)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=length;
	if(len==0)
	{	
		FlagMySdFatc=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFilec (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			FlagMySdFatc=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				FlagMySdFatc=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				FlagMySdFatc=res;
				return 1;
				//return f_close (&fp);
			}			
		}
	}
	else
	{
		return 0;
	}
}






uint16_t cleanFlagsc(void)
{
  FlagMySdFatc = 0;
  return FlagMySdFatc;
}


