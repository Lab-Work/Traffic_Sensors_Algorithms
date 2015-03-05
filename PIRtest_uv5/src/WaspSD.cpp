/*
 *  Copyright (C) 2009 Libelium Comunicaciones Distribuidas S.L.
 *  http://www.libelium.com
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 2.1 of the License, or
 *  (at your option) any later version.
   
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
  
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Version:		0.10
 *  Design:		David Gasc贸n
 *  Implementation:	David Cuartielles, Alberto Bielsa, Roland Riegel, Ingo Korb, Aske Olsson
 */
 

#ifndef __WPROGRAM_H__
#include "WaspClasses.h"
#endif
//#include "mytest.h"
#include "sdio_sd.h"
    #include "mysdfat.h"	
// Variables ///////////////////////////////////////////////////////////////////

// Constructors ////////////////////////////////////////////////////////////////

WaspSD::WaspSD()
{
    // nothing to do
}

 //PD15
void SD_poweron(void)  
{
	GPIO_InitTypeDef GPIO_InitStructure;	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);
	
	GPIO_SetBits(GPIOD,GPIO_Pin_7);
}

void SD_poweroff(void)  
{
	GPIO_InitTypeDef GPIO_InitStructure;	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE); 	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);
	
	GPIO_ResetBits(GPIOD,GPIO_Pin_7);
}






// Public Methods //////////////////////////////////////////////////////////////

/*
 * ON (void) - It powers the SD card, initialize it and prepare it to work with
 *
 *  It powers the SD card, initialize it and prepare it to work with
 */
void WaspSD::ON(void)
{
	SD_poweron();
}


/*
 * OFF (void) - It powers off the SD card and closes the pointers
 *
 *  It powers off the SD card and closes the pointers
 */
void WaspSD::OFF(void)
{
	SD_poweroff() ;
}


/* begin() - sets switch and sd_present pin as output and input
* 
*/
void	WaspSD::begin()
{
  // activate the power to the SD card
/*  pinMode(MEM_PW, OUTPUT);
  pinMode(SD_PRESENT, INPUT);  */
}


/* setMode() - sets energy mode
* 
* It sets SD ON/OFF, switching On or Off the coresponding pin
* 
*/
void	WaspSD::setMode(uint8_t mode)
{
}


/* init ( void ) - initializes the use of SD cards, looks into the
 * root partition, opens the file system and initializes the
 * public pointer that can be used to access the filesystem
 * It answers a human readable string that can be printed
 * by the user directly.
 *
 * It updates the flagDOS with an error code indicating the
 * possible error messages:
 *
 * - NOTHING_FAILED 
 * - CARD_NOT_PRESENT
 * - INIT_FAILED 
 * - PARTITION_FAILED 
 * - FILESYSTEM_FAILED 
 * - ROOT_DIR_FAILED 
 */

	
char* WaspSD::init()
{
	if(SD_Init()!= SD_OK)//初始化SD卡
	{
	    flag = INIT_FAILED;
	    return INIT_FAILED_em;
	}
	else
	{

		 f_mount(0,&fs1);//初始化磁盘
	}

  flag = NOTHING_FAILED;   
  return NOTHING_FAILED_em;
  
}


/*
 * close (void) - closes the directory, filesystem and partition pointers
 *
 * this function closes all the pointers in use in the library, so that they 
 * can be reused again after a call to init(). It becomes very useful if e.g.
 * the card is removed and inserted again
 */
void WaspSD::close()
{
/*
  // close dir 
  fat_close_dir(dd);

  // close file system 
  fat_close(fs);

  // close partition 
  partition_close(partition);
  
  pinMode(SD_SS,INPUT);
  pinMode(SD_SCK,INPUT);
  pinMode(SD_MISO,INPUT);
  pinMode(SD_MOSI,INPUT);
  */
}


/*
 * isSD (void) - returns 1 if SD card is present, 0 otherwise
 *
 * here we make a call to close(), to avoid errors if users tried to call
 * any functions making use of the card
 */
uint8_t WaspSD::isSD()
{

 /* if (digitalRead(SD_PRESENT))    return 1;
    
  // if the SD is not there, the best is to close all the pointers
  // just to avoid problems later one with calls to functions  
  close(); */
  return 0;
}


//得到SD的fat文件系统对应的整个盘(或者说"0"盘)的大小
FRESULT getUfatsize(const TCHAR *path,uint64_t *size)
{
	FRESULT res;
 	FATFS fs; 
	FATFS *pfs =&fs;	

	DWORD nclst;

 	res=f_getfree("0",&nclst,&pfs);
	if(res!=FR_OK)
	{
		return res;	 
	}
	//一共多少簇  每簇多少扇区  每扇区多少字节
	*size = (uint64_t)(pfs->n_fatent-2) * (uint64_t)(pfs->csize) * 512;
	return res;
} 


//得到SD的fat文件系统对应的整个盘(或者说"0"盘)的剩余空间大小
FRESULT getUfatfreesize(const TCHAR *path,uint64_t *size)
{
	FRESULT res;
 	FATFS fs; 
	FATFS *pfs =&fs;	

	DWORD nclst;

 	res=f_getfree("0",&nclst,&pfs);
	if(res!=FR_OK)
	{
		return res;	 
	}
	//一共多少簇  每簇多少扇区  每扇区多少字节
	*size = (uint64_t)(pfs->free_clust) * (uint64_t)(pfs->csize) * 512;
	return res;
} 







/*
 * getDiskSize (void) - disk total size
 *
 * answers the total size for the disk, but also updates the
 * DOS.diskSize variable.
 */
offset_t WaspSD::getDiskSize()
{
	uint64_t fsize;

	flag=getUfatsize("0",&fsize);
    return fsize;
}

/*
 * getDiskFree (void) - disk free space
 *
 * answers the total available space for the disk, but also updates the
 * DOS.diskFree variable.
 */
offset_t WaspSD::getDiskFree()
{
	uint64_t fsize;
	flag=getUfatfreesize("0",&fsize);
    return fsize;
}

/*
 * print_disk_info (void) - disk information
 *
 * packs all the data about the disk into the buffer and returns it back. The
 * DS.buffer will then be available and offer the data to the developers as
 * a human-readable encoded string.
 *
 * An example of the output by this system would be:
 *
 * manuf:  0x2
 * oem:    TM
 * prod:   SD01G
 * rev:    32
 * serial: 0x9f70db88
 * date:   2/8
 * size:   947 MB
 * free:   991428608/992608256
 * copy:   0
 * wr.pr.: 0/0
 * format: 0
 *
 * Not all the information is relevant for further use. It is possible to access
 * different parts via the fat-implemented variables as you will see in this
 * function
 *
 * However, we have made a couple of the variables available for quick use. They are
 * the total avaiable space (DOS.diskFree) and the disk total size (DOS.diskSize). Both
 * are accounted in bytes
 */
char* WaspSD::print_disk_info()
{
//    struct fat_fs_struct* _fs;
//    _fs=fs;
    
    // check if the card is there or not
//  if (!isSD())
//  {
//    flag = CARD_NOT_PRESENT;
//    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//    return buffer;
//  }

//    if(!fs)
//        return 0;

//    struct sd_raw_info disk_info;
//    if(!sd_raw_get_info(&disk_info))
//        return 0;

    // update the publicly available size-related variables
/*下面两句话暂时屏蔽掉*/
//    diskFree = fat_get_fs_free(_fs);
//    diskSize = fat_get_fs_size(_fs);
 
//    sprintf(buffer, "" \
//    "manuf:  0x%x\n" \   
//    "oem:    %s\n" \     
//    "prod:   %s\n" \     
//    "rev:    %x\n" \     
//    "serial: 0x%lx\n" \   
//    "date:   %u/%u\n" \  
//    "size:   %u MB\n" \
//    "free:   %lu/%lu\n" \
//    "copy:   %u\n" \     
//    "wr.pr.: %u/%u\n" \  
//    "format: %u\n",   
//    disk_info.manufacturer,          
//    (char*) disk_info.oem,           
//    (char*) disk_info.product,       
//    disk_info.revision,              
//    disk_info.serial,                
//    disk_info.manufacturing_month,    
//    disk_info.manufacturing_year,    
//    disk_info.capacity / 1024 / 1024,                
//    diskFree>>16,
//    diskSize>>16,
//    disk_info.flag_copy,             
//    disk_info.flag_write_protect_temp,
//    disk_info.flag_write_protect,    
//    disk_info.format); 

    //sprintf(buffer,"size:   %u MB\n free:   %u\n",SDCardInfo.CardCapacity>>20,diskFree>>16);            
    sprintf(buffer,"size:   %u MB ",SDCardInfo.CardCapacity>>20); 
    return buffer;
}

#if FAT_DATETIME_SUPPORT
void WaspSD::get_datetime(uint16_t* year, uint8_t* month, uint8_t* day, uint8_t* hour, uint8_t* min, uint8_t* sec)
{
    *year = 2007;
    *month = 1;
    *day = 1;
    *hour = 0;
    *min = 0;
    *sec = 0;
}
#endif

/*
 * DOS_cd changes directory, answers 0 if error
 */

/*
uint8_t WaspSD::cd(struct fat_dir_entry_struct subdir_entry)
{
    struct fat_fs_struct* _fs;
    _fs=fs;
    
    struct fat_dir_struct* dd_new = fat_open_dir(_fs, &subdir_entry);
    if(dd_new)
    {
        fat_close_dir(dd);
        dd = dd_new;
        return 1;
    }
    return 0;
}
*/


/*
 * DOS_cd changes directory, answers 0 if error
 */
uint8_t WaspSD::cd(const char* command)
{
/*
    struct fat_fs_struct* _fs;
    _fs=fs;
    
    uint8_t exit=2;
    while(exit>0)
    {
        // check if the card is there or not
        if (!isSD())
        {
            flag = CARD_NOT_PRESENT;
            sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
            return 0;
        }
        struct fat_dir_entry_struct subdir_entry;
        if(find_file_in_dir(command, &subdir_entry))
        {
            struct fat_dir_struct* dd_new = fat_open_dir(_fs, &subdir_entry);
            if(dd_new)
            {
                fat_close_dir(dd);
                dd = dd_new;
                return 1;
            }
        }
        exit--;
        if(!exit) return 0;
    }

*/
return 0;
}


/*
 * ls ( offset, scope, info ) - directory listing
 *
 * returns and stores in DOS.buffer the listing of the current directory (DS.dd)
 * where the pointer is located. It has three parameters:
 *
 * - offset: it jumps over "offset" filenames in the list
 * - scope: it includes a total of "scope" filenames in the DOS.buffer
 * - info: limits the amount of information to be sent back, ranges from NAMES, SIZES, to ATTRIBUTES
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the directory listing
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * Examples of calls to ls(int, int, byte) are:
 *
 * - DOS.ls(2,0, NAMES): lists the name of the files 2nd, 3rd, 4th ... up to the end of the directory
 * - DOS.ls(4,2, SIZES): lists two files from position 4 including the size
 *
 * The information is sent back as a string where each file is listed in one line, an
 * example of this would be:
 *
 * empty.txt    2945
 * hola.txt     1149
 * secret/      0
 *
 * Files are shown with their extention, while folders have a slash as the last character. The
 * string is encoded with EOL ('\n'), EOS ('\0'), and tabulators ('\t') in case it needs further
 * parsing. The file size is shown in bytes.
 *
 */
//char* WaspSD::ls(int offset, int scope, uint8_t info = NAMES)
//{
//    struct fat_fs_struct* _fs;
//    struct fat_dir_struct* _dd;
//    _fs=fs;
//    _dd=dd;
//    
//  // check if the card is there or not
//  if (!isSD())
//  {
//    flag = CARD_NOT_PRESENT;
//    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//    return buffer;
//  }
//    
//  // if scope is zero, then we should make it 
//  // ridiculously big, so that it goes on reading forever
//  if (scope <= 0) scope = 1000;
//  
//  flag &= ~(TRUNCATED_DATA);
//  struct fat_dir_entry_struct dir_entry;
//  
//  // clean the offset (directory listings before the one requested)
//  if (offset-- > 0) while(fat_read_dir(_dd, &dir_entry) && offset > 0) 
//  {
//      offset--;
//  }
//
//  // declare the variables needed to iterate the listings
//  int buff_count = 0, j = 0, buff_size = 30, BUFF_SIZE = 30, dos_buff_left = DOS_BUFFER_SIZE;
//  
//  // iterate the directory
//  // if there is a directory entry, then create a string out of it
//  // and add it to the buffer
//  while(fat_read_dir(_dd, &dir_entry) && scope > 0)
//  {
//    buff_size = BUFF_SIZE;
//    
//    // create the line variable
//    char line[BUFF_SIZE];
//    
//    // fill in the variable with the corresponding string
//    switch (info)
//    {
//      case NAMES:
//        sprintf(line, "%s\n", dir_entry.long_name);
//        break;
//      case SIZES:
//        sprintf(line, "%s%c\t%lu\n", dir_entry.long_name, dir_entry.attributes & FAT_ATTRIB_DIR ? '/' : ' ', dir_entry.file_size);
//        break;
//      case ATTRIBUTES:
//        sprintf(line, "%c%c %s\t%lu\n", dir_entry.attributes & FAT_ATTRIB_DIR ? 'd' : '-', dir_entry.attributes & FAT_ATTRIB_READONLY ? 'r' : 'w', dir_entry.long_name, dir_entry.file_size);
//        break;
//      default:
//        // by default print only filenames, like the linux "ls" command
//        sprintf(line, "%s\n", dir_entry.long_name);
//        break;
//    }
//    
//    // add the line to the buffer (this won't work with another sprintf, libc-avr goes bananas)
//    for (j=0; j < BUFF_SIZE; j++) 
//    {
//      if (j + buff_count < DOS_BUFFER_SIZE - 4) 
//      {
//        if (line[j] != '\0') buffer[j + buff_count] = line[j];
//      
//        // in case you reach end of string, jump off the loop
//        if (buffer[j + buff_count] == '\n' || line[j] == '\0') 
//        {
//          buff_size = j + buff_count + 1;
//          //continue;
//          j = BUFF_SIZE;  // dirty way of leaving the loop
//        }
//      } 
//      else
//      {
//        // in case we filled up the whole buffer, we add a
//        // visual end of buffer indicator and leave the loop
//        buffer[DOS_BUFFER_SIZE - 4] = '>';
//        buffer[DOS_BUFFER_SIZE - 3] = '>';
//        buffer[DOS_BUFFER_SIZE - 2] = '\n';
//        buffer[DOS_BUFFER_SIZE - 1] = '\0';
//        flag |= TRUNCATED_DATA; 
//
//        // go to the end of the directory
//        while(fat_read_dir(_dd, &dir_entry)); 
//        return buffer;  // leave the function here
//      }
//    }
//    buff_count = buff_size;
//    scope--;
//  }
//
//  // add an end of string to the buffer
//  buffer[buff_count] = '\0';
//  
//  // go to the end of the directory
//  while(fat_read_dir(_dd, &dir_entry)); 
//
//  // return the buffer as long as there is any
//  // data inside, zero if the directory was empty or
//  // if there was an error
//  if (buff_count > 0) return buffer;
//  return 0;
//}

/*
 * ls ( offset ) - directory listing
 *
 * returns and stores in DOS.buffer the listing of one file within the current 
 * directory (DS.dd) where the pointer is located. It has one parameter:
 *
 * - offset: it jumps over "offset" filenames in the list
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the directory listing
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * Examples of calls to ls(int) are:
 *
 * - DOS.ls(0): lists the name of the first file in the current directory
 * - DOS.ls(7): lists the name of the 7th file in the current directory
 *
 * The information is sent back as a string where each file is listed in one line, an
 * example of this would be:
 *
 * empty.txt    2945
 * 
 * (for a file)
 *
 * secret/      0
 *
 * (for a directory)
 *
 * Files are shown with their extention, while folders have a slash as the last character. The
 * string is encoded with EOL ('\n'), EOS ('\0'), and tabulators ('\t') in case it needs further
 * parsing. The file size is shown in bytes.
 *
 */
//char* WaspSD::ls(int offset)
//{
//  return ls(offset, 1, SIZES);  // call DOS_ls with just the offset parameter
//}

/*
 * ls ( void ) - directory listing
 *
 * returns and stores in DOS.buffer the full listing of the current directory (DS.dd)
 * where the pointer is located. It has no parameters
 *
 * There is a limitation in size, due to DOS_BUFFER_SIZE. If the directory listing
 * was bigger than that, the function will include the characters ">>" at the end
 * and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to
 * check this value to assure data integrity.
 *
 * An example of calls to ls(void) is:
 *
 * - DOS.ls(): lists the name of all the files up to the end of the directory
 *
 * The information is sent back as a string where each file is listed in one line, an
 * example of this would be:
 *
 * empty.txt    2945
 * hola.txt     1149
 * secret/      0
 *
 * Files are shown with their extention, while folders have a slash as the last character. The
 * string is encoded with EOL ('\n'), EOS ('\0'), and tabulators ('\t') in case it needs further
 * parsing. The file size is shown in bytes.
 *
 */
//char* WaspSD::ls(void)
//{
//  return ls(0,0, SIZES);    // calls DOS_ls to list files until filling the buffer
//}

/*
 * find_file_in_dir (name, dir_entry) - tests existance of files in the dd folder
 *
 * answers whether the file or directory "name" is available in the current directory
 *
 * If the file is available in the folder, it will answer 1 (TRUE), if not
 * available it will answer 0 (FALSE)
 *
 */
uint8_t WaspSD::find_file_in_dir(const char* name, struct fat_dir_entry_struct* dir_entry)
{
/*
    struct fat_dir_struct* _dd;
    _dd=dd;
  // check if the card is there or not
    if (!isSD())
    {
        flag = CARD_NOT_PRESENT;
        sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
        return 0;
    }

    while(fat_read_dir(_dd, dir_entry))
    {
        if(strcmp(dir_entry->long_name, name) == 0)
        {
            fat_reset_dir(_dd);
            return 1;
        }
    }

*/

    return 0;
}

/*
 * isFile (filename) - tests existence of files in the current folder
 *
 * answers whether the file "filename" is available in the current directory
 *
 * If the file is available in the folder, it will answer 1 (TRUE), if it is not
 * a file it will answer 0 (FALSE), and if it is not available it will answer -1 (ERROR)
 *
 * Assumes there is a valid filesystem pointer (DOS.fs), and directory pointer (DOS.dd)
 */
//int8_t WaspSD::isFile(const char* filename)
//{
//    struct fat_dir_entry_struct file_entry;
//    if(find_file_in_dir(filename,&file_entry)) 
//    {
//        return file_entry.attributes & FAT_ATTRIB_DIR ? 0 : 1;
//    }
//    return -1;
//}

/*
 * openFile (filename) - opens a file
 *
 * opens the file "filename" if available and assigns the fp pointer to it, in the current directory
 *
 * If the file is not available in the folder, it will answer 0 (FALSE), it will also
 * update the DOS.flag to FILE_OPEN_ERROR
 *
 */
//struct fat_file_struct* WaspSD::openFile(const char* filename)
//{
///*
//    struct fat_fs_struct* _fs;
//    struct fat_dir_struct* _dd;
//    _fs=fs;
//    _dd=dd;	*/
//    struct fat_file_struct* _fd;
//    _fd=fd;
//    /*
//  // check if the card is there or not
//  if (!isSD())
//  {
//    flag = CARD_NOT_PRESENT;
//    flag |= FILE_OPEN_ERROR;
//    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//    return 0;
//  }
//
//  flag &= ~(FILE_OPEN_ERROR);
//  struct fat_dir_entry_struct file_entry;
//    if(!find_file_in_dir(filename,&file_entry))
//    {
//        flag |= FILE_OPEN_ERROR;
//        return 0;
//    }
//
//    _fd=fat_open_file(_fs,&file_entry);
//    if(!_fd) return fat_open_file(_fs,&file_entry);*/
//    return _fd;
//}

//#define MAXPATHLENGTHFILESD (128+14)
//char CommonPathNewFileSD[MAXPATHLENGTHFILESD];	//公共路径存放处

#define MAXPATHLENGTHSD 128
static char CommonPathNewSD[MAXPATHLENGTHSD];	//公共路径存放处
static
char * codelink( char *path)
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
void pathcuttail( char *path)
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


uint8_t WaspSD::openFile (
	FIL *fp,			/* Pointer to the blank file object */
	const char *path	/* Pointer to the file name */
)
{
	char *pathnew;
	FRESULT res;

	pathnew=codelink( (char *)path);
	res=f_open(fp,pathnew,FA_WRITE|FA_READ) ;
	flag=res;
	if(res==FR_OK)return 1;
	else return 0;		
}





static
FRESULT   getFileInf(
	const TCHAR *path,	/* Pointer to the file path */
	FILINFO *fno
)
{
	char *pathnew;

	pathnew=codelink( (char *)path);
	
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
int32_t WaspSD::getFileSize(const char* path)
{
	int32_t size;
	FILINFO fno;
	FRESULT res;

	res=getFileInf(path,&fno);
	if(res==FR_OK)
	{
		size =fno.fsize;
		flag=NOTHING_FAILED;
		return size;
	}
	else
	{
		flag=res;
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
uint8_t WaspSD::getAttributes(const char* path)
{
	FILINFO fno;
	FRESULT res;
	
	res=getFileInf(path,&fno);
	flag=res;
	if(res==FR_OK)
	{
		flag=res;
		return fno.fattrib;
	}
	else
		return 0;
}

#define MAXLNLENR 1024
unsigned char BufferStrLnR[MAXLNLENR];

#define MAXLNSCOPELENR DOS_BUFFER_SIZE
unsigned char BufferStrLnScopeR[MAXLNSCOPELENR];
static 
FRESULT f_readln (
	FIL *fp, 		/* Pointer to the file object */
	unsigned char *str,		/* Pointer to data buffer */
	UINT *size		/* Number of bytes to read */
)
{
	FRESULT res;
	//uint32_t len=1;
	UINT  pbw;
	unsigned int i=0;

	*size=0;
	while(1)
	{
		res=f_read (fp,str,1,&pbw);
		if(res!=FR_OK)
		{
			return res;	 
		}
		BufferStrLnR[*size]=str[0];
		(*size)++;

		if((str[0]=='\n')||((*size)==MAXLNLENR))
		{	
			for(i=0;i<(*size);i++)
			{
				str[i]=BufferStrLnR[i];
			}		
			return res;
		}		
	}
//	return res;
}
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
char* WaspSD::cat (const char* path, int32_t offset, uint16_t scope)
{
	readSD(path, buffer, offset,scope);
	return buffer;
}

uint8_t* WaspSD::catBin (const char* path, int32_t offset, uint16_t scope)
{
	readSD(path, bufferBin, offset,scope);
	return bufferBin;
}

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
char* WaspSD::catln (const char* path, uint32_t offset, uint16_t scope)
{
	FRESULT res;
	FIL fp;
	unsigned int size;
	unsigned long count=0;
	unsigned int strsizemax=0;


	if(openFile (&fp,path))
	{		
		res=f_lseek (&fp, 0);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return buffer;	
		}
	
		count=0;
		while(count<offset)
		{	
			res=f_readln (&fp,(unsigned char *)buffer,&size);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return buffer;	
			}
			count++;
		}
	
		count=0;
		BufferStrLnScopeR[0]=0;
		while(1)
		{	
			if(scope==0)
			{
				res=f_close (&fp);
				flag=res;
				if(res!=FR_OK)
				{
					//???
					return buffer;	
				}
			}
			res=f_readln (&fp,(unsigned char *)buffer,&size);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return buffer;	
			}
			strsizemax +=size;
			if(strsizemax>=MAXLNSCOPELENR)
			{
				//flag=??
				return buffer;
			}
			count++;
			strncat((char *)BufferStrLnScopeR,(char *)buffer,size);
			if(count==scope)
			{
				size=0;
				while(1)
				{
					buffer[size]=BufferStrLnScopeR[size];
					if(BufferStrLnScopeR[size]==0x00)
					{
						res=f_close (&fp);
						flag=res;
						if(res!=FR_OK)
						{
							//???
							return buffer;	
						}						
					}
					size++;
				}
			}
		} 
	}	
	 return buffer;

}

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
int32_t WaspSD::indexOf (const char* path, const char* pattern, uint32_t offset)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;
	uint32_t offsetnow;
	int count;

	char* str=NULL;

	len=strlen(pattern);
//
	str=(char *)calloc(6,sizeof(char));

	if(str==NULL)
	{
		flag=0;
		return -1;
		//return FR_OK;
	}
	flag=openFile (&fp,path);
	if(flag!=0)
	{	
		free(str);		
		return -1;	 
	}
	offsetnow=offset;

	while(1)
	{
		res=f_lseek (&fp, offsetnow);
		if(res!=FR_OK)
		{
			free(str);
			f_close (&fp);
			flag=res;
			return -1;	
		}

		res=f_read (&fp,str,len,&pbw);
		if(res!=FR_OK)
		{
			free(str);
			f_close (&fp);
			flag=res;
			return -1;	
		}

		if(strncmp(str,pattern,len)==0)
		{
			count = offsetnow-offset;
			free(str);
			res=f_close (&fp);
			flag=res;
			return count;		
		}

		else 
		{
			offsetnow++;	
		}
	}				
}



int32_t WaspSD::numln(const TCHAR *path)
{
	FRESULT res;
	FIL fp;
	uint32_t i;
	UINT  pbw;
	char str[1];
	int32_t count;
	int fsize;
	
	fsize=getFileSize(path);
	if(fsize==-1)
	{
		return -1;	 
	}

	flag=openFile (&fp,path);
	if(flag!=0)
	{
		return -1;	 
	}

	res=f_lseek (&fp, 0);
	if(res!=FR_OK)
	{
		f_close (&fp);
		flag=res;
		return -1;	
	}
	
	count=0;
	for(i=0;i<fsize;i++)
	{
		res=f_read (&fp,str,1,&pbw);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return -1;	
		}
		if(str[0]=='\n')
		{
			(count)++;
		}
	}
	f_close (&fp);
	return count;			


}



/*
 * isDir ( dir_entry ) - answers whether a file is a dir or not
 *
 * returns 1 if it is a dir, 0 if error, will
 * not modify any flags
 */



/*
uint8_t WaspSD::isDir(struct fat_dir_entry_struct dir_entry)
{
    return dir_entry.attributes & FAT_ATTRIB_DIR ? 1 : 0;
}*/

/*
 * isDir ( dirname ) - tests the existence of a directory in the current directory
 * and checks if it is adirectory or no
 *
 * returns 1 if it exists and it is a dir, 0 if exists but it is not a dir, -1 if error
 * will not modify any flags
 */
int8_t WaspSD::isDir(const char* path)
{
	FILINFO fno;
	FRESULT res;
	
	res=getFileInf(path,&fno);
	if(res==FR_OK)
	{
		if(fno.fattrib==0x10)
			return 1;
		else return 0;
	}
	return -1;
}

//1: exist the file, 0:exist but not file maybe dir, -1 error
int8_t WaspSD::isFile(const char* path)
{
	FILINFO fno;
	FRESULT res;
	
	res=getFileInf(path,&fno);
	if(res==FR_OK)
	{
		if(fno.fattrib==0x20)
			return 1;
		else return 0;
	}
	return -1;
}

/*
 * delFile ( file_entry ) - answers whether a file is deleted or not
 *
 * returns 1 if file_entry is deleted via fat command, 0 if error, will
 * not modify any flags
 */
 /*
uint8_t WaspSD::delFile(struct fat_dir_entry_struct file_entry)
{
    struct fat_fs_struct* _fs;
    _fs=fs;
    if(fat_delete_file(_fs,&file_entry)) return 1;
    return 0;
}*/
uint8_t WaspSD::delFile(const char* path)
{
  return del(path);
}

/*
 * delDir ( depth ) - answers whether a dir is deleted or not
 *
 * returns 1 if fileentry is deleted via iterative fat commands, 0 if error, will
 * not modify any flags
 * 'depth' is still not used, but function supports developers actualizations
 */
uint8_t WaspSD::delDir(const char* path)
{
  return del(path);
}
//能删除指定的路径  文件和文件夹都可以删除 例如 del ("0/07/070/0704"); del ("0/07/070/55.txt");
//Delete a File or folder , example: del ("0/07/070/0704"); del ("0/07/070/55.txt");

/*
 * del ( name ) - delete file or directory
 *
 * returns 1 if it is possible to delete "name", 0 if error, will
 * not modify any flags
 *
 * Version 1a, as for 20090512 allows only erasing depth one directories, thus
 * if the user calls to erase a directory with subdirs, it will exit with error
 * without erasing the directory
 *
 * It also allows erasing current directory "." under the same premises: it should
 * contain no subdirectories or it will exit with error
 *
 * Thanks to this function, together with delFile, delDir and isDir it is possible to
 * create more complex delete functions that could iterate through any directory structure
 */
 //能删除指定的路径  文件和文件夹都可以删除 例如 del ("0/07/070/0704"); del ("0/07/070/55.txt");
//Delete a File or folder , example: del ("0/07/070/0704"); del ("0/07/070/55.txt");
//但是如果这个文件里面有东西则不能删除
uint8_t WaspSD::del(const char* path)
{
	FRESULT res;
	char *pathnew;

	pathnew=codelink( (char *)path);
	//printf("%s",pathnew);
	res= f_unlink(pathnew);
	flag=res;
	//printf(" res=%d ",res);
	if(res==FR_OK)return 1;
	else return 0;
}


////删除此目录下的所有文件
//uint8_t WaspSD::deldirallfiles(const char* path)
//{	
//	FRESULT res=FR_OK;
//	char *pathnew;
//	DIR  direct;
//	FILINFO finfo;
//
//	pathnew=codelink( (char *)path);
//	res=f_opendir(&direct,pathnew);
//	if(res!=FR_OK)return 0;
//
//	while(1)//读取目录下所有类型符合的文件
//	{
//		f_readdir(&direct,&finfo);
//		if(!finfo.fname[0])//没有文件了
//		{
//			break;
//		}
//		if((finfo.fname[0]&0xf0)==0xe0)
//			continue;
//		printf(" delf:fszie=%d fattrib=%x  %s ",finfo.fsize,finfo.fattrib,finfo.fname);		
//		//if((finfo.fattrib!=AM_DIR))
//		if((finfo.fattrib&AM_ARC))//如果是文件，就删除此文件	
//		{
//			strcpy(CommonPathNewFileSD,pathnew);	
//			strncat(CommonPathNewFileSD,"/",1);
//			strncat(CommonPathNewFileSD,finfo.fname,13);
//			printf(" pathnew=%s ",CommonPathNewFileSD);
//
//			res= f_unlink(CommonPathNewFileSD);
//			flag=res;
//			if(res!=FR_OK)return 1;
//			printf(" res=%d ",res);				
//		}		
//	}
//	return 1;//
//}

//unsigned char CountDelDepth=3;
//删除此目录下的所有文件，如果是文件夹，则深入此文件夹把这个文件夹里面的文件删掉在返回
//如果删除了就返回1 ，其他返回0
uint8_t WaspSD::deldirall(const char* path)
{
	
	FRESULT res=FR_OK;
	char *pathnew;
	DIR  direct;
	FILINFO finfo;
	unsigned char returnflag;
	
	pathnew=codelink( (char *)path);
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
				flag=res;
				if(res!=FR_OK)return 0;
				//printf(" res=%d ",res);
				pathcuttail(pathnew);//注意结束了要把路径还原成原来的路径				
			}
			else if((finfo.fattrib==AM_DIR))//如果是文件夹，则再次调用本函数.为了删除子文件夹里面的文件（如果是再是文件夹就再次调用这个文件）
			{								//当文件夹里面的文件都删掉的话，通过跳出前删除本文件夹来实现的
				returnflag=	deldirall(pathnew);
				pathcuttail(pathnew); //把 文件路径还原成原来的路径

				if(returnflag==1);
				else return returnflag;									
			}
		}		
	}

	res= f_unlink(pathnew);//跳出前删除本文件夹
	flag=res;
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
FRESULT createfile (
	const char *path	/* Pointer to the file name */
)
{
	FRESULT res;
	char *pathnew;
	FIL fp;

	pathnew=codelink( (char *)path);  //FA_CREATE_NEW|FA__WRITTEN|
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
FRESULT createdir (
	const char *path	/* Pointer to the directory path */
)
{
	FRESULT res;
	char *pathnew;

	pathnew=codelink( (char *)path);
	res=f_mkdir (pathnew) ;	
	return res;
}

uint8_t WaspSD::mkdir(const char* path)
{
	flag= createdir(path);
	return flag;
}


//不清楚新建的是文件还是文件夹，按照path里面最后一个\那边有没有.出现，出现则认为是文件，否则按照文件夹处理
//the funtion is not good function, so, when the last level of path  include '.', the path will be thought as a file path, else, a folder path.
//example: 	"0/07/074/741.txt" is a file path, "0:0/07/073/0737" is a folder path

/*
 * create ( filename ) - create file
 *
 * returns 1 on file creation, 0 if error, will mark the flag with
 * FILE_CREATION_ERROR 
 */
uint8_t WaspSD::create(const char* path)
{
	FRESULT res;
	char len;
	char i;

	len=strlen(path);
	for(i=len-1;;i--)
	{
		if(path[i]=='.')
		{
			res=createfile (path);
			flag=res;
			if(res==FR_OK)
				return 1;
			else return 0;
		}
		if(path[i]=='/')break;
		if(path[i]==0)break;
	}
	//res=createfile(path);
	res=createdir(path);
	flag=res;
	if(res==FR_OK)
		return 1;
	else return 0;

}

/*
 * mkdir ( dirname ) - create directory
 *
 * returns 1 on directory creation, 0 if error, will mark the flag with
 * DIR_CREATION_ERROR 
 */
//uint8_t WaspSD::mkdir(const char* dirname)
//{
//    struct fat_dir_struct* _dd;
//    _dd=dd;
//  // check if the card is there or not
//  if (!isSD())
//  {
//    flag = CARD_NOT_PRESENT;
//    flag |= DIR_CREATION_ERROR;
//    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//    return 0;
//  }
//
//  flag &= ~(DIR_CREATION_ERROR);
//  struct fat_dir_entry_struct dir_entry;
//  if( (find_file_in_dir(dirname,&dir_entry)) || (!fat_create_dir(_dd, dirname, &dir_entry)) )
//  {
//      flag |= DIR_CREATION_ERROR;
//      return 0;
//  }
//  return 1;
//}

/*
 * numFiles ( void ) - returns the amount of files in dir
 *
 * returns the amount of files in the current directory
 * a negative answer indicates error, zero means no files in the folder
 */
 //如果遇到文件打不开之类的错误返回-1，其他返回文件数目个数，当里面没有文件时返回0
int8_t WaspSD::numFiles(const char *path)//
{//

	int8_t count=0;
	FRESULT res=FR_OK;
	char *pathnew;
	DIR  direct;
	FILINFO finfo;

	pathnew=codelink( (char *)path);
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

/*
 * append ( filename, str ) - write strings at the end of files
 *
 * writes the string "str" at the end of the file "filename"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
//uint8_t WaspSD::append(const char* filename, const char* str)
//{
//    return writeSD(filename, str, getFileSize(filename));
//}

/*
 * append ( filename, str, length ) - write strings at the end of files
 *
 * writes the string "str" at the end of the file "filename"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
//uint8_t WaspSD::append(const char* filename, const char* str, uint16_t length)
//{
//	return writeSD(filename, str, getFileSize(filename), length);
//}


/*
 * append ( filename, str ) - write array of numbers at the end of files
 *
 * writes the array of numbers "str" at the end of the file "filename"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
//uint8_t WaspSD::append(const char* filename, uint8_t* str)
//{
//	return writeSD(filename, str, getFileSize(filename));
//}

/*
 * appendln ( filename, str ) - write strings at the end of files
 *
 * writes the string "str" at the end of the file "filename" adding end
 * of line
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
//uint8_t WaspSD::appendln(const char* filename, const char* str)
//{
//    uint8_t exit = 0;
//    exit = append(filename, str);
//#ifndef FILESYSTEM_LINUX
//    if (exit) exit &= append(filename, "\r");
//#endif
//    if (exit) exit &= append(filename, "\n");
//    return exit;
//}

/*
 * appendln ( filename, str ) - write array of numbers at the end of files
 *
 * writes the array of numbers "str" at the end of the file "filename" adding end
 * of line
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
//uint8_t WaspSD::appendln(const char* filename, uint8_t* str)
//{
//	uint8_t exit = 0;
//	exit = append(filename, str);
//#ifndef FILESYSTEM_LINUX
//	if (exit) exit &= append(filename, "\r");
//#endif
//	if (exit) exit &= append(filename, "\n");
//	return exit;
//}



uint8_t WaspSD::readSD(const TCHAR *path, void *str, uint32_t offset,uint32_t length)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;
	len=length;

	if(openFile (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_read (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return 0;
				//return res;	
			}
			else
			{
				res=f_close (&fp);
				flag=res;
				if(res!=FR_OK)
					{return 0;}
				else 
				  return 1;
			}			
		}
	}
	else return 0;
}




/*
 * writeSD ( filename, str, offset ) - write strings to files
 *
 * writes the string "str" to the file "filename" after a certain "offset"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
uint8_t WaspSD::writeSD(const char* path, const char* str, int32_t offset)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=strlen((const char *)str);
	if(len==0)
	{	
		flag=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFile (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				flag=res;
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

/*
 * writeSD ( filename, str, offset ) - write numbers to files
 *
 * writes the aray of integers "str" to the file "filename" after a certain "offset"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
uint8_t WaspSD::writeSD(const char *path, uint8_t* str, int32_t offset)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=strlen((const char *)str);
	if(len==0)
	{	
		flag=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFile (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				flag=res;
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


/*
 * writeSD ( filename, str, offset, length ) - write strings to files
 *
 * writes the string "str" to the file "filename" after a certain "offset"
 *
 * returns 1 on success, 0 if error, will mark the flag with
 * FILE_WRITING_ERROR
 */
uint8_t WaspSD::writeSD(const char* path, const char* str, int32_t offset, int16_t length)
{
	FRESULT res;
	FIL fp;
	uint32_t len;
	UINT  pbw;

	len=length;
	if(len==0)
	{	
		flag=FR_OK;
		return 1;
		//return FR_OK;
	}

	if(openFile (&fp,path))
	{
		res=f_lseek (&fp, offset);
		if(res!=FR_OK)
		{
			f_close (&fp);
			flag=res;
			return 0;
			//return res;	
		}
		else
		{
			res=f_write (&fp,str,len,&pbw);
			if(res!=FR_OK)
			{
				f_close (&fp);
				flag=res;
				return 0;
				//return res;	
			}
			else
			{	
				res= f_close (&fp);
				flag=res;
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






/*
 * numln ( filename ) - returns the amount of lines in file
 *
 * returns the amount of lines in "filename" that should be in the current directory,
 * a negative answer indicates error, zero means no lines in the file
 *
 * This method counts the occurrence of the character '\n' in the file. If there
 * was a problem opening it, the FILE_OPEN_ERROR would be activated and will return -1
 */
//int32_t WaspSD::numln(const char* filename)
//{
//    struct fat_file_struct* _fd;
//    _fd=fd;
//  // check if the card is there or not
//  if (!isSD())
//  {
//    flag = CARD_NOT_PRESENT;
//    flag |= FILE_OPEN_ERROR;
//    sprintf(buffer,"%s", CARD_NOT_PRESENT_em);
//    return -1;
//  }
//    
//  flag &= ~(FILE_OPEN_ERROR);
//
//  // search file in current directory and open it 
//  // assign the file pointer to the general file pointer "fp"
//  // exit if error and modify the general flag with FILE_OPEN_ERROR
//  _fd = openFile(filename);
//  if(!_fd)
//  {
//    sprintf(buffer, "error opening %s", filename);
//    flag |= FILE_OPEN_ERROR;
//    return -1;
//  }
//
//  byte maxBuffer = 1;  // size of the buffer to use when reading
//  uint8_t bufferSD[maxBuffer];
//  uint32_t cont = 0;
//  
//  // jump over offset lines
//  uint8_t readRet = fat_read_file(_fd, bufferSD, sizeof(bufferSD));
//  while( readRet > 0)   
//  {
//    for(uint8_t i = 0; i < maxBuffer; ++i)
//    {
//      if (bufferSD[i] == '\n')
//        cont++;
//    }
//    readRet = fat_read_file(_fd, bufferSD, sizeof(bufferSD));
//  }
//
//  fat_close_file(_fd);
//
//  return cont;
//
//}

// Utils ////////////////////////////////////////////////////////////////////

/*
 * cleanFlags ( void ) - resets all the flags, returns the flags
 */
uint16_t WaspSD::cleanFlags(void)
{
  flag = 0;
  return flag;
}


// Private Methods /////////////////////////////////////////////////////////////

// Preinstantiate Objects //////////////////////////////////////////////////////

WaspSD SD = WaspSD();


