/*! \file WaspSD.h
    \brief Library for managing the SD Card
    
    Copyright (C) 2009 Libelium Comunicaciones Distribuidas S.L.
    http://www.libelium.com
 
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.
   
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
  
    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
    Version:		0.6

    Design:		David Gasc髇

    Implementation:	David Cuartielles, Alberto Bielsa, Roland Riegel, Ingo Korb, Aske Olsson

*/
 


 /*! \def WaspSD_h
    \brief The library flag
    
  */
#ifndef WaspSD_h
#define WaspSD_h

/******************************************************************************
 * Includes
 ******************************************************************************/

#include <inttypes.h>

#include "ff.h"	/* Basic integer types */

//the low level FAT16/32 libraries
/*
#include "sd_raw_config.h"
#include "sd_raw.h"
#include "partition.h"
#include "fat_config.h"
#include "fat.h" */

/******************************************************************************
 * Definitions & Declarations
 ******************************************************************************/
    typedef uint64_t offset_t;


 
/*! \def VERSION
    \brief Version Control
*/
#define VERSION "WaspSD 01a, (c) 2009 Libelium.com\n"

/*! \def FILESYSTEM_LINUX
    \brief determines the type of EOL character, uncomment for Windows, leave for LIN/MAC
*/
#define FILESYSTEM_LINUX

/*! \def DOS_BUFFER_SIZE
    \brief Buffer size for storing data
*/
/*! \def BIN_BUFFER_SIZE
    \brief Buffer size for storing binary data
 */
#define DOS_BUFFER_SIZE 256
#define	BIN_BUFFER_SIZE	100

/*! \def NAMES
    \brief shows information available from files and directories. It shows the name
 */
/*! \def SIZES
    \brief shows information available from files and directories. It shows the size
 */
/*! \def ATTRIBUTES
    \brief shows information available from files and directories. It shows the attributes
 */
#define NAMES 0
#define SIZES 1
#define ATTRIBUTES 2

/*! \def NOTHING_FAILED
    \brief Flag possible values. Nothing failed in this case
 */
/*! \def CARD_NOT_PRESENT
    \brief Flag possible values. Card not present in this case
 */
/*! \def INIT_FAILED
    \brief Flag possible values. Initialization failed in this case
 */
/*! \def PARTITION_FAILED
    \brief Flag possible values. Opening partition failed in this case
 */
/*! \def FILESYSTEM_FAILED
    \brief Flag possible values. Opening filesystem failed in this case
 */
/*! \def ROOT_DIR_FAILED
    \brief Flag possible values. Opening root directory failed in this case
 */
/*! \def TRUNCATED_DATA
    \brief Flag possible values. Data has been truncated in this case
 */
/*! \def FILE_OPEN_ERROR
    \brief Flag possible values. Opening a file failed in this case
 */
/*! \def FILE_CREATION_ERROR
    \brief Flag possible values. Creating a file failed in this case
 */
/*! \def DIR_CREATION_ERROR
    \brief Flag possible values. Creating a directory failed in this case
 */
/*! \def FILE_WRITING_ERROR
    \brief Flag possible values. Writing a file failed in this case
 */
#define NOTHING_FAILED 0
#define CARD_NOT_PRESENT 1
#define INIT_FAILED 2
#define PARTITION_FAILED 4
#define FILESYSTEM_FAILED 8
#define ROOT_DIR_FAILED 16
#define TRUNCATED_DATA 32
#define FILE_OPEN_ERROR 64
#define FILE_CREATION_ERROR 128
#define DIR_CREATION_ERROR 256
#define FILE_WRITING_ERROR 512


/*! \def NOTHING_FAILED_em
    \brief Flag error messages. Nothing failed in this case
 */
/*! \def CARD_NOT_PRESENT_em
    \brief Flag possible values. Card not present in this case
 */
/*! \def INIT_FAILED_em
    \brief Flag possible values. Initialization failed in this case
 */
/*! \def PARTITION_FAILED_em
    \brief Flag possible values. Opening partition failed in this case
 */
/*! \def FILESYSTEM_FAILED_em
    \brief Flag possible values. Opening filesystem failed in this case
 */
/*! \def ROOT_DIR_FAILED_em
    \brief Flag possible values. Opening root directory failed in this case
 */
#define NOTHING_FAILED_em "OK"
#define CARD_NOT_PRESENT_em "no SD in the slot"
#define INIT_FAILED_em "MMC/SD initialization failed"
#define PARTITION_FAILED_em "Opening partition failed"
#define FILESYSTEM_FAILED_em "Opening filesystem failed"
#define ROOT_DIR_FAILED_em "Opening root dir failed"


/*! \def SD_ON
    \brief SD Power Modes. ON in this case
 */
/*! \def SD_OFF
    \brief SD Power Modes. OFF in this case
 */
#define	SD_ON	1
//#define	SD_OFF	2

void SD_poweron(void);
void SD_poweroff(void);

/******************************************************************************
 * Class
 ******************************************************************************/
 
//! WaspSD Class
/*!
	WaspSD Class defines all the variables and functions used to manage the SD Card
 */
class WaspSD
{
  private:

  public:

  //! Variable : buffer containing the information coming from the card used to avoid calls to UART functions inside the library. Beware, there could be data longer than the buffer size
  char buffer[DOS_BUFFER_SIZE];
  
  //! Variable : buffer containing the binary information coming from the card used to avoid calls to UART functions inside the library. Beware, there could be data longer than the buffer size
  uint8_t bufferBin[BIN_BUFFER_SIZE];

  //! Variable : flag storing the state of the SD card during initialization and operation
  uint16_t flag;
  
  //这个完全是为了SD卡初始化后，初始化FAT系统，把盘当做"0"即0盘，具体这个要追溯到stm公司提供的ff.c文件里面函数的用法了，在客户使用时完全可以不看这个，具体可以看例程即知
  FATFS fs1;

  
  //! Variable : amount of free bytes in the drive
  offset_t diskFree;
  
  //! Variable : total byte size of the drive
  offset_t diskSize;
  
  //! class constructor
  /*!
  It does nothing
  \param void
  \return void
  */
  WaspSD();
  
  //! It clears the flag
  /*!
  It does nothing
  \param void
  \return 'flag' variable
   */
  uint16_t cleanFlags(void);
  
  //目前这里没有用
  //! It checks if there is a SD Card in the slot
  /*!
  \param void
  \return '1' if SD is present, '0' otherwise
   */
  uint8_t isSD ();
  
  //! It powers the SD card, initialize it and prepare it to work with
  /*!
  \param void
  \return void
  \sa close(), begin()
  */ 
  void ON();
  
  //! It powers off the SD card and closes the pointers
  /*!
  \param void
  \return void
  \sa close(), begin()
   */ 
  void OFF();
  
  //! It initializes the use of SD cards, looks into the root partition, opens the file system and initializes the public pointer that can be used to access the filesystem
  /*!
  \param void
  \return human readable string indicating success or possible errors that can be printed by the user directly
   */
  char* init();
  
  //! It closes the directory, filesystem and partition pointers
  /*!
  	This function closes all the pointers in use in the library, so that they can be reused again after a call to init(). It becomes very useful if e.g. the card is removed and inserted again
  \param void
  \return void
   */
  void close();
  
  //! It sets switch and sd_present pin as output and input
  /*!
  \param void
  \return void
   */
  void	begin();

  //! It sets power mode
  /*!
  \param uint8_t mode : SD_ON or SD_OFF
  \return void
   */
  void	setMode(uint8_t mode);
  
  //! It packs all the data about the disk into the buffer and returns it back. The buffer will then be available and offer the data to the developers as a human-readable encoded string.
  /*!
  \param void
  \return human readable string indicating success or possible errors that can be printed by the user directly
   */
  char* print_disk_info();
  
  //! It gets the total disk size
  /*!
  \param void
  \return the total size for the disk
   */
  offset_t getDiskSize();
  
  //! It gets the free disk size
  /*!
  \param void
  \return the total available space for the disk
   */
  offset_t getDiskFree();

  //! It changes the directory
  /*!
  \param const char* command : the directory we want to change to
  \return '1' on success, '0' if error
  \sa cd(struct fat_dir_entry_struct subdir_entry)
   */
  uint8_t cd(const char* command);
  
  //! It changes the directory
  /*!
  \param struct fat_dir_entry_struct subdir_entry : the directory we want to change to
  \return '1' on success, '0' if error
  \sa cd(const char* command)
   */
  uint8_t cd(struct fat_dir_entry_struct subdir_entry);
  
  //! It gets the amount of files in a directory
  /*!
  \param void
  \return '0' if no files, a negative value if error and a possitive value indicating the amount of files
   */
  int8_t numFiles(const char *path);
  
  //! It lists a directory
  /*!
  \param void
  \return 'buffer' variable containing the corresponding listing
  \sa ls(int offset), ls(int offset, int scope, uint8_t info)
   */
//  char* ls(void);
  
  //! It lists a directory
  /*!
  \param int offset : it jumps over "offset" filenames in the list
  \return 'buffer' variable containing the corresponding listing
  \sa ls(void), ls(int offset, int scope, uint8_t info)
   */
//  char* ls(int offset);
  
  //! It lists a directory
  /*!
  \param int offset : it jumps over "offset" filenames in the list
  \param int scope : it includes a total of "scope" filenames in the buffer
  \param int info : limits the amount of information to be sent back, ranges from NAMES, SIZES, to ATTRIBUTES
  \return 'buffer' variable containing the corresponding listing
  \sa ls(void), ls(int offset)
   */
//  char* ls(int offset, int scope, uint8_t info);
  
  //! It tests existence of files in the dd folder
  /*!
  \param const char* name : the file to find
  \param struct fat_dir_entry_struct* dir_entry : the directory pointer to find the file in
  \return '1' if the file is availabe, '0' otherwise
   */
  uint8_t find_file_in_dir(const char* name, struct fat_dir_entry_struct* dir_entry);
  
  //! It creates a directory
  /*!
  \param const char* dirname : the directory to create
  \return '1' on success, '0' otherwise
   */
//  uint8_t mkdir(const char* dirname);
  
  //! It checks if an entry is a file or a directory
  /*!
  \param struct fat_dir_entry_struct file_entry : the entry to check
  \return '1' if it is a directory, '0' otherwise
  \sa isDir(const char* dirname)
   */
//  uint8_t isDir(struct fat_dir_entry_struct file_entry);
  
  //! It checks if an entry is a file or a directory
  /*!
  \param const char* dirname : the entry to check
  \return '1' if it is a directory, '0' otherwise
  \sa isDir(struct fat_dir_entry_struct file_entry)
   */
  int8_t isDir(const char* path);
int8_t isFile(const char* path);  
  //! It deletes a file or a directory
  /*!
  	It allows only erasing depth one directories, thus if the user calls to erase a directory with subdirs, it will exit with error without erasing the directory .
  	It also allows erasing current directory "." under the same premises: it should contain no subdirectories or it will exit with error.
  	Thanks to this function, together with delFile, delDir and isDir it is possible to create more complex delete functions that could iterate through any directory structure
  \param const char* name : the file or directory to delete
  \return '1' on success, '0' otherwise
  \sa delDir(uint8_t depth), delFile(struct fat_dir_entry_struct file_entry)
   */
  uint8_t del(const char* name);

//  uint8_t deldirallfiles(const char* path);
  uint8_t deldirall(const char* path);
  
  //! It deletes a directory
  /*!
  It allows only erasing depth one directories, thus if the user calls to erase a directory with subdirs, it will exit with error without erasing the directory .
  It also allows erasing current directory "." under the same premises: it should contain no subdirectories or it will exit with error.
  Thanks to this function, together with delFile, del and isDir it is possible to create more complex delete functions that could iterate through any directory structure
  \param uint8_t depth : is still not used, but function supports developers actualizations
  \return '1' on success, '0' otherwise
  \sa del(const char* name), delFile(struct fat_dir_entry_struct file_entry)
   */
  uint8_t delDir(const char* path);
  
  //! It deletes a file
  /*!
  Thanks to this function, together with delDir, del and isDir it is possible to create more complex delete functions that could iterate through any directory structure
  \param struct fat_dir_entry_struct file_entry : the file to delete
  \return '1' on success, '0' otherwise
  \sa del(const char* name), delDir(uint8_t depth)
   */
  uint8_t delFile(const char* path);
  
  //! It opens a file
  /*!
  \param const char* filename : the file to open
  \return '0' if error, file pointer on success
   */
//  struct fat_file_struct* openFile(const char* filename);



  uint8_t openFile (
	FIL *fp,			/* Pointer to the blank file object */
	const char *path	/* Pointer to the file name */
);
  
  //! It closes a file
  /*!
  \param struct fat_file_struct* _fd : the file to close
  \return void
   */
//  void closeFile (struct fat_file_struct* _fd);

  //! It tests the existence of a file in the current folder
  /*!
  \param const char* filename : the file to check
  \return '1' on success, '0' if it is a directory, '-1' otherwise
   */
//  int8_t isFile(const char* filename);
  
  //! It gets the amount of lines in a file
  /*!
  \param const char* filename : the file to check
  \return number of lines on success, '-1' otherwise
   */
//  int32_t numln (const char* filename);
  
  //! It gets the file size for filename in the current folder
  /*!
  \param const char* name : the file to check
  \return file size on success, '-1' otherwise
   */
  int32_t getFileSize(const char* name);
  uint8_t getAttributes(const char* path);
  //! It gets the attributes for a directory or file entry 
  /*!
  \param const char* name : the file or directory to check
  \return returns the attributes for a directory or file entry in the current directory. 
  	The attributes are encoded with two characters:
   		- char #1: it is either "d" for a directory or "-" for a file entry
   		- char #2: is either "r" for read only, and "w" if the file/directory is also writeable
   	If the file or directory is not available in the folder, it will answer "--"
   */
//  char* getAttributes(const char* name);

  //! It dumps into the buffer the amount of bytes in scope after offset coming from filename
  /*!
  	There is a limitation in size, due to DOS_BUFFER_SIZE. If the data read was bigger than that, the function will include the characters ">>" at the end and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to check this value to assure data integrity.
  \param const char* filename : the file to get the data from
  \param int32_t offset : amount of bytes to jump before start dumping the data to the buffer
  \param uint16_t scope : amount of bytes for dumping to the buffer
  \return 'buffer' variable where the data has been dumped
  \sa catBin (const char* filename, int32_t offset, uint16_t scope), catln (const char* filename, uint32_t offset, uint16_t scope)
   */
  char* cat (const char* filename, int32_t offset, uint16_t scope);
  
  //! It dumps into the bufferBin the amount of bytes in scope after offset coming from filename
  /*!
  \param const char* filename : the file to get the data from
  \param int32_t offset : amount of bytes to jump before start dumping the data to the buffer
  \param uint16_t scope : amount of bytes for dumping to the buffer
  \return 'bufferBin' variable where the data has been dumped
  \sa cat (const char* filename, int32_t offset, uint16_t scope), catln (const char* filename, uint32_t offset, uint16_t scope)
   */
  uint8_t* catBin (const char* filename, int32_t offset, uint16_t scope);
  
  //! It dumps into the buffer the amount of lines in scope after offset lines coming from filename
  /*!
  	There is a limitation in size, due to DOS_BUFFER_SIZE. If the data read was bigger than that, the function will include the characters ">>" at the end and activate the TRUNCATED_DATA value in the DOS.flag. It is recommened to check this value to assure data integrity.
  \param const char* filename : the file to get the data from
  \param uint32_t offset : amount of lines to jump before start dumping the data to the buffer
  \param uint16_t scope : amount of lines for dumping to the buffer
  \return 'buffer' variable where the data has been dumped
  \sa cat (const char* filename, int32_t offset, uint16_t scope), catBin (const char* filename, int32_t offset, uint16_t scope)
   */
  char* catln (const char* filename, uint32_t offset, uint16_t scope);
  
  //! It searches for first occurrence of a string in a file
  /*!
  \param const char* filename : the file where looking for the pattern
  \param const char* pattern : pattern to find
  \param uint32_t offset : amount of bytes to jump before start looking for the pattern
  \return the amount of bytes to the pattern from the offset
   */
  int32_t indexOf (const char* filename, const char* pattern, uint32_t offset);
  int32_t numln(const TCHAR *path);
  //! It creates a file
  /*!
  \param const char* filename : the file to create
  \return '1' on success, '0' otherwise
   */
  uint8_t create(const char* filename);
uint8_t mkdir(const char* path);  
uint8_t readSD(const TCHAR *path, void *str, uint32_t offset,uint32_t length);


  //! It writes strings to a file
  /*!
  \param const char* filename : the file to write to
  \param const char* str : the string to write into the file
  \param int32_t offset : amount of bytes to jump before start writing the string
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, const char* str), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)
   */
  uint8_t writeSD(const char* filename, const char* str, int32_t offset);
  
  //! It writes integer array to a file
  /*!
  \param const char* filename : the file to write to
  \param uint8_t* str : the integer array to write into the file
  \param int32_t offset : amount of bytes to jump before start writing the string
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), append(const char* filename, const char* str), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)

   */
  uint8_t writeSD(const char* filename, uint8_t* str, int32_t offset);
  
  //! It writes strings to a file of a specific length
  /*!
  \param const char* filename : the file to write to
  \param const char* str : the string to write into the file
  \param int32_t offset : amount of bytes to jump before start writing the string
  \param int16_t length : amount of bytes to write to the file
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, const char* str), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)
   */
  uint8_t writeSD(const char* filename, const char* str, int32_t offset, int16_t length);

  //! It writes strings at the end of files
  /*!
  \param const char* filename : the file to write to
  \param const char* str : the string to write into the file
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)
   */
//  uint8_t append(const char* filename, const char* str);
  
  //! It writes strings at the end of files of a specific length
  /*!
  \param const char* filename : the file to write to
  \param const char* str : the string to write into the file
  \param uint16_t length : the length to write
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)
   */
//  uint8_t append(const char* filename, const char* str, uint16_t length);
  
  //! It writes integer arrays at the end of files
  /*!
  \param const char* filename : the file to write to
  \param uint8_t* str : the integer array to write into the file
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, const char* str), appendln(const char* filename, const char* str), appendln(const char* filename, uint8_t* str)
   */
//  uint8_t append(const char* filename, uint8_t* str);
  
  //! It writes strings at the end of files adding an EOL
  /*!
  \param const char* filename : the file to write to
  \param const char* str : the string to write into the file
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, const char* str), append(const char* filename, uint8_t* str), appendln(const char* filename, uint8_t* str)
   */
//  uint8_t appendln(const char* filename, const char* str);
  
  //! It writes integer arrays at the end of files adding an EOL
  /*!
  \param const char* filename : the file to write to
  \param uint8_t* str : the integer array to write into the file
  \return '1' on success, '0' otherwise
  \sa writeSD(const char* filename, const char* str, int32_t offset), writeSD(const char* filename, uint8_t* str, int32_t offset), append(const char* filename, const char* str), append(const char* filename, uint8_t* str), appendln(const char* filename, const char* str)
   */
//  uint8_t appendln(const char* filename, uint8_t* str);
  

  //! It gets the library version
  /*!
  \param void
  \return the library version
   */
//  const char* getLibVersion(void) {return VERSION;};
};

/// END FUNCTIONS ///////////////////////////////////////////////////////////////////////

extern WaspSD SD;

#endif

