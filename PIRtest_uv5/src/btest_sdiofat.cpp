/*
这个是板子的功耗例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：     
#define BTESTSDIOFAT 0		 SD卡读写程序
#define BTESTSDIONULL 100	 不选用功耗例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()
*/
#include "allboardinc.h"
#define BTESTSDIOFAT 0
#define BTESTSDIONULL 100

#define EXAMPLESDIO BTESTSDIONULL
/*
SD卡读写程序
    涉及函数：开SD电源，SD初始化，得到SD总空间，可用剩余空间，生成、删除文件夹和文件，
	      判断文件夹或文件，读出文件夹里文件数目，按要求读写文件里面数据，得到文件大小
*/
//SDIOFAT
#if EXAMPLESDIO==BTESTSDIOFAT
void setup()
{
	uint8_t readstr1[512]="strhaha";
	uint64_t fsize;
	int8_t filecnt=-2;
	int32_t filesize=-2;
		
//开串口
	Mux_poweron(); //借用一个电源正给串口用的，方便打印数据
	beginSerial(115200, PRINTFPORT);
	delay_ms(60);
	printf("this is SD FAT test \r\n");

//SD卡电源开
	printf("SD poweron. \r\n");		 
 	SD.ON();//在上完电要延时一点点时间在初始化SD卡，要不然某些SD卡初始化不成功

//SD初始化，成功的话也初始化一下FAT	
	printf("SD init. \r\n");    
	SD.init();
	if(SD.flag!=NOTHING_FAILED)
	{
		printf("  failed!  ");
		while(1);
	}

//得到卡的实际能使用的所有空间
	printf("Get disk fat size. \t");
	fsize=SD.getDiskSize();
	if(SD.flag==FR_OK)
	{
		printf(" fatsize=%d KB \r\n",(uint32_t)(fsize>>10));
	}
	else
	{
		printf(" errflag=%x ",SD.flag);
		while(1);
	}

//得到卡的实际能使用的剩余空间	
	fsize=SD.getDiskFree();
	printf("Get free fat size. \t");
	if(SD.flag==FR_OK)
	{
		printf("freesize=%d KB \r\n",(uint32_t)(fsize>>10));
	}
	else
	{
		printf(" errflag=%x ",SD.flag);
		while(1);
	}

//打印SD卡信息
	printf("Get SD information. \t");
	SD.print_disk_info();
	printf("%s\r\n",SD.buffer);


//根目录下有没有dir0的文件夹
//如果有则删除，生成一个新的文件夹
	printf("Does the folder dir0 exist? \r\n");
	if(SD.isDir("dir0")==1)
	{
		printf("  yes.");
		printf(" Prepare use fuction del to delete the folder dir0,");
		if(SD.del("dir0"))//这个当文件夹里面有东西就删除不了了
		{
			printf("  del success. ");
		}
		else
		{
			printf(" errflag=%d.",SD.flag);
			printf(" When dir0 have subfolders or files, it doesnt work if we use the fuction del. We should use deldirall. ");
			printf(" Prepare use fuction deldirall to delete the folder dir0, ");
			if(SD.deldirall("dir0")==1)//彻底删除
			{
				printf("delall success. ");		
			}
			else 
			{
				printf(" delall err, there are something wrong! please tell author.\n ");
				while(1);
			}
		}
	}
	else
	{	
		printf("no.");
	}


	printf("\r\nDoes the folder dir0 exist? \r\n");
	if(SD.isDir("dir0")==1)
	{
		printf(" yes. ");
		printf(" There are something WRONG! please tell author.\n ");
		while(1);
	}
	else
	{
		printf("  There is no folder dir0, now create the folder. ");
		if(SD.create("dir0"))
		{
			printf(" success ");
		}
		else
		{
			printf(" failed  ");
			while(1);
		}

	}

//读取此目录下多少个文件
	printf("\r\nGet the number of files in the folder dir0, it should be 0. \r\n");
	filecnt=SD.numFiles("dir0");
	if(filecnt<0)
	{
		printf(" err, filecnt=%d ",filecnt);
		while(1);
	}
	else
	{
		printf("filecnt=%d ",filecnt);
		if(filecnt!=0)
		{
			printf(" WRONG! ");
			while(1);
		}	
	}


//生成若干文件或者文件夹
	printf("\r\nCreate some files and subfolders. \r\n");
	SD.create("dir0/file1.txt");
	SD.create("dir0/file2.txt");
	SD.create("dir0/file3.txt");
	SD.create("dir0/file7.txt");

 	SD.create("dir0/04");
	SD.create("dir0/04/004");
	SD.create("dir0/04/004/0004");
	SD.create("dir0/04/004/file0004.txt");

	SD.mkdir("dir0/05");
	SD.create("dir0/06");

//检查生成的dir0/05文件夹存不存在
	printf("\r\nDoes the folder dir0/05 exist?\r\n");
	if(SD.isDir("dir0/05")==1)
	{
		printf(" yes. ");

	}
	else
	{
			printf(" WRONG! ");
			while(1);
	}

//当使用了删除文件夹命令，再次检查文件夹dir0/05存不存在
	printf("\r\nUse the function delDir to delete dir0/05. \r\n");
	SD.delDir("dir0/05");
	printf(" Check if the dir0/05 exist. ");
	if(SD.isDir("dir0/05")==1)
	{
		printf(" exist. ");
		printf(" There are something WRONG! please tell author.\n ");
		while(1);
	}
	else
	{
			printf(" no. ");
	}
		
//检查生成的dir0/file7.txt文件存不存在
	printf("\r\nDoes the file dir0/file7.txt exist?\r\n");
	if(SD.isFile("dir0/file7.txt")==1)
	{
		printf(" yes. ");

	}
	else
	{
		printf(" no. ");
		printf(" There are something WRONG! please tell author.\n ");
		while(1);
	}


//得到当前文件夹dir0里的文件个数，按道理是4个
	printf("\r\nGet the number of files in the folder dir0, it should be 4(file1.txt,file2.txt,file3.txt,file7.txt). \r\n");
	filecnt=SD.numFiles("dir0");
	printf("  filecnt=%d. ",filecnt);
	if(filecnt!=4)
	{
			printf(" WRONG! ");
			while(1);				
	}

//删除文件
	printf("\r\nDelete the file file7.txt in the folder dir0\r\n");
	SD.del("dir0/file7.txt");

	printf("\r\n  Get the number of files in the folder dir0, it should be 3(file1.txt,file2.txt,file3.txt). \r\n");
	filecnt=SD.numFiles("dir0");
	printf("  filecnt=%d. ",filecnt);
	if(filecnt!=3)
	{
			printf(" WRONG! ");
			while(1);				
	}

//写入数据到文件里面
	printf("\r\nWrite some datas in dir0/file1.txt.\r\n");
	SD.writeSD("dir0/file1.txt", "I am a frog, do you like me? My\ngirl\nis\na\ngreat\nwoman\ndont\nshe?\nyes,\nshe\nis !!!!", 0);

//得到文件dir0/file1.txt的大小
	printf("\r\nGet the size of dir0/file1.txt. \r\n");
	filesize=SD.getFileSize("dir0/file1.txt");
	if(filesize>=0)
	{
		printf("filesize=%d ",filesize);
	}
	else
	{
			printf(" WRONG! ");
			while(1);	
	}


//把文件里面的数据读出来	
	printf("\r\nGet 20 datas in dir0/file1.txt and printf them.\r\n");
	SD.readSD("dir0/file1.txt", readstr1, 0,20);
	printf("<%s>",readstr1);

	printf(" Get 20 datas in dir0/file1.txt from offset=10 and printf them.");
	SD.cat("dir0/file1.txt", 10,20);
	printf("<%s>",SD.buffer);

	printf(" Get 20 datas in dir0/file1.txt from offset=20 and printf them.");
	SD.catBin("dir0/file1.txt", 20,20);
	printf("<%s>",SD.bufferBin);

//把文件里面的数据读出来	
	readstr1[0]=0x00;
 	printf("\r\nGet 20 datas in dir0/file2.txt printf them. them should nothing.\r\n");
	SD.readSD("dir0/file2.txt", readstr1, 0,20);
	printf("<%s>",readstr1);

//按行读取文件里面的数据 
	printf("\r\nGet 2 packet datas in dir0/file1.txt from offset=2 and printf them. Notice: the last data of each packet datas is enter(0x13).\r\n");
	SD.catln ("dir0/file1.txt", 1, 4);
	printf("<%s>",SD.buffer);
	
	printf("\r\n OK end");
}

void loop()
{
	while(1);
}
#endif // end SDIOFAT




