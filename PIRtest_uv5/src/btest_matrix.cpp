/*
这个是板子的I2C例程,格式和maple的接近，使用setup()和loop()两个函数，区别的是这里必须调用头文件	#include "allboardinc.h"
例程有如下几种：   
#define BTESTMATRIXSIMPLE 0		 简单测试建立MATRIX时间什么的

#define BTESTMATRIXNULL 100	 不选用MATRIX例程
不管选择哪个，程序都是有setup()和loop()两部分，当选择不选用功耗例程就没有这两个函数。要保证在整个工程里面只有一个setup()和loop()
*/
#include "allboardinc.h"


#include "arm_math.h" 
//#include "math_helper.h" 


#define BTESTMATRIXSIMPLE 0        //
#define BTESTMATRIXSIMPLE2 2        //20130329测试普通乘法和加法
#define BTESTMATRIXSIMPLE3 3        ////用dsp用乘法 加法 求逆 转置
#define BTESTMATRIXSIMPLE4 4        // 自己编的加乘 转置，求逆，基于float类型
#define BTESTMATRIXSIMPLE5 5        // 自己编的加乘 转置，求逆，基于uint16_t类型


#define BTESTMATRIXNULL 100  //	 

#define EXAMPLEMATRIX BTESTMATRIXNULL
//


#define MATRIXMUL 1
#define MATRIXADD 1
#define MATRIXTRANS 0
#define MATRIXINV 0


#if EXAMPLEMATRIX==BTESTMATRIXSIMPLE3 //用dsp用乘法 加法 求逆 转置

#define SNR_THRESHOLD 	90 
 
float32_t snr; 
char TimeStamp1[50];
char TimeStamp2[50];
void printmatrix(arm_matrix_instance_f32 A){
	printf("row:%d column:%d ",A.numRows,A.numCols);
	for(uint16_t i=0;i<A.numRows*A.numCols;i++)
		printf("%f ",(*(A.pData+i))); //(float)
}


#define CISHUMAX 6
void setup()
{

	arm_matrix_instance_f32 A;		/* Matrix A Instance */ 
	arm_matrix_instance_f32 AT;		/* Matrix AT(A transpose) instance */  
	arm_matrix_instance_f32 B;		/* Matrix B instance */ 
	arm_matrix_instance_f32 C;		

	 
	uint32_t srcRows, srcColumns;	/* Temporary variables */
	arm_status status; 
	uint32_t times=0;
	uint8_t  cishu=0;
#if MATRIXADD==1
	uint16_t shijianchaadd[CISHUMAX];
#endif
#if MATRIXTRANS==1
	uint16_t shijianchatran[CISHUMAX];
#endif
#if MATRIXMUL==1
	uint16_t shijianchamul[CISHUMAX];
#endif
#if MATRIXINV==1
	uint16_t shijianchainv[CISHUMAX];
#endif

	uint32_t shijianqian;   // shijian = time, qian = before
	uint32_t shijianhou;    // hou= after
	uint8_t  juzhensize[CISHUMAX]; //juzhen = matrix
	Mux_poweron(); //为打印串口弄一个VCC
	Utils.initLEDs();//调用灯之前必须初始化一下
	for(int i=0;i<5;i++)
		Utils.blinkLEDs(100);
	monitor_onuart3TX();
	beginSerial(115200, PRINTFPORT);
	delay_ms(60);
	printf("this is martix test \r\n\r\n\r\n\r\n");
	RTCbianliang.ON();
	RTCbianliang.begin();



	// Initialise A Matrix Instance with numRows, numCols and data array(A_f32) // 
//	srcRows = 50; 
//    srcColumns = 50; 

	float32_t *f32a ;
	float32_t *f32b ;
	float32_t *f32c ;

	for(cishu=0;cishu<CISHUMAX;cishu++)
	{		
		if(cishu==0){srcRows = 5; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
		else  if(cishu==1){srcRows = 10; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
		else  if(cishu==2){srcRows = 20; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
		else  if(cishu==3){srcRows = 30; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
		else  if(cishu==4){srcRows = 40; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
		else  if(cishu==5){srcRows = 50; srcColumns = srcRows;juzhensize[cishu]=srcRows;}

		//矩阵的建立与输出
		printf("\r\n\r\n\r\n create matrix row=%d col=%d ",srcRows,srcColumns);		
		f32a = (float32_t *)malloc(sizeof(float32_t)*srcRows*srcColumns);
		if(f32a==NULL){
		 	printf(" no enough room for f32a ");
			while(1);
		}
		for(uint32_t i=0;i<srcRows*srcColumns;i++){
			*(f32a+i)=0.0;
		}
		*f32a=2.0;
		arm_mat_init_f32(&A, srcRows, srcColumns, (float32_t *)f32a);
		
	
		
		f32b = (float32_t *)malloc(sizeof(float32_t)*srcRows*srcColumns);
		if(f32b==NULL){
		 	printf(" no enough room for f32b ");
			free(f32a);
			while(1);
		}
		for(uint32_t i=0;i<srcRows*srcColumns;i++){
			*(f32b+i)=0.0;
		}
		*f32b=3.0;
		arm_mat_init_f32(&B, srcRows, srcColumns, (float32_t *)f32b);
		
		
		f32c = (float32_t *)malloc(sizeof(float32_t)*srcRows*srcColumns);
		if(f32c==NULL){
		 	printf(" no enough room for f32c ");
			free(f32a);free(f32b);
			while(1);
		}
		for(uint32_t i=0;i<srcRows*srcColumns;i++){
			*(f32c+i)=0.0;
		}
		*f32c=4.0;
		arm_mat_init_f32(&C, srcRows, srcColumns, (float32_t *)f32c);
			 
	 
		for(int i=0;i<5;i++)
			Utils.blinkLEDs(100);
		//printf("\r\n printf original data:");
		printf("\r\nA : ");printmatrix(A);
		printf("\r\nB : ");printmatrix(B);
		printf("\r\nC : ");printmatrix(C);
	
	
#if MATRIXMUL==1
		times=1000;
		for(uint32_t i=0;i<srcColumns;i++){		
			*(f32a+i+i*srcRows)=1.0;
			*(f32b+i+i*srcRows)=2.0;
		}
		printf("\r\n\r\n prepare multi %d times:",times*100);
		RTCbianliang.getTime();
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		for(uint32_t k=0;k<times;k++){
			for(uint32_t i=0;i<srcColumns;i++){		
				*(f32a+i+i*srcRows)=1.0;
				*(f32b+i+i*srcRows)=2.0;
			}		
			for(uint32_t i=0;i<100;i++)
				status = arm_mat_mult_f32(&A, &B, &A);
		}
	
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n%s  ",TimeStamp1);
		printf("%s",TimeStamp2);
	
		printf("\r\nA  ");printmatrix(A);
		printf("\r\nB  ");printmatrix(B);
		//printf("\r\nC  ");printmatrix(C);
		printf("\r\nbefore multi %s  ",TimeStamp1);
		printf(" after multi %s",TimeStamp2);
		shijianchamul[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchamul[cishu]);
#endif

	
#if MATRIXADD==1	
		times=100000;
		printf("\r\n\r\n  prepare add %dtimes:",times);
		RTCbianliang.getTime();
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		//for(int k=0;k<10;k++){
			for(uint32_t i=0;i<srcColumns;i++){		
				*(f32a+i+i*srcRows)=1.0;
				*(f32b+i+i*srcRows)=2.0;
			}		
			for(uint32_t i=0;i<times;i++)
				status = arm_mat_add_f32(&A, &B, &A);
		//}
	
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n%s  ",TimeStamp1);
		printf("%s",TimeStamp2);	
		printf("\r\nA : ");printmatrix(A);
		printf("\r\nB : ");printmatrix(B);
		//printf("\r\nC : ");printmatrix(C);
		printf("\r\nbefore add %s  ",TimeStamp1);
		printf(" after add %s",TimeStamp2);
		shijianchaadd[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchaadd[cishu]);
#endif	
	


#if MATRIXTRANS==1
		times=100002;
		printf("\r\n\r\n  prepare transpose %dtimes :",times);
		RTCbianliang.getTime();
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
			for(uint32_t i=0;i<srcColumns;i++){		
				*(f32a+i)=1.0;
			}
		for(uint32_t i=0;i<times;i++)
			status = arm_mat_trans_f32(&A, &C);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n%s  ",TimeStamp1);
		printf("%s",TimeStamp2);
		printf("\r\nA : ");printmatrix(A);
		//printf("\r\nB : ");printmatrix(B);
		printf("\r\nC : ");printmatrix(C);
		printf("\r\n before transpose %s  ",TimeStamp1);
		printf(" after transpose %s",TimeStamp2);
		shijianchatran[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchatran[cishu]);
#endif	
	

#if MATRIXINV==1	
		for(uint32_t  k=0;k<srcColumns;k++){
			for(uint32_t i=0;i<srcRows;i++){		
				*(f32a+i+k*srcRows)=0.0;
			}		
		}
		times=50000;
		printf("\r\n\r\n  prepare inverse %dtime:",times*2);
	
		for(uint32_t i=0;i<srcColumns;i++){		
			*(f32a+i+i*srcRows)=1.0;
		}
		*(f32a)=2.0;  //*(f32a+1)=3.0; *(f32a+4)=4.0;
		printf("\r\n printf A and C before inverse");
		printf("\r\nA : ");printmatrix(A);
		printf("\r\nC : ");printmatrix(C);
	
		RTCbianliang.getTime();
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		for(uint32_t i=0;i<times;i++)
		{
			status = arm_mat_inverse_f32(&A, &C);
			status = arm_mat_inverse_f32(&C, &A);
		}

		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n%s  ",TimeStamp1);
		printf("%s",TimeStamp2);
		printf("\r\nA : ");printmatrix(A);
		//printf("\r\nB : ");printmatrix(B);
		printf("\r\nC : ");printmatrix(C);
		printf("\r\nbefore inverse %s  ",TimeStamp1);
		printf(" after inverse %s",TimeStamp2);
		shijianchainv[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchainv[cishu]);
#endif

	
		free(f32a);free(f32b);free(f32c);//free(f32b);
		printf(" end ");




	}


	for(cishu=0;cishu<CISHUMAX;cishu++)
	{		
		printf("\r\nrow=%d ",juzhensize[cishu]);
#if MATRIXADD==1				
		printf("\tadd 100000times =%ds \r\n",shijianchaadd[cishu]);
#endif

#if MATRIXTRANS==1
		printf("\ttrans 100000times =%ds \r\n",shijianchatran[cishu]);
#endif

#if MATRIXMUL==1
		printf("\tmul 100000times =%ds \r\n",shijianchamul[cishu]);
#endif
#if MATRIXINV==1
		printf("\tinv 100000times =%ds \r\n",shijianchainv[cishu]);	
#endif				
	}

	printf("\r\n END!!! ");
	 
	/* ---------------------------------------------------------------------- 
	** Loop here if the signals fail the PASS check. 
	** This denotes a test failure 
	** ------------------------------------------------------------------- */	 
	if( status != ARM_MATH_SUCCESS) 
	{ 
	  while(1); 
	} 
}

void loop()
{
	while(1); 	
}
#endif //end BTESTMATRIXSIMPLE



#if EXAMPLEMATRIX==BTESTMATRIXSIMPLE2
char TimeStamp1[50];
char TimeStamp2[50];
const int TestArray[]={
 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
10,11,12,13,14,15,16,17,18,19,
20,21,22,23,24,25,26,27,28,29,
30,31,32,33,34,35,36,37,38,39,
40,};
 

#define MATRIXSIZE 50
#define TIMES 1000
void setup()
{
	int dataint[] = {1,2,3,4,5,6,7,8};
	double datadb[] = {1.00,2.567,3.8043569,4.754,5.890,6.809,7.7,8.8};
	char datachar[] = {1,2,3,4,5,6,7,8};
 	long datalong[] = {1,2,3,4,5,6,7,8};
//	CMat *matint = NULL;
	CMat *matchar = NULL;
	CMat *matlong = NULL; 
	CMat *matdb = NULL;
	CMat *matmul = NULL;
	int  *t = NULL;

	CMat *matint1 = NULL;
	CMat *matint2 = NULL;
	CMat *matint3 = NULL;


	uint16_t shijianchaadd;
//	uint16_t shijianchatran;
	uint16_t shijianchamul;
//	uint16_t shijianchainv;
	uint32_t shijianqian;
	uint32_t shijianhou;

	monitor_on();//打开usb_232电源

	for(int i=0;i<5;i++)
		Utils.blinkLEDs(100);

	beginSerial(9600, PRINTFPORT);
	delay_ms(60);
	printf("this is martix test ");
	RTCbianliang.ON();
	RTCbianliang.begin();

	//矩阵的建立与输出
	printf("\r\n create matrix\r\n");
	RTCbianliang.getTime();
	//printf("%s",RTCbianliang.timeStamp);
	strcpy(TimeStamp1,RTCbianliang.timeStamp);
	//printf("%s",TimeStamp1);	
	//// //MAT_DATA_TYPE_CHAR
	matint1 = CreateMat(MATRIXSIZE,MATRIXSIZE,MAT_DATA_TYPE_INT,NULL);
	matint2 = CreateMat(MATRIXSIZE,MATRIXSIZE,MAT_DATA_TYPE_INT,NULL);
	matint3 = CreateMat(MATRIXSIZE,MATRIXSIZE,MAT_DATA_TYPE_INT,NULL);

	RTCbianliang.getTime();
	//printf("%s",RTCbianliang.timeStamp);
	strcpy(TimeStamp2,RTCbianliang.timeStamp);
	printf("%s",TimeStamp1);
	printf("%s",TimeStamp2);
	//PrintMat(matint1);

//	matint = CreateMat(4,2,MAT_DATA_TYPE_INT,dataint);
//	PrintMat(matint);
//
//	matdb = CreateMat(2,4,MAT_DATA_TYPE_DOUBLE,NULL);
//	PrintMat(matdb);
//	matchar = CreateMat(4,2,MAT_DATA_TYPE_UCHAR,datachar);
//	PrintMat(matchar);
//    //数据的设置
//	//printf("\n数据的设置 \n");
//	printf("\r\n set matrix\r\n");
//    SetData2D(matdb,0,0,4,MAT_DATA_TYPE_NULL);
//	PrintMat(matdb);
//	SetData2D(matchar,0,0,4,MAT_DATA_TYPE_NULL);
//	PrintMat(matchar);
//	//矩阵的加法

	SetData2D(matint1,0,0,1,MAT_DATA_TYPE_NULL);
	SetData2D(matint1,0,1,2,MAT_DATA_TYPE_NULL);
	SetData2D(matint1,0,2,3,MAT_DATA_TYPE_NULL);
	SetData2D(matint1,0,3,4,MAT_DATA_TYPE_NULL);

	SetData2D(matint2,0,0,2,MAT_DATA_TYPE_NULL);
	SetData2D(matint2,0,1,3,MAT_DATA_TYPE_NULL);
	SetData2D(matint2,0,2,4,MAT_DATA_TYPE_NULL);
	SetData2D(matint2,0,4,5,MAT_DATA_TYPE_NULL);
//	SetData2D(matint2,1,0,1,MAT_DATA_TYPE_NULL);
//	SetData2D(matint2,0,0,10,MAT_DATA_TYPE_NULL);
//	SetData2D(matint2,0,1,20,MAT_DATA_TYPE_NULL);
//	SetData2D(matint2,0,2,30,MAT_DATA_TYPE_NULL);
//	SetData2D(matint2,0,3,10,MAT_DATA_TYPE_NULL);

	//printf("matin1:\r\n");
	//PrintMat(matint1);
	//printf("matin2:\r\n");
	//PrintMat(matint2);

#if MATRIXADD==1
	printf("\r\n add matrix\r\n");
	delay_ms(1000);
	RTCbianliang.getTime();
	shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
	strcpy(TimeStamp1,RTCbianliang.timeStamp);
	//空间已经在外部开辟
	for(uint32_t i=0;i<TIMES;i++)//一万次加这个50*50矩阵82秒
	{
		AddMat(matint3,matint1,matint2);
	} 
	RTCbianliang.getTime();
	shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
	strcpy(TimeStamp2,RTCbianliang.timeStamp);
	printf("%s",TimeStamp1);
	printf("%s",TimeStamp2);
	PrintMat(matint3);
	shijianchaadd=(uint16_t)(shijianhou -shijianqian);
	printf("cha=%d \r\n",shijianchaadd);
#endif

#if MATRIXTRANS==1
	printf("\r\n transpose matrix\r\n");
	printf("1:");
	PrintMat(matint1);
	TransposeMat(matint1,matint3);
	printf("1:");
	PrintMat(matint1);
	printf("3:");
	PrintMat(matint3);
#endif

#if MATRIXMUL==1
	printf("\r\n mul matrix\r\n");
	delay_ms(1000);
	RTCbianliang.getTime();
	shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
	strcpy(TimeStamp1,RTCbianliang.timeStamp);
	for(uint32_t i=0;i<TIMES;i++)//乘100次 这个50*50矩阵28秒
		MulMat(matint3,matint1,matint2); 
	RTCbianliang.getTime();
	shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
	strcpy(TimeStamp2,RTCbianliang.timeStamp);
	printf("%s",TimeStamp1);
	printf("%s",TimeStamp2);
	PrintMat(matint3);
	printf("end");
	shijianchamul = (uint16_t)(shijianhou -shijianqian);
	printf("cha=%d \r\n",shijianchamul);
#endif

	printf("\r\nsize =%d \r\n",MATRIXSIZE);

#if MATRIXADD==1
	printf("\r\nadd %dtimes =%ds \r\n",TIMES,shijianchaadd);
#endif
#if MATRIXMUL==1
	printf("\r\namul %dtimes =%ds \r\n",TIMES,shijianchamul);
#endif
//	PrintMat(matint);
//    matlong = AddMat(matlong,matint,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
//	PrintMat(matlong);
//	//printf("\n矩阵减法 \n"); ///
//	//矩阵减法
//	printf("\r\n sub matrix\r\n");
//	SubMat(matlong,matlong,matint); //空间已经在外部开辟
//	PrintMat(matlong);
//    SubMat(matlong,matlong,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
//	PrintMat(matlong);
//
//	//printf("\n矩阵乘法 \n");
//	//矩阵减法
//	printf("\r\n time matrix\r\n");
//    matint  =  CreateMat(4,2,MAT_DATA_TYPE_INT,dataint);
//	PrintMat(matint);
//    matchar = CreateMat(2,4,MAT_DATA_TYPE_UCHAR,datachar);
//	PrintMat(matchar);
//    matmul =  CreateMat(2,2,MAT_DATA_TYPE_DOUBLE,NULL);
//	matmul = MulMat(matmul,matint,matchar); 
//	PrintMat(matmul);
//   	// MulMat(matlong,matlong,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
//   	// PrintMat(matlong);
//	//数据类型转换
//    //*t = 12.568;
//	//w = (int*)DataTypeTo(t,MAT_DATA_TYPE_INT);
//   	//printf("数据类型转换%d \n",*w);
//	//改变 x y
//	//ReshapeMat(matlong,1,8);
// 	//PrintMat(matlong);
}

void loop()
{
	while(1); 	
}
#endif //end BTESTMATRIXSIMPLE

#if EXAMPLEMATRIX==BTESTMATRIXSIMPLE
void setup()
{
	int dataint[] = {1,2,3,4,5,6,7,8};
	double datadb[] = {1.00,2.567,3.8043569,4.754,5.890,6.809,7.7,8.8};
	char datachar[] = {1,2,3,4,5,6,7,8};
 	long datalong[] = {1,2,3,4,5,6,7,8};
	CMat *matint = NULL;
	CMat *matchar = NULL;
	CMat *matlong = NULL; 
	CMat *matdb = NULL;
	CMat *matmul = NULL;
	int  *t = NULL;

	for(int i=0;i<5;i++)
		Utils.blinkLEDs(100);
  
  monitor_onuart3TX();
	beginSerial(115200, PRINTFPORT);
	delay_ms(60);
	printf("this is martix test ");

    //矩阵的建立与输出
	//printf("\n矩阵的建立与输出 \n");
	printf("\r\n create matrix\r\n");
	matint = CreateMat(2,4,MAT_DATA_TYPE_INT,NULL);
	PrintMat(matint);
	matint = CreateMat(4,2,MAT_DATA_TYPE_INT,dataint);
	PrintMat(matint);

	matdb = CreateMat(2,4,MAT_DATA_TYPE_DOUBLE,NULL);
	PrintMat(matdb);
	matchar = CreateMat(4,2,MAT_DATA_TYPE_UCHAR,datachar);
	PrintMat(matchar);
    //数据的设置
	//printf("\n数据的设置 \n");
	printf("\r\n set matrix\r\n");
    SetData2D(matdb,0,0,4,MAT_DATA_TYPE_NULL);
	PrintMat(matdb);
	SetData2D(matchar,0,0,4,MAT_DATA_TYPE_NULL);
	PrintMat(matchar);
	//矩阵的加法
	printf("\r\n add matrix\r\n");
	//printf("\n矩阵加法 \n");
	AddMat(matint,matint,matint); //空间已经在外部开辟
	PrintMat(matint);
    matlong = AddMat(matlong,matint,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
	PrintMat(matlong);
	//printf("\n矩阵减法 \n"); ///
	//矩阵减法
	printf("\r\n sub matrix\r\n");
	SubMat(matlong,matlong,matint); //空间已经在外部开辟
	PrintMat(matlong);
    SubMat(matlong,matlong,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
	PrintMat(matlong);

	//printf("\n矩阵乘法 \n");
	//矩阵减法
	printf("\r\n time matrix\r\n");
    matint  =  CreateMat(4,2,MAT_DATA_TYPE_INT,dataint);
	PrintMat(matint);
    matchar = CreateMat(2,4,MAT_DATA_TYPE_UCHAR,datachar);
	PrintMat(matchar);
    matmul =  CreateMat(2,2,MAT_DATA_TYPE_DOUBLE,NULL);
	matmul = MulMat(matmul,matint,matchar); 
	PrintMat(matmul);
   // MulMat(matlong,matlong,matint); //矩阵没有建立，需要在内部建立,在初始化的时候必须设置指针为NULL
   // PrintMat(matlong);
	//数据类型转换
    //*t = 12.568;
//	w = (int*)DataTypeTo(t,MAT_DATA_TYPE_INT);
   // printf("数据类型转换%d \n",*w);
	//改变 x y
//	ReshapeMat(matlong,1,8);
 //   PrintMat(matlong);
}

void loop()
{
	while(1); 	
}
#endif //end BTESTMATRIXSIMPLE



#if EXAMPLEMATRIX==BTESTMATRIXSIMPLE4
void printmatrix(float *a, uint16_t row, uint16_t col){
	printf("row:%d column:%d \r\n",row,col);
	for(uint16_t i=0;i<row;i++)
	{
	   for(uint16_t k=0;k<col;k++)
		  printf("%f ",(*(a+i*col+k)));
		printf("\r\n");
	}
	printf("\r\n");
		 //(float)
}
char TimeStamp1[50];
char TimeStamp2[50];
#define CISHUMAX 6

void setup()
{
//	int i,k;
//	int hangliek=10;
	float *a;
	float *b;
	float *c; 
	uint16_t srcRows, srcColumns;
#if MATRIXINV==1
	int flag=0;
#endif
	uint8_t cishu=0;
	uint32_t times;

#if MATRIXADD==1
	uint16_t shijianchaadd[CISHUMAX];
#endif
#if MATRIXTRANS==1
	uint16_t shijianchatran[CISHUMAX];
#endif
#if MATRIXMUL==1
	uint16_t shijianchamul[CISHUMAX];
#endif
#if MATRIXINV==1
	uint16_t shijianchainv[CISHUMAX];
#endif

	uint16_t  juzhensize[CISHUMAX];
	uint32_t shijianqian;
	uint32_t shijianhou;



monitor_on(); //为打印串口弄一个VCC

	for(int i=0;i<5;i++)
		Utils.blinkLEDs(100);

	RTCbianliang.ON();
	RTCbianliang.begin();
	beginSerial(9600, PRINTFPORT);
	delay_ms(60);
	printf("this is martix test ");

    //矩阵的建立与输出
	//printf("\n矩阵的建立与输出 \n");
	printf("\r\n create matrix\r\n");
		printf("add transpose mul inv in my way with float, k=10 20 30 40 50 \r\n");

	for(cishu=0;cishu<CISHUMAX;cishu++)
	{		
		if(cishu==0){srcRows = 5; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
//		else  if(cishu==1){srcRows = 10; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
//		else  if(cishu==2){srcRows = 20; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
//		else  if(cishu==3){srcRows = 30; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
//		else  if(cishu==4){srcRows = 40; srcColumns = srcRows;juzhensize[cishu]=srcRows;}
//		else  if(cishu==5){srcRows = 50; srcColumns = srcRows;juzhensize[cishu]=srcRows;}

		else{
			srcRows = cishu*10; srcColumns = srcRows;juzhensize[cishu]=srcRows;
		}

	
	    a = (float *)malloc(sizeof(float)*srcRows*srcColumns);
	    if(a==NULL)
	    {
	      printf(" no enough room for a ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(a+i)=1.0;
	    }
	   	for(uint16_t i=0;i<srcRows;i++){ 
	     *(a+i*srcRows+i)=2.0;
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(a+i)=(float)i;
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(a+i*srcRows)=(float)(2*i);
		}				   
	 
	    b = (float *)malloc(sizeof(float)*srcRows*srcColumns);
	    if(b==NULL)
	    {
	      printf(" no enough room for b ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(b+i)=4.0;
	    }
	    for(uint16_t i=0;i<srcRows;i++){ 
	     *(b+i*srcRows+i)=3.0;
	    }
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(b+i)=(float)(i+1);
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(b+i*srcRows)=(float)(3*i);
		}		    
	    

	 
	    c = (float *)malloc(sizeof(float)*srcRows*srcColumns);
	    if(c==NULL)
	    {
	      printf(" no enough room for c ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(c+i)=0.0;
	    }
	    *c=4.0;
		
		printf("\r\n a:"); printmatrix(a, srcRows, srcColumns);
		printf("\r\n b:"); printmatrix(b, srcRows, srcColumns);	  
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
	

#if MATRIXADD==1
		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;		
		times=100000;
		for(uint32_t m=0; m<times;m++)
			zppaddmatrix(srcRows,srcColumns,a,c,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n %s  ",TimeStamp1);
		printf("%s",TimeStamp2);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		printf(" before add %s  ",TimeStamp1);
		printf("after add %s\r\n",TimeStamp2);
		shijianchaadd[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchaadd[cishu]);
#endif	

#if MATRIXTRANS==1
		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)
			zpptransposematrix(srcRows,srcColumns,a,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n %s  ",TimeStamp1);
		printf("%s",TimeStamp2);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		printf(" before transpose %s  ",TimeStamp1);
		printf("after transpose %s\r\n",TimeStamp2);
		shijianchatran[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchatran[cishu]);
#endif

#if MATRIXMUL==1

	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(a+i)=101.2;
	    }
	   	for(uint16_t i=0;i<srcRows;i++){ 
	     *(a+i*srcRows+i)=2.0;
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(a+i)=(float)i;
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(a+i*srcRows)=(float)(2*i);
		}	


	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(b+i)=104.1;
	    }
	    for(uint16_t i=0;i<srcRows;i++){ 
	     *(b+i*srcRows+i)=3.0;
	    }
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(b+i)=(float)(i+1);
		}
	   	for(uint16_t i=1;i<srcRows;i++){ 
	     *(b+i*srcRows)=(float)(3*i);
		}


		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)			
			zppmulmatrix(srcRows,srcColumns,a,b,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n %s  ",TimeStamp1);
		printf("%s",TimeStamp2);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		printf(" before mul %s  ",TimeStamp1);
		printf("after mul %s\r\n",TimeStamp2);
		shijianchamul[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchamul[cishu]);
#endif		

#if MATRIXINV==1
		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)					
			flag=zppinversematrix(a,srcRows,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		printf("\r\n %s  ",TimeStamp1);
		printf("%s",TimeStamp2);

		if(flag==0)//说明有逆矩阵
		{
			printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
			printf(" before inv %s  ",TimeStamp1);
			printf("after inv %s\r\n",TimeStamp2);
			shijianchainv[cishu]=(uint16_t)(shijianhou -shijianqian);
			printf("cha=%d \r\n",shijianchainv[cishu]);				
		}
		else
		{
			printf("\r\n err=%d ",flag);
			shijianchainv[cishu]=0;
			printf("cha=%d \r\n",shijianchainv[cishu]);
		}
#endif
		printf("end");
	}
	printf("add transpose mul inv in my way with float\n");
	for(cishu=0;cishu<CISHUMAX;cishu++)
	{	

		printf("\r\nrow=%c ",juzhensize[cishu]);//

#if MATRIXADD==1				
		printf("\tadd 100000times =%ds \r\n",shijianchaadd[cishu]);
#endif

#if MATRIXTRANS==1
		printf("\ttrans 100000times =%ds \r\n",shijianchatran[cishu]);
#endif

#if MATRIXMUL==1
		printf("\tmul 100000times =%ds \r\n",shijianchamul[cishu]);
#endif
#if MATRIXINV==1
		printf("\tinv 100000times =%ds \r\n",shijianchainv[cishu]);	
#endif
		
				
	}
	
	printf("END");
}

void loop()
{
	while(1); 	
}
#endif //end BTESTMATRIXSIMPLE4


#if EXAMPLEMATRIX==BTESTMATRIXSIMPLE5
void printmatrix(int16_t *a, uint16_t row, uint16_t col){
	printf("row:%d column:%d \r\n",row,col);
	for(uint16_t i=0;i<row;i++)
	{
		//printf(" tt ");
		for(uint16_t k=0;k<col;k++)
			printf("%d ",(*(a+i*col+k)));
		printf("\r\n");
	}
	//printf("\r\n");
		 //(float)
}
char TimeStamp1[50];
char TimeStamp2[50];
void setup()
{
//	int i,k;
//	int hangliek=10;
	int16_t *a;
	int16_t  *b;
	int16_t  *c; 
	int16_t srcRows, srcColumns;
	//int flag=0;
	uint8_t cishu=0;
	uint32_t times;

	uint16_t shijianchaadd[5];
	uint16_t shijianchatran[5];
	uint16_t shijianchamul[5];
	uint32_t shijianqian;
	uint32_t shijianhou;

	for(int i=0;i<5;i++)
		Utils.blinkLEDs(100);

	RTCbianliang.ON();
	RTCbianliang.begin();
	beginSerial(9600, PRINTFPORT);
	delay_ms(60);
	printf("this is martix test ");

    //矩阵的建立与输出
	//printf("\n矩阵的建立与输出 \n");
	printf("\r\n create matrix\r\n");
	printf("add transpose mul in my way with uint16_t, k=10 20 30 40 50 \r\n");

	for(cishu=0;cishu<5;cishu++)
	{
		if(cishu==0){srcRows = 10; srcColumns = srcRows;}
		else if(cishu==1){srcRows = 20; srcColumns = srcRows;}
		else if(cishu==2){srcRows = 30; srcColumns = srcRows;}
		else if(cishu==3){srcRows = 40; srcColumns = srcRows;}
		else if(cishu==4){srcRows = 50; srcColumns = srcRows;}

	
	    a = (int16_t  *)malloc(sizeof(int16_t )*srcRows*srcColumns);
	    if(a==NULL)
	    {
	      printf(" no enough room for a ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(a+i)=0.0;
	    }
	   	for(uint16_t i=0;i<srcRows;i++){ 
	     *(a+i*srcRows+i)=2.0;
	  }   
	    *(a+1)=1.0;*(a+2)=1.0;
	 
	    b = (int16_t  *)malloc(sizeof(int16_t )*srcRows*srcColumns);
	    if(b==NULL)
	    {
	      printf(" no enough room for b ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(b+i)=0;
	    }
	    for(uint16_t i=0;i<srcRows;i++){ 
	     *(b+i*srcRows+i)=3.0;
	    }    
	    *b=3.0; *(b+1)=10.0;*(b+2)=11.0;
	 
	    c = (int16_t  *)malloc(sizeof(int16_t )*srcRows*srcColumns);
	    if(c==NULL)
	    {
	      printf(" no enough room for c ");
	      while(1);
	    }
	    for(uint16_t i=0;i<srcRows*srcColumns;i++){
	      *(c+i)=0;
	    }
	    *c=4;
		
		printf("\r\n a:"); printmatrix(a, srcRows, srcColumns);
		printf("\r\n b:"); printmatrix(b, srcRows, srcColumns);	  
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
	
		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60 + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)
			zppaddmatrix16(srcRows,srcColumns,a,b,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		//delay_ms(100);
		printf("\r\n %s  ",TimeStamp1);
		//delay_ms(100);
		printf("%s",TimeStamp2);
		//delay_ms(100);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		//delay_ms(100);
		printf(" before add %s  ",TimeStamp1);
		//delay_ms(100);
		printf("after add %s ",TimeStamp2);
		//printf("h=%d q=%d ",shijianhou,shijianqian)
		shijianchaadd[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchaadd[cishu]);
	
		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)
			zpptransposematrix16(srcRows,srcColumns,a,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		//delay_ms(100);
		printf("\r\n %s  ",TimeStamp1);
		//delay_ms(100);
		printf("%s",TimeStamp2);
		//delay_ms(100);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		//delay_ms(100);
		printf(" before transpose %s  ",TimeStamp1);
		//delay_ms(100);
		printf("after transpose %s ",TimeStamp2);
		shijianchatran[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchatran[cishu]);

		RTCbianliang.getTime();
		strcpy(TimeStamp1,RTCbianliang.timeStamp);
		shijianqian= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		times=100000;
		for(uint32_t m=0; m<times;m++)			
			zppmulmatrix16(srcRows,srcColumns,a,b,c);
		RTCbianliang.getTime();
		shijianhou= ((uint32_t)RTCbianliang.hour)*3600+ ((uint32_t)RTCbianliang.minute)*60  + RTCbianliang.second;
		strcpy(TimeStamp2,RTCbianliang.timeStamp);
		//delay_ms(100);
		printf("\r\n %s  ",TimeStamp1);
		//delay_ms(100);
		printf("%s",TimeStamp2);
		//delay_ms(100);
		printf("\r\n c:"); printmatrix(c, srcRows, srcColumns);
		//delay_ms(100);
		printf(" before mul %s  ",TimeStamp1);
		//delay_ms(100);
		printf("after mul %s",TimeStamp2);
		//delay_ms(100);
		shijianchamul[cishu]=(uint16_t)(shijianhou -shijianqian);
		printf("cha=%d \r\n",shijianchamul[cishu]);

		printf("end");
	}

	printf("add transpose mul in my way with uint16_t, k=10 20 30 40 50 \r\n");
	for(cishu=0;cishu<5;cishu++)
	{
		if(cishu==0){srcRows = 10; srcColumns = srcRows;}
		else if(cishu==1){srcRows = 20; srcColumns = srcRows;}
		else if(cishu==2){srcRows = 30; srcColumns = srcRows;}
		else if(cishu==3){srcRows = 40; srcColumns = srcRows;}
		else if(cishu==4){srcRows = 50; srcColumns = srcRows;}
		
		printf("\r\nrow=%d ",srcRows);		
		printf("\tadd 100000times =%ds \r\n",shijianchaadd[cishu]);
		printf("\ttrans 100000times =%ds \r\n",shijianchatran[cishu]);
		printf("\tmul 100000times =%ds \r\n",shijianchamul[cishu]);
			
	}
	printf("END");
}

void loop()
{
	while(1); 	
}
#endif //end BTESTMATRIXSIMPLE5




