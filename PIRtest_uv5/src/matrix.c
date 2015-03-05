#include "matrix.h"
#include <stdint.h>

//A+B->C
void zppaddmatrix16(uint16_t numRows,uint16_t numCols,int16_t *A,int16_t *B,int16_t *C)
{
	uint16_t i;
	uint16_t k;	
  for( i=0;i<numRows;i++){ 
    for( k=0;k<numCols;k++)
    {
      (*(C+i*numCols+k)) = (*(A+i*numCols+k)) + (*(B+i*numCols+k));
    }
  }
}
//A*B->C
void zppmulmatrix16(uint16_t numRows,uint16_t numCols,int16_t *A,int16_t *B,int16_t *C)
{
	uint16_t i;
	uint16_t k;	
  for(i=0;i<numRows;i++){ 
    for(k=0;k<numCols;k++)
    {
      (*(C+i*numCols+k)) = (int16_t)((*(A+i*numCols+k)) * (*(B+i*numCols+k)));
    }
  }
}
//A->C
void zpptransposematrix16(uint16_t numRows,uint16_t numCols,int16_t *A,int16_t *C)//
{
	uint16_t i;
	uint16_t k;	
  for(i=0;i<numRows;i++){ 
    for(k=0;k<numCols;k++)
    {
      (*(C+k*numRows+i)) = (int16_t)(*(A+i*numCols+k)) ;
    }
  }
}


//A+B->C
void zppaddmatrix(uint16_t numRows,uint16_t numCols,float *A,float *B,float *C)
{
	uint16_t i;
	uint16_t k;	
  for( i=0;i<numRows;i++){ 
    for( k=0;k<numCols;k++)
    {
      (*(C+i*numCols+k)) = (*(A+i*numCols+k)) + (*(B+i*numCols+k));
    }
  }
}
//A*B->C
void zppmulmatrix(uint16_t numRows,uint16_t numCols,float *A,float *B,float *C)
{
	uint16_t i;
	uint16_t k;	
  for(i=0;i<numRows;i++){ 
    for(k=0;k<numCols;k++)
    {
      (*(C+i*numCols+k)) = ((*(A+i*numCols+k)) * (*(B+i*numCols+k)));
    }
  }
}
//A->C
void zpptransposematrix(uint16_t numRows,uint16_t numCols,float *A,float *C)//
{
	uint16_t i;
	uint16_t k;	
  for(i=0;i<numRows;i++){ 
    for(k=0;k<numCols;k++)
    {
      (*(C+k*numRows+i)) = (*(A+i*numCols+k)) ;
    }
  }
}
//下面这些若干个函数都是为了求逆做准备的。
//求逆有很多方式，这里采用了左边添加10矩阵（就是那个对角线都是1），然后行变换得到求逆的
//如果没有逆矩阵会返回-1
//矩阵行对换
void zijuhangchange(float *a,uint16_t row, uint16_t col, uint16_t row1, uint16_t row2)
{
	uint16_t i;
	float temp;
	for(i=0;i<col;i++)
	{
		temp = *(a+col*row1+i);
		*(a+col*row1+i) = *(a+col*row2+i);		
		 *(a+col*row2+i) = temp;
	}
}/**/
//a row*col -> c row1 * 2col//把a分拆成左右两部分，右边的为col1，给c
void jianjietoyou(float *a, float *c, uint16_t row, uint16_t col,uint16_t col1)
{
	uint16_t m,n;
	for(m=0;m<row;m++)
	{
		for(n=0;n<col1;n++)
		{
			*(c+m*col1+n) =*(a+m*col+(col-col1)+n) ;
		}
	}
}

//a row*col b row*col   -> c row * 2col//左右拼接矩阵
void pingjiezuoyou(float *a, float *b, float *c, uint16_t row, uint16_t col)
{
	uint16_t m,n;
	for(m=0;m<row;m++)
	{
		for(n=0;n<col;n++)
		{
			*(c+m*2*col+n) = *(a+m*col+n);
		}
		for(n=0;n<col;n++)
		{
			*(c+m*2*col+col+n) = *(b+m*col+n);
		}
	}
}
//初始化矩阵，让其对角线为1，其他为0
void init10juzhen(float *a, uint16_t k)
{
	uint16_t i;
	for(i=0;i<k*k;i++)
		*(a+i)=0;
	for(i=0;i<k;i++)
		*(a+i*k+i)=1;
}
float absfloat(float a)
{
	if(a<0.0)return -a;
	else return a;
}

//寻找矩阵指定jibie行列右下角（包括当前jiebie行列）的矩阵 开头值最小的那一行//这里的最小是指绝对值最小
int minkaitou(float *a, uint16_t row, uint16_t col, uint16_t jibie)
{
	uint16_t m;
	float min;
	int minjibie;
	float panbie0;
	//先判断有没有非0 的值，如果都是0值，那返回错误-1，这种情况是肯定没有逆矩阵的
	//第一个非0值为 min值，
	minjibie=-1;
	for(m=jibie;m<row;m++)
	{
		panbie0 = absfloat(*(a+m*col+jibie));
		if((panbie0>-0.0001)&&(panbie0<0.0001));
		else 
		{
			minjibie=m;
			min = panbie0;
			break;
		}
	}
	if(minjibie==-1)return -1;


	//printf("first min =%f \n",min);
	for(m=jibie+1;m<row;m++)
	{
		//printf("m=%d zhi=%f ",m,*(a+m*col+jibie));
		panbie0 = absfloat(*(a+m*col+jibie));
		if((panbie0>-0.0001)&&(panbie0<0.0001))continue;

		if(panbie0<min)
		{			
			min = panbie0;
			minjibie = m;
			//printf(" now min =%f  minjibie+%d \n",min,minjibie);
		}
	}
	return minjibie;
}

//矩阵指定第i j元素变成1，那么矩阵那一行需要乘以多少，把那一行都乘上这个数，目的就是为了i j这个元素变为1
void juzhenchangelementto1(float *a, uint16_t row, uint16_t col, uint16_t i, uint16_t j)
{
	uint16_t m;
	if((*(a+i*col+j)>0.9999)&&(*(a+i*col+j)<1.0001))return;

	for(m=0;m<col;m++)
	{
		if(m==j)continue;
		else *(a+i*col+m) = *(a+i*col+m)/(*(a+i*col+j));
	}
	*(a+i*col+j)=1.0;
}

//矩阵 指定行减去另一个指定行的*bei
void juzhensub(float *a, uint16_t row, uint16_t col, uint16_t row1, float bei, uint16_t row2)
{
	uint16_t m;
	if((bei>-0.0001)&&(bei<0.0001))return;
	for(m=0;m<col;m++)
	{
		*(a+row2*col+m) = *(a+row2*col+m) - (*(a+row1*col+m))*bei;
	}
}
//这里采用了左边添加10矩阵（就是那个对角线都是1），然后行变换得到求逆的
//如果没有逆矩阵会返回-1
int zppinversematrix(float *a, uint16_t k,float *c)
{
	float *b;
	float *d;
	uint16_t m,n;
	uint16_t jibie=0;
	int hang=0;
	uint16_t i1;
	b = (float *)malloc(sizeof(float)*k*2*k);
	if(b==NULL){
		//printf(" no enough room for fb ");
		return -2;
	}

	d = (float *)malloc(sizeof(float)*k*k);
	if(d==NULL){
		//printf(" no enough room for fd ");
		return -3;
	}
	//得到对角线都是1的那个矩阵
	init10juzhen(d, k);
	//把原矩阵和对角线都是1的矩阵两个拼接一下
	pingjiezuoyou(a, d, b, k, k);
	free(d);
//	for(m=0;m<k;m++)
//	{
//		for(n=0;n<2*k;n++)
//			printf("%f ",*(b+m*2*k+n));
//		printf("\n");
//	}

	for(jibie=0;jibie<k;jibie++)
	{
		hang=minkaitou(b, k, 2*k, jibie);		
//		printf("hang=%d \n",hang);
		if(hang==-1)
		{
			printf(" no inv \n");
			return -1;
		}

		zijuhangchange(b,k, 2*k, jibie, hang);
//		for(m=0;m<k;m++)
//		{
//			for(n=0;n<2*k;n++)
//				printf("%f ",*(b+m*2*k+n));
//			printf("\n");
//		}

		juzhenchangelementto1(b, k, 2*k, jibie, jibie);
//		for(m=0;m<k;m++)
//		{
//			for(n=0;n<2*k;n++)
//				printf("%f ",*(b+m*2*k+n));
//			printf("\n");
//		}
//		printf("to 1\n");

//		for(i1=0;i1<k;i1++)
//		{
//			if(i1==jibie)continue;
//			printf(" *(b+i1*2*k+jibie)=%f \n",*(b+i1*2*k+jibie));
//			juzhensub(b, k, 2*k, jibie, *(b+i1*2*k+jibie), i1);	
//		}
//		for(m=0;m<k;m++)
//		{
//			for(n=0;n<2*k;n++)
//				printf("%f ",*(b+m*2*k+n));
//			printf("\n");
//		}
//		printf("sub\n");
	}


	jianjietoyou(b, c, k, 2*k,k);
//	for(m=0;m<k;m++)
//	{
//		for(n=0;n<k;n++)
//			printf("%f ",*(c+m*k+n));
//		printf("\n");
//	}


	free (b);

	return 0;

}









/*************************************************
函数名： mSizeOf
函数作用 ：返回数据类型占的字节数
参数：MAT_DATA_TYPE 类型的变量
返回：矩阵的头指针
说明：
**************************************************/
uchar mSizeOf(MAT_DATA_TYPE type)
{
     if (IS_RIGHT_DATATYPE(type))
     {
		 switch (type)
		 {
		 case MAT_DATA_TYPE_CHAR:
			 return sizeof(uchar);
			 //break;
		 case MAT_DATA_TYPE_UCHAR:
			 return sizeof(char);
			 //break;
		 case MAT_DATA_TYPE_INT:
			 return sizeof(int);
			 //break;
		 case MAT_DATA_TYPE_UINT:
			 return sizeof(uint);
			 //break;
		 case MAT_DATA_TYPE_LONG:
			 return sizeof(long);
			 //break;
		 case MAT_DATA_TYPE_ULONG:
			 return sizeof(ulong);
			 //break;
		 case MAT_DATA_TYPE_FLOAT:
			 return sizeof(float);
			 //break;
		 case MAT_DATA_TYPE_DOUBLE:
			 return sizeof(double);
			 //break;
     }

	}
	 return 0;
}
/*************************************************
函数名：   DataTypeTo
函数作用 ：对数据进行转换
参数：     MAT_DATA_TYPE 类型的变量
           data void*指针
返回：     返回转换后的指针，
说明：     返回后的数据必须进行强类型转换
**************************************************/
void* DataTypeTo(void *data,MAT_DATA_TYPE type)
{
     if (IS_RIGHT_DATATYPE(type))
     {
		 switch (type)
		 {
		 case MAT_DATA_TYPE_CHAR:
			 return (uchar*)data;
			 //break;
		 case MAT_DATA_TYPE_UCHAR:
			 return (char*)data;
			 //break;
		 case MAT_DATA_TYPE_INT:
			 return (int*)data;
			 //break;
		 case MAT_DATA_TYPE_UINT:
			 return (uint*)data;
			 //break;
		 case MAT_DATA_TYPE_LONG:
			 return (long*)data;
			 //break;
		 case MAT_DATA_TYPE_ULONG:
			 return (ulong*)data;
			 //break;
		 case MAT_DATA_TYPE_FLOAT:
			 return (float*)data;
			 //break;
		 case MAT_DATA_TYPE_DOUBLE:
			 return (double*)data;
			 //break;
     }

	}
	 return NULL;
}
/*************************************************
函数名：   mGetAddr()
函数作用 ：返回指定x y 的地址
参数：     
返回：     返回类型uchar *
说明：
**************************************************/
uchar * mGetAddr(CMat *mat , uint x ,uint y)
{
	uchar *ptr;
	if ( mat==NULL )
    {
		return NULL;
    }
	if ( !IS_MAT(mat) )       //is mat
    {
		return NULL;
    }
	if (! IS_EMPTY_MAT(mat)) //data is valid
	{
		return NULL;
	}
    if (IS_VALID_XY(mat,x,y) ) //判断xy有效
    {
		ptr = (uchar*)(mat->data.ptr + y * mat->step + x * mSizeOf(mat->type));
    }
	return ptr ;	
}
/*************************************************
函数名：   mGetRealData()
函数作用 ：根据数据类型返回该地址处的数据
参数：     
返回：     double类型
说明：     
**************************************************/
double mGetRealData(uchar * addr , MAT_DATA_TYPE type)
{
	 double val = 0 ;
     
     if (IS_RIGHT_DATATYPE(type))
     {
		 switch (type)
		 {
		 case MAT_DATA_TYPE_CHAR:
			 val = * ((char*)addr);
			 break;
		 case MAT_DATA_TYPE_UCHAR:
			 val =  * ((uchar*)addr);
			 break;
		 case MAT_DATA_TYPE_INT:
		 	 val =  * ((int*)addr);
			 break;
		 case MAT_DATA_TYPE_UINT:
			 val =  * ((uint*)addr);
			 break;
		 case MAT_DATA_TYPE_LONG:
			 val =  * ((long*)addr);
			 break;
		 case MAT_DATA_TYPE_ULONG:
			 val = * ((ulong*)addr);
			 break;
		 case MAT_DATA_TYPE_FLOAT:
			 val =  * ((float*)addr);
			 break;
		 case MAT_DATA_TYPE_DOUBLE:
			 val =  * ((double*)addr);
			 break;
		 }
		 
	 }
	 else
	 {
		return -1;
	 }

	 return val ;
}
/*************************************************
函数名：   mSetRealData()
函数作用 ：根据数据类型返回该地址处的数据
参数：     
返回：     double类型
说明：     
**************************************************/
double mSetRealData(uchar * addr , MAT_DATA_TYPE type , double val)
{   
     if (IS_RIGHT_DATATYPE(type))
     {
		 switch (type)
		 {
		 case MAT_DATA_TYPE_CHAR:
			 * ((char*)addr) = (char)val;
			 break;
		 case MAT_DATA_TYPE_UCHAR:
			 * ((uchar*)addr) = (uchar)val;
			 break;
		 case MAT_DATA_TYPE_INT:
		 	 * ((int*)addr) = (int)val;
			 break;
		 case MAT_DATA_TYPE_UINT:
			 * ((uint*)addr) = (uint)val;
			 break;
		 case MAT_DATA_TYPE_LONG:
			 * ((long*)addr) = (long)val;
			 break;
		 case MAT_DATA_TYPE_ULONG:
			 * ((ulong*)addr) = (ulong)val;
			 break;
		 case MAT_DATA_TYPE_FLOAT:
			 * ((float*)addr) = (float)val;
			 break;
		 case MAT_DATA_TYPE_DOUBLE:
			 * ((double*)addr) = (double)val;
			 break;
		 }
		 
	 }
	 else
	 {
		return -1;
	 }

	 return val ;
}
/*************************************************
函数名： CreateMatHeader
函数作用 ：创建一个矩阵
参数：
     cols 列数
	 rows 行数
	 type 数据类型
	 data 数据指针
返回：矩阵的头指针
说明：函数在使用的时候，如果data == NULL ，则创建空间
**************************************************/
CMat *CreateMat(uint cols,uint rows,MAT_DATA_TYPE type, void *data)
{
	CMat *tmat ;
	tmat = (CMat *)malloc(sizeof(CMat));                   //这里需要重新看一下，void指针不知道开辟多少空间
	if(cols <= 0 || rows <= 0 || type > MAT_DATA_TYPE_MAX)
	{
		return (CMat*)NULL;
	}
	tmat->rows = rows;
	tmat->cols = cols;
	tmat->type = type;
	tmat->step = mSizeOf(type) * cols;
    tmat->data.ptr = malloc(rows*tmat->step);
	memset(tmat->data.ptr,0,rows *tmat->step);
	if (data != NULL)
	{
		memcpy(tmat->data.ptr,data,rows *tmat->step);
	}
	return tmat ;	   
}
/*************************************************
函数名：SetMatData
函数作用 ：设置矩阵的数据
参数：
    mat  矩阵的指针
	data 数据
返回：矩阵的头指针
说明：
**************************************************/
CMat *SetMatData(CMat* mat , void *data)
{
	if ( mat==NULL )
    {
		return NULL;
    }
    if ( !IS_MAT(mat) )
    {
		return (CMat *)NULL;
    }
    memcpy(mat->data.ptr,data,mat->rows *mat->step);
    return mat ;	   
}
/*************************************************
函数名：RealseMat
函数作用 ：释放矩阵占的空间
参数：
    mat  矩阵的指针
返回：
说明：
**************************************************/
void *RealseMat(CMat* mat)
{
	if ( mat==NULL )
    {
		return NULL;
    }
    if ( !IS_MAT(mat) )
    {
		return (CMat *)NULL;
    }
	free(mat->data.ptr);
	free(mat);
    return mat ;	   
}
/*************************************************
函数名：GetData2D
函数作用 ：获得矩阵的数据
参数：
    mat  矩阵的指针
	rows -->y
	cols -->x
返回：
说明：
**************************************************/
double GetData2D(CMat* mat,uint cols,uint rows)
{
	uchar *ptr = NULL;
	double val =0 ;
	if ( mat==NULL )
    {
		return -1;
    }
    if ( !IS_MAT(mat) )
    {
		return -1;
    }
	if (! IS_EMPTY_MAT(mat))
	{
		return -1;
	}
    
	ptr = mGetAddr(mat,cols,rows);
	if (ptr)
	{
		val = mGetRealData(ptr,mat->type);
	}

	return val;	
}
/*************************************************
函数名：SetData2D
函数作用 ：获得矩阵的数据
参数：
    mat  矩阵的指针
	rows -->y
	cols -->x
	val  -->设置的值，浮点类型
	type -->设置的类型，如果为0，使用默认矩阵的类型，也就是db类型，否则使用设置的类型
返回：
说明：
**************************************************/
double SetData2D(CMat* mat,uint cols,uint rows,double val,MAT_DATA_TYPE type)
{
	uchar *ptr = NULL;

    if ( mat==NULL )
    {
		return -1;
    }
    if ( !IS_MAT(mat) )
    {
		return -1;
    }
	if (! IS_EMPTY_MAT(mat))
	{
		return -1;
	}
    
	ptr  = mGetAddr(mat,cols,rows);
	if (type == MAT_DATA_TYPE_NULL)
	{
		if (mSetRealData(ptr,mat->type,val) != val)
		{
			return -1;
		}
	}
	else
	{
		if (mSetRealData(ptr,type,val) != val)
		{
			return -1;
		}
	}

    return val;
}
/*************************************************
函数名：PrintMat
函数作用 ：输出矩阵
参数：
    mat  矩阵的指针
返回：
说明：
**************************************************/
void  PrintMat(CMat* mat)
{
	uint rows=0;
	uint cols=0;
    if (IS_EMPTY_MAT(mat) )
    {
		for(rows=0; rows < mat->rows; rows++)//行
		{
			printf("\r\n");
			for (cols = 0 ;cols < mat->cols ; cols ++)
			{
				if (IS_RIGHT_DATATYPE(mat->type))
				{
					switch (mat->type)
					{
					case MAT_DATA_TYPE_CHAR:
						printf(" %4d " , (char)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_UCHAR:
						printf(" %4d " , (uchar)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_INT:
						printf(" %d" , (int)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_UINT:
						printf(" %d" ,  (uint)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_LONG:
						printf(" %d " ,  (long)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_ULONG:
						printf(" %d " , (ulong)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_FLOAT:
						printf(" %f " , (float)GetData2D(mat,cols,rows));
						break;
					case MAT_DATA_TYPE_DOUBLE:
						printf(" %f " , (double)GetData2D(mat,cols,rows));
						break;
					}
					
				}
			}
		}

	}
 
}
/*************************************************
函数名：    ReshapeMat
函数作用    改变矩阵的行数和列数
参数：
            mat
返回：      改变以后的矩阵c*r = c1 * r1
说明：      
**************************************************/
CMat *ReshapeMat(CMat *mat,uint rows,uint cols)
{
	if (mat == NULL)
	{
		return NULL;
	}
	if (!IS_MAT(mat))
	{
		return NULL;
	}
	if (mat->cols * mat->rows != rows * cols)
	{
		return NULL;
	}
	mat->cols = cols;
	mat->rows = rows;
	mat->step = cols * mSizeOf(mat->type);

	return mat;
}
/*************************************************
函数名：    AddMat
函数作用    矩阵加法
参数：
            mat mat1 mat2 矩阵指针
返回：      发挥相同大小的矩阵
说明：      mat在外部没有开辟空间的话，在内部会根据加数矩阵去开辟响应的空间
            mat的结果与mat本身的类型相同
			如果mat与加数矩阵中的一个相同，那么mat按相同的那个操作，不再对mat内部的data进行释放个重新开辟的操作
			mat = mat1+mat2
**************************************************/
CMat *AddMat(CMat *mat ,CMat *mat1,CMat *mat2)
{
	uint rows =0;
	uint cols = 0;
	uint i = 0;
	uint j = 0;	
	double val =0 ;
	if (mat1 == NULL || mat2 == NULL )
	{   
		return NULL;
	}
	if (!IS_MAT(mat1) || !IS_MAT(mat2) )
	{   
		return NULL;
	}
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols )
	{
		return NULL;
	}
	rows = mat1->rows;
	cols = mat1->cols;
//	这样不能判断是不是矩阵，因为你在外部没有初始化，不能使用IS_MAt
    if (mat == NULL)                                       //如果mat不是矩阵，根据MAT11和MAT2的类型来新建矩阵
	{
		if (mat1->type == mat2->type)
		{
			mat = CreateMat(cols,rows,mat1->type,NULL);
		}
		else
		{
			mat = CreateMat(cols,rows,MAT_DATA_TYPE_DOUBLE,NULL);
		}
	}
	
	if (mat->cols != cols || mat->rows != rows)       
	{
		return NULL;
	}

    if ( !(mat->type > MAT_DATA_TYPE_NULL && mat->type <= MAT_DATA_TYPE_MAX) )    	 //如果原来的类型是空的                
    {
		free(mat->data.ptr);
		mat->data.ptr = malloc(rows * cols * mSizeOf(MAT_DATA_TYPE_DOUBLE));
		for (i =0 ; i < rows ; i ++)
		{
			for (j =0 ;j < cols ;j++)
			{
				val = GetData2D(mat1,j,i) + GetData2D(mat2,j,i);
				if (SetData2D(mat,j,i,val,MAT_DATA_TYPE_DOUBLE) != val)
				{
					return NULL;
				}		
			}
		}
    }
	else
	{
		if (mat != mat1 && mat != mat2)                              //如果mat和mat1 mat2都不是一个矩阵
		{
			free(mat->data.ptr);                                     //释放原有的空间，以面造成内存泄漏
			mat->data.ptr = malloc(rows * cols * mSizeOf(mat->type));//重新分配空间,按照mat的数据类型
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = GetData2D(mat1,j,i) + GetData2D(mat2,j,i);
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}		
				}
			}
		}
		else if (mat == mat1 ||  mat == mat2)
		{
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = GetData2D(mat1,j,i) + GetData2D(mat2,j,i);
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}		
				}
			}
		}
	
	}

	return mat;		
}


CMat *TransposeMat(CMat *mat1,CMat *mat2)
{
	uint rows =0;
	uint cols = 0;
	uint i = 0;
	uint j = 0;	
	double val =0 ;
	if (mat1 == NULL || mat2 == NULL )
	{   
		return NULL;
	}
	if (!IS_MAT(mat1) || !IS_MAT(mat2) )
	{   
		return NULL;
	}
	if (mat1->rows != mat2-> cols|| mat1->cols != mat2->rows )
	{
		return NULL;
	}

	rows = mat1->rows;
	cols = mat1->cols;
	for (i =0 ; i < rows ; i ++)
	{
		for (j =0 ;j < cols ;j++)
		{
			val = GetData2D(mat1,j,i) ;
			//val = 5;
			if (SetData2D(mat2,i,j,val,mat2->type) != val)
			{
				return NULL;
			}		
		}
	}
	return mat2;		
}




/*************************************************
函数名：    SubMat
函数作用    矩阵减法
参数：
            mat mat1 mat2 矩阵指针
返回：      发挥相同大小的矩阵
说明：      见 AddMat 的说明
            mat = mat1 - mat2;
**************************************************/
CMat *SubMat(CMat *mat ,CMat *mat1,CMat *mat2)
{
	uint rows =0;
	uint cols = 0;
	uint i = 0;
	uint j = 0;	
	double val =0 ;
	if (mat1 == NULL || mat2 == NULL )
	{   
		return NULL;
	}
	if (!IS_MAT(mat1) || !IS_MAT(mat2) )
	{   
		return NULL;
	}
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols )
	{
		return NULL;
	}
	rows = mat1->rows;
	cols = mat1->cols;
	//	这样不能判断是不是矩阵，因为你在外部没有初始化，不能使用IS_MAt
    if (mat == NULL)                                       //如果mat不是矩阵，根据MAT11和MAT2的类型来新建矩阵
	{
		if (mat1->type == mat2->type)
		{
			mat = CreateMat(cols,rows,mat1->type,NULL);
		}
		else
		{
			mat = CreateMat(cols,rows,MAT_DATA_TYPE_DOUBLE,NULL);
		}
	}
	
	if (mat->cols != cols || mat->rows != rows)       
	{
		return NULL;
	}
	
    if ( !(mat->type > MAT_DATA_TYPE_NULL && mat->type <= MAT_DATA_TYPE_MAX)  )      	 //如果原来的类型是空的                
    {
		free(mat->data.ptr);
		mat->data.ptr = malloc(rows * cols * mSizeOf(MAT_DATA_TYPE_DOUBLE));
		for (i =0 ; i < rows ; i ++)
		{
			for (j =0 ;j < cols ;j++)
			{
				val = GetData2D(mat1,j,i) - GetData2D(mat2,j,i);
				if (SetData2D(mat,j,i,val,MAT_DATA_TYPE_DOUBLE) != val)
				{
					return NULL;
				}		
			}
		}
    }
	else
	{
		if (mat != mat1 && mat != mat2)                              //如果mat和mat1 mat2都不是一个矩阵
		{
			free(mat->data.ptr);                                     //释放原有的空间，以面造成内存泄漏
			mat->data.ptr = malloc(rows * cols * mSizeOf(mat->type));//重新分配空间,按照mat的数据类型
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = GetData2D(mat1,j,i) - GetData2D(mat2,j,i);
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}		
				}
			}
		}
		else if (mat == mat1 ||  mat == mat2)
		{
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = GetData2D(mat1,j,i) - GetData2D(mat2,j,i);
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}		
				}
			}
		}
		
	}
	
	return mat;		
	
}
/*************************************************
函数名：    MulMat
函数作用    矩阵乘法
参数：
            mat mat1 mat2 矩阵指针
返回：      
说明：      乘法运算中 mat1的cols（列数）等于mat2的rows（行数）
**************************************************/
CMat *MulMat(CMat *mat ,CMat *mat1,CMat *mat2)
{
	uint rows =0;
	uint cols = 0;
	uint i = 0 ;
	uint j = 0 ;	
	uint n = 0 ;
	double val =0 ;
	if (mat1 == NULL || mat2 == NULL )
	{   
		return NULL;
	}
	if (!IS_MAT(mat1) || !IS_MAT(mat2) )
	{   
		return NULL;
	}
	if (mat1->cols != mat2->rows )    //乘法运算中 mat1的cols（列数）等于mat2的rows（行数）
	{
		return NULL;
	}
	rows = mat1->rows;                //得到的结果：mat的行数rows等于mat1的 行数 列数（cols）等于mat2的cols
	cols = mat2->cols;
	//	这样不能判断是不是矩阵，因为你在外部没有初始化，不能使用IS_MAt
    if (mat == NULL)                                       //如果mat不是矩阵，根据MAT11和MAT2的类型来新建矩阵
	{
		if (mat1->type == mat2->type)
		{
			mat = CreateMat(cols,rows,mat1->type,NULL);
		}
		else
		{
			mat = CreateMat(cols,rows,MAT_DATA_TYPE_DOUBLE,NULL);
		}
	}
	
	if (mat->cols != cols || mat->rows != rows)       
	{
		return NULL;
	}
	
    if ( !(mat->type > MAT_DATA_TYPE_NULL && mat->type <= MAT_DATA_TYPE_MAX) )      	 //如果原来的类型不符合要求              
    {
		free(mat->data.ptr);
		mat->data.ptr = malloc(rows * cols * mSizeOf(MAT_DATA_TYPE_DOUBLE));
		for (i =0 ; i < rows ; i ++)
		{
			for (j =0 ;j < cols ;j++)
			{
				val = 0;
				for (n = 0 ; n < mat1->cols ; n++)
				{
					val += GetData2D(mat1,n,i) * GetData2D(mat2,j,n);				
				}	
				if (SetData2D(mat,j,i,val,MAT_DATA_TYPE_DOUBLE) != val)
				{
					return NULL;
				}	
			}
		}
    }
	else
	{
		if (mat != mat1 && mat != mat2)                              //如果mat和mat1 mat2都不是一个矩阵
		{
			free(mat->data.ptr);                                     //释放原有的空间，以面造成内存泄漏
			mat->data.ptr = malloc(rows * cols * mSizeOf(mat->type));//重新分配空间,按照mat的数据类型
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = 0;
					for (n = 0 ; n < mat1->cols ; n++)
					{
						val += GetData2D(mat1,n,i) * GetData2D(mat2,j,n);				
					}	
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}	
				}
			}
		}
		else if (mat == mat1 ||  mat == mat2)
		{
			for (i =0 ; i < rows ; i ++)
			{
				for (j =0 ;j < cols ;j++)
				{
					val = 0;
					for (n = 0 ; n < mat1->cols ; n++)
					{
						val += GetData2D(mat1,n,i) * GetData2D(mat2,j,n);				
					}	
					if (SetData2D(mat,j,i,val,mat->type) != val)
					{
						return NULL;
					}	
				}
			}
		}
		
	}

	return mat;		
}

