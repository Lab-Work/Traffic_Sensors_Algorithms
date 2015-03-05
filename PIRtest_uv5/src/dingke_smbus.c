
#include "dingke_delay.h"
//////////////////////////////////////////////////////////////////////////////////	 
//MLX 9016使用说明：
//1.当有个新的（未使用过，或者不知道其器件地址）MLX90614，单独接到SMBUS总线上，也就是SMBUS总线上只有这一个MLX90614
//2.读一下它的地址，用slave_add= MEM_READ1(0x00,0x2e);slave_add低8位为这个新的MLX90614的器件地址，经测试，未使用的一般为0x5a
//3.把这个MLX90614的器件地址改成其他地址，比如0x2a, 则程序写成 changeMlx90614Subadd(0x00, 0x2a) 或者changeMlx90614Subadd(0x5a, 0xa0); 断电后再上电这个新地址才起作用
//4.当这个新的MLX90614 器件地址改过之后，多个不同的器件地址的MLX90614才可以接到一起
//5.当需要读传感器的温度时，调用	temp1=readMlx90614ObjectTemp(0xa0);	这里的0xa0是上面改好器件地址，temp1是得到的温度值为浮点数
//6.当好多器件已经接在一起时也可以改器件地址，注意 1. changeMlx90614Subadd(现在地址（不能是0x00）, 新地址)	2.改完地址之后必须要重新上电才可以
//7.读出器件地址slave_add= MEM_READ1(0x00,0x2e)只有当总线上只有一个器件读出来的地址才是正确的地址。
//
//
/*
注意事项：
1.写入之前要先擦除，也就是先也入0 ，在写入数据。
2.主要读写时，任意一位数据错误都将导致数据存在问题。
3.单个传感器接在系统时，从机地址可使用0x00,传感器默认地址为0x5a.
4.多个传感器并联时，地址将都改变成为相同的地址，并且数据错误。
5.若只将传感器的电源线去除，则读出的地址将变为0，并且数据错误。
6.若多个相同地址的传感器改变地址时，传感器地址仍相同。
7.多个相同传感器修改地址时，需单个一次修改，写地址后，下载后重新开始后才生效。因此要注意，并且若地址不生效时，后面的读写数据不正常。
8.难道传感器的地址只能一个一个的改？ 否 这是可以的 但是需要注意的一点 数据修改完毕后需要延时一段时间，释放总线。

*/
////////////////////////////////////////////////////////////////////////////////// 	  
#define TIMEZPPSMBUS 20
#define    SMBusSetSDA()  GPIO_SetBits(GPIOB,GPIO_Pin_11) //
#define    SMBusResetSDA()  GPIO_ResetBits(GPIOB,GPIO_Pin_11)  //
#define    SMBusSetSCL()  GPIO_SetBits(GPIOB,GPIO_Pin_10) //
#define    SMBusResetSCL()  GPIO_ResetBits(GPIOB,GPIO_Pin_10) //
#define    SMBusREAD_SDA   GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_11) //输入SDA 

//初始化IIC
void SMBusSDA_OUT(void)
{
 	GPIO_InitTypeDef GPIO_InitStructure;					     

	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
	
	//I2C_SCL PB.8   I2C_SDA PB.9 
	GPIO_InitStructure.GPIO_Pin =GPIO_Pin_11 ;//SDA
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
}

void SMBusSDA_IN(void)
{
 	GPIO_InitTypeDef GPIO_InitStructure;					     

	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
	
	//I2C_SCL PB.8   I2C_SDA PB.9 
	GPIO_InitStructure.GPIO_Pin =GPIO_Pin_11 ;	//SDA
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
}

void SMBus_poweron(void)//PE3
{
	GPIO_InitTypeDef GPIO_InitStructure;

	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE); 
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4  ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	GPIO_SetBits(GPIOA,GPIO_Pin_4);
}
//SMBus这边电源开了，把SDA SCL引脚初始化
void SMBus_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
						     
	SMBus_poweron();
	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); 
	
	//I2C_SCL PB.8   I2C_SDA PB.9 
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10 |GPIO_Pin_11 ;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
		
	GPIO_SetBits(GPIOB,GPIO_Pin_10);//SCL  
	GPIO_SetBits(GPIOB,GPIO_Pin_11);//SDA
}

//SMBus请求，这个芯片进入SMBus状态要请求下
void SMBus_Apply(void)
{
	 SMBusResetSCL();
				//SMBus请求时间，将PWM模式转换为SMBus模式(至少为2ms)
     delay_ms(3); //最小1.44ms 
	 SMBusSetSCL();
}


void send_bit(unsigned char bit_out)
{
   
  SMBusSDA_OUT();			  //设置SDA为开漏输出以在总线上传送数据


  if(bit_out==0)				  //核对字节的位
            					  //如果bit_out=1，设置SDA线为高电平
             //SDA=0;
	SMBusResetSDA();
  else							  
             //SDA=1;				  //如果bit_out=0，设置SDA线为低电平
	SMBusSetSDA();	

	delay_us(2 );
  //SCL=1;				  //设置SCL线为高电平
  SMBusSetSCL(); //SCL 
  delay_us(6 );

  //SCL=0;				  //设置SCL线为低电平
  SMBusResetSCL(); //SCL
  delay_us(2 );

}

unsigned char receive_bit()
{
  unsigned char bit_in;
  //_SDA_INPUT;				                //设置SDA为高阻输入
  SMBusSDA_IN();
  delay_us(2 );
  //SCL=1;					   //设置SCL线为高电平
  SMBusSetSCL(); //SCL 
  delay_us(6 );
  //if(SDA==1)					    //从总线上读取一位，赋给bit_in
  if(SMBusREAD_SDA)
       bit_in=1;
  else
       bit_in=0;
  //SCL=0;					   //设置SCL线为低电平
  SMBusResetSCL(); //SCL  
  delay_us(2);
  return bit_in;                                                          //返回bit_in值
}

unsigned char RX_byte(unsigned char ack_nack)
{
    unsigned char RX_buffer=0;
    unsigned char Bit_counter;
    for(Bit_counter=8;Bit_counter;Bit_counter--)
    {
		if(receive_bit()==1)	                //由SDA线读取一位
		   {
			RX_buffer<<=1;		   //如果位为"1"，赋"1"给RX_buffer 
			RX_buffer|=0x01;
		   }
		else				   //如果位为"0"，赋"0"给RX_buffer
		   {
			RX_buffer<<=1;
			RX_buffer&=0xfe;
		   }		
      } 
	 send_bit(ack_nack);			   //发送确认位

	 return RX_buffer;
}


void TX_byte(unsigned char TX_buffer)
{
   unsigned char Bit_counter;
   unsigned char bit_out;
     
   for(Bit_counter=8;Bit_counter;Bit_counter--)
   {
       if(TX_buffer&0x80)
		     bit_out=1;	               //如果TX_buffer的当前位是1,设置bit_out为1
		else
		     bit_out=0;	         	  //否则，设置bit_out为0
     send_bit(bit_out);			  //发送SMBus总线上的当前位   
     TX_buffer<<=1;		               //核对下一位		  
	}			            	                      
}

unsigned char slave_ack()
{
   unsigned char ack;
   ack=0;
   //_SDA_INPUT;				    //设置SDA为高阻输入
   SMBusSDA_IN();
	delay_us(2 );
   //SCL=1;					    //设置SCL线为高电平
   SMBusSetSCL();
   	delay_us(6 );
   //if(SDA==1)					    //从总线上读取一位，赋给ack
   if(SMBusREAD_SDA)
         ack=0;
   else
         ack=1; 

   //SCL=0;					    //设置SCL线为低电平
   SMBusResetSCL(); //SCL
   	delay_us(2 );   
   return ack;
}



unsigned char PECARR[7];				 //存储已发送字节的缓冲器

/*----------------------------------------------------------------------------------------------------------------------------------------//
计算PEC包裹校验码
函数名: PEC_cal
功能: 根据接收的字节计算PEC码
参数: unsigned char pec[], int n
返回值: pec[0] - 该字节包含计算所得crc数值
注解:	参考"系统管理总线说明书-版本2.0"和应用指南"MCU和MLX90614的SMBus通信"
//----------------------------------------------------------------------------------------------------------------------------------------*/
unsigned char PEC_cal(unsigned char pec[],int n)
{
     unsigned char crc[6];
     unsigned char Bitposition=n*8-1;
     unsigned char shift;
     unsigned char i;
     unsigned char j;
     unsigned char temp;

	for(i=0;i<7;i++)
	PECARR[i]=pec[i];
   do{
          crc[5]=0;           			       
          crc[4]=0;
          crc[3]=0;
          crc[2]=0;
          crc[1]=0x01;						    //载入 CRC数值 0x000000000107
          crc[0]=0x07;
          Bitposition=n*8-1;     		                     //设置Bitposition的最大值为47
          shift=0;
          //在传送的字节中找出第一个"1"

          i=n-1;                			        //设置最高标志位 (包裹字节标志)
          j=0;                			        //字节位标志，从最低位开始
          while((pec[i]&(0x80>>j))==0 && (i>0))	  
	  {
             Bitposition--;
             if(j<7)
	   {
                    j++;
                 }
             else
	      {
                   j=0x00;
                   i--;
                   }
           }//while语句结束，并找出Bitposition中为"1"的最高位位置
          shift=Bitposition-8;                                   //得到CRC数值将要左移/右移的数值"shift"
	                                                              //对CRC数据左移"shift"位
          while(shift)
	     {
              for(i=n-1;i<0xFF;i--)
		 {  
                    if((crc[i-1]&0x80) && (i>0))          //核对字节的最高位的下一位是否为"1"
		     {   			       //是 - 当前字节 + 1
                          temp=1;		       //否 - 当前字节 + 0
                     }				       //实现字节之间移动"1"
                    else
	             {
                          temp=0;
                     }
                     crc[i]<<=1;
                     crc[i]+=temp;
                  } 

                  shift--;
              } 
           //pec和crc之间进行异或计算
           for(i=0;i<=n-1;i++)
		   {
                   pec[i]^=crc[i];
		   }  
      }while(Bitposition>8); 
	return pec[0];                                  //返回计算所得的crc数值
} 


unsigned char PECwbparr[7];				 //存储已发送字节的缓冲器
unsigned char PECreg;				 //存储计算所得PEC字节

	 
/*----------------------------------------------------------------------------------------------------------------------------------------//
函数名: start_bit
功能: 在SMBus总线上产生起始状态
注解: 参考"系统管理总线说明书-版本2.0"
//----------------------------------------------------------------------------------------------------------------------------------------*/
void start_bit()
{
//GPIO_InitTypeDef GPIO_InitStructure;
   SMBusSDA_OUT();
   SMBusSetSDA();//设置SDA线为高电平		                   				       
   delay_us(2);
   			  
   SMBusSetSCL();// //SCL 				       //设置SCL线为高电平
   delay_us(5);				       //在终止和起始状态之间产生总线空闲时间(Tbuf=4.7us最小值)
   SMBusResetSDA(); //SDA				       //设置SDA线为低电平
   delay_us(5);				      
   //（重复）开始状态后的保持时间，在该时间后，产生第一个时钟信号
  					       //Thd:sta=4us最小值
   SMBusResetSCL(); //SCL				       //设置SCL线为低电平
   delay_us(10);

}


/*----------------------------------------------------------------------------------------------------------------------------------------//
函数名: stop_bit
功能: 在SMBus总线上产生终止状态
注解: 参考"系统管理总线说明书-版本2.0"
//----------------------------------------------------------------------------------------------------------------------------------------*/
void stop_bit()
{
  SMBusSDA_OUT();//sda线输出				 //设置SDA为输出
  SMBusResetSCL(); //SCL 			     		 //设置SCL线为低电平
  delay_us(5);
  SMBusResetSDA(); //SDA 					 //设置SDA线为低电平
  delay_us(5);
  SMBusSetSCL(); //SCL				             //设置SCL线为高电平
  delay_us(5);				             //终止状态建立时间(Tsu:sto=4.0us最小值)
  SMBusSetSDA();//SDA 				             //设置SDA线为高电平 
}
//判断SMBUS地址是否存在， 若存在返回地址，不存在时返回0XFF
unsigned char ACKaddress(unsigned char ADDRESS)
{
  unsigned char SLA;
  SLA=(ADDRESS<<1);	             
  start_bit(); 	  //发送起始位
  TX_byte(SLA);   //发送受控器件地址，写命令  
  delay_us(100);            			               					 
  if(!slave_ack())
  {
    stop_bit();
    return 0;
  }
  else
  {
    stop_bit();
    return 1;
  }
}


/*----------------------------------------------------------------------------------------------------------------------------------------//
由MLX90614 RAM/EEPROM 读取的数据
函数名: MEM_READ
功能: 给定受控地址和命令时由MLX90614读取数据
参数: unsigned char slave_addR (受控地址)
         unsigned char cmdR (命令)
返回值: unsigned long int Data
//----------------------------------------------------------------------------------------------------------------------------------------*/
//slave_addR 
//   SMBUS保留地址 	 0000 0000  广播呼叫地址				  //0x00
//                       0000 0001  起始地址					  //0x01
//                       0000 001x  CBUS地址					  //0x0?
//                       0000 010x  地址留给不同的总线格式		//0x0?
//                       0000 011x  保留将来使用				  //0x0?
//                       0000 1xxx  保留将来使用				  //0x0?
//                       0101 000x  保留供ACCESS.bus主机	      //0x5?
//                       0110 111x  保留供ACCESS.bus默认的地址	//0x6?
//                       1111 0xxx  10位从地址					  //0xf?
//                       1111 1xxx  保留将来使用				  //0xf?
//                       0001 000x  SMBus主机					  //0x1?
//                       0001 100x  SMBus报警响应地址			  //0x1?
//                       1100 001x  SMBus设备默认地址			  //0xc?



//cmdR 
//   MLX90614  000x xxxx 访问RAM  
//             001x xxxx 访问EEPROM
//	           1111 0000 读取标志符
//	           1111 1111 进入SLEEP模式

//             EPPROM  00 Tomax	     实际使用是并上001x xxxx(访问EEPROM的命令)	为0x20
//                     01 Tomin		                                             	为0x21
//                     02 PWMCTRL	                                             	为0x22
//                     03 Ta范围	                                            	为0x23
//                     04 发射率校准系数                                          为0x24
//                     05 配置寄存器1                                            	为0x25
//                     0e SMBUS地址                                            	    为0x2e
//                     1c ID编号
//                     1d ID编号
//                     1e ID编号
//                     1f ID编号

//             RAM     04  原始数据IR通道1 
//                     05  原始数据IR通道2 
//                     06  Ta 环境温度 
//                     07  Tobj1 目标温度1 
//                     08  Tobj1 目标温度2 


long MEM_READ1(unsigned char slave_addR, unsigned char cmdR)
{	
	 unsigned char DataL;		                           //
	 unsigned char DataH;				 //由MLX90614读取的数据包
   
   	 unsigned char SLA;
	 unsigned char Pec;
	 unsigned char cntack=0;
	 unsigned char cntrepeat=0;		

			// Initialising of ErrorCounter	

beginread:

	cntrepeat++;//每次重新来一次就加1，当超过100，说明smbus链路不通,返回-1
	if(cntrepeat>5)return -1;
										
	 SLA=(slave_addR<<1);	             
	 start_bit(); 	  //发送起始位
	 TX_byte(SLA);   //发送受控器件地址，写命令              			               
	 
	 cntack=0;					 
	 while(!slave_ack()){
	 	cntack++;
		if(cntack>=100)
		{
			cntack=0;  
			goto beginread;
		}
	 }
					  //发送命令
     TX_byte(cmdR);

	 cntack=0;
	 while(!slave_ack()){
	 	cntack++;
		if(cntack>=100)
		{
			cntack=0;  
			goto beginread;
		}
	 }			
	 start_bit(); 	                                                      //发送重复起始位				
	 TX_byte(SLA+1); 
	 
	 cntack=0; //发送受控器件地址，读命令                                                 
	 while(!slave_ack()){
	 	cntack++;
		if(cntack>=100)
		{
			cntack=0;
			goto beginread;
		}
	 }


	 DataL=RX_byte(0);				  //
							  //读取两个字节数据
	 DataH=RX_byte(0);				  //
	 Pec=RX_byte(1);

	 stop_bit(); 

	 PECwbparr[6]=(Pec);
	 PECwbparr[5]=(SLA);
     PECwbparr[4]=cmdR;
     PECwbparr[3]=(SLA+1);               
     PECwbparr[2]=DataL;
     PECwbparr[1]=DataH;
     PECwbparr[0]=0;                  										 
	 PECreg=PEC_cal(PECwbparr,6);  			  //调用计算 CRC 的函数

	  //读取MLX90614的PEC码                                                       //发送终止
	return 	 (   (((unsigned int)DataH)<<8)|(((unsigned int)DataL)&0x00ff)  );
}

/*----------------------------------------------------------------------------------------------------------------------------------------//
由MLX90614 RAM/EEPROM 读取的数据
函数名: MEM_READ
功能: 给定受控地址和命令时由MLX90614读取数据
参数: unsigned char slave_addR (受控地址)
         unsigned char cmdR (命令)
返回值: unsigned long int Data
//----------------------------------------------------------------------------------------------------------------------------------------*/
void MEM_WRITE1(unsigned char slave_addR, unsigned char cmdR,unsigned int data)
{	
   	 unsigned char SLA;
	 unsigned char DataL;		                           //
	 unsigned char DataH;				 //由MLX90614读取的数据包

	 SLA=(slave_addR<<1);
	 DataL=(unsigned char)(data);
	 DataH=(unsigned char)(data>>8);
     PECwbparr[4]=(SLA);
     PECwbparr[3]=cmdR;               
     PECwbparr[2]=DataL;
     PECwbparr[1]=DataH;
     PECwbparr[0]=0;  
	                   										 
	 PECreg=PEC_cal(PECwbparr,5);  			  //调用计算 CRC 的函数	
//	 printf(" PECcal=%x ewqd=%x  ",PECreg,PECARR[6]);		
					 	             
	   start_bit(); 	  //发送起始位

	  TX_byte(SLA);   //发送受控器件地址，写命令              			               
	   while(!slave_ack());					  //发送命令

      TX_byte(cmdR);
	   while(!slave_ack());				                                                      //发送重复起始位	
	 			
 	  TX_byte(DataL);                                                   //发送数据低地址
	  while(!slave_ack());

	  TX_byte(DataH);                                                   //发送数据高字节
	  while(!slave_ack());
	 
	    TX_byte(PECreg);                                                   //发送PEC
	    while(!slave_ack()); 

	  stop_bit(); 
	 delay_ms(10);
}

//读器件地址（总线上只有一个红外传感器才行，要不然多出来的值是所有的且值，即遇低变低）
int readMlx90614Subadd(void) 
{
	long zhi;
	unsigned char flag=0;
	while(1)
	{
		zhi = MEM_READ1(0x00,0x2e);
		if(zhi==-1)		  //zhi=-1说明尝试了很多次读，都读不到东西
		{
			SMBus_Apply();
			return -1;
		}
		if((zhi>=0x01)&&(zhi<=0xff))
		{
			return 	zhi;
		}
		flag++;
		if(flag>100)
		{	
			SMBus_Apply();
			return -1;
		}
	}
}


//修改温度传感器地址。INIT_addR:表示上一次的地址值。CURR_addR：表示当前地址值，也就是修改后的地址值
void changeMlx90614Subadd(unsigned char INIT_addR, unsigned int CURR_addR) 
{
  if(ACKaddress(INIT_addR))
  {
    MEM_WRITE1(INIT_addR,0x2E,0);	
	MEM_WRITE1(INIT_addR,0x2E,CURR_addR);
  }
}

//返回的是小数，例如返回是19.87，则就是19.87度
static float trantocentigradefloat(unsigned int zhi)
{
	//return (((float)zhi/50.0)-273.15);
	return (((double)zhi*0.02)-273.15);
}


//读环境温度 返回值为小数
float readMlx90614AmbientTemp(unsigned char slave_addR)
{	
	long zhi;
	unsigned char flag=0;
	while(1)
	{
		zhi = MEM_READ1(slave_addR,0x06);
		if(zhi==-1)		  //zhi=-1说明尝试了很多次读，都读不到东西
		{
			SMBus_Apply();
			return 0.0;
		}
		if((zhi>=0x2DEA)&&(zhi<=0x4DC4))
		{
			return 	trantocentigradefloat(zhi);
		}
		flag++;
		if(flag>100)
		{	
			SMBus_Apply();
			return 0.0;
		}
	}
}
//读目标温度
//这个函数发现读对应地址数据有问题会重启这个温度传感器的，如果不这样做，一旦有一个传感器没有接好或者地址不对造成读数据错（本身这个问题不是什么大问题） 会造成以后读其他地址也会错
float readMlx90614ObjectTemp(unsigned char slave_addR)
{	
	long zhi;
	unsigned char flag=0;
	while(1)
	{
		zhi = MEM_READ1(slave_addR,0x07);
		if(zhi==-1)	//zhi=-1说明尝试了很多次读，都读不到东西
		{
			SMBus_Apply();
			return 0.0;
		}
		
		if((zhi<=0x7fff))
		{
		   return 	trantocentigradefloat(zhi);
		}
		flag++;
		if(flag>2)
		{
			SMBus_Apply();
			return 0.0;
		}
	}
}














