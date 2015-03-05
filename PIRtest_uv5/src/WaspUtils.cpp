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
 *  Version:		0.12
 *  Design:		David Gascón
 *  Implementation:	Alberto Bielsa, David Cuartielles
 */
  

#ifndef __WPROGRAM_H__
	#include "WaspClasses.h"
#endif

#include <inttypes.h>
#include <stm32f4xx.h>

/*
 * Constructor
 */
WaspUtils::WaspUtils (void)
{
}


/*
 * sizeOf ( str ) - size of a string or array
 *
 * returns the size of a string or byte array, it makes a check
 * for end of string ('\0') and discounts 1 if needed
 */
int WaspUtils::sizeOf(const char* str)
{
  int cont = 0;
  while(*str++) cont++;
  if (*str == '\0')
    return cont--; // it will end with '\0', therefore we gotta take it out
  return cont;
}

/*
 * strCmp ( str1, str2, size ) - compare two strings
 *
 * returns 0 if str1 is equal to str2 for the first "size" characters, 
 * returns 1 otherwise
 */
uint8_t WaspUtils::strCmp(const char* str1, const char* str2, uint8_t size)
{
  uint8_t cmp = 0;
  for (int i = 0; i < size; i++)
    if (str1[i] != str2[i]) cmp = 1;
  return cmp;
}

void	WaspUtils::initLEDs(void)
{
  	GPIO_InitTypeDef  GPIO_InitStructure;//GPIO初始化结构体
  	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);//外设时钟使能
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);


	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;	//PD5是置高为亮，PE0PE1为置地为亮 //新班子的引脚
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);	
	GPIO_SetBits(GPIOE, GPIO_Pin_0); //灭
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1;	//PD5是置高为亮，PE0PE1为置地为亮 //新班子的引脚
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOE, &GPIO_InitStructure);	
	GPIO_SetBits(GPIOE, GPIO_Pin_1);//灭
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5;	//PD5是置高为亮，PE0PE1为置地为亮 //新班子的引脚
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);	
	GPIO_ResetBits(GPIOD, GPIO_Pin_5);//灭
}



/* setLED(led, state) - set the specified LED to the specified state(ON or OFF)
 *
 * It sets the specified LED to the specified state(ON or OFF)
 */
void	WaspUtils::setLED(uint8_t led, uint8_t state)
{
	if(led==0)//LED0   //
	{	
		if(state==0)//灭
		{
			GPIO_SetBits(GPIOE, GPIO_Pin_0);//
		}
		else if(state==1)//亮
		{
			GPIO_ResetBits(GPIOE, GPIO_Pin_0);					
		}	
	}
	else if(led==1)//LED1 //
	{	
		if(state==0)//灭
		{
			GPIO_SetBits(GPIOE, GPIO_Pin_1);//	
		}
		else if(state==1)//亮
		{
			GPIO_ResetBits(GPIOE, GPIO_Pin_1);	
		}
	}

	else if(led==2)//LED2 //高亮灯 红灯
	{
		if(state==0)//灭
		{
			GPIO_ResetBits(GPIOD, GPIO_Pin_5);	
		}
		else if(state==1)//亮
		{
			GPIO_SetBits(GPIOD, GPIO_Pin_5);//	
		}
	}
}


/* getLED(led) - gets the state of the specified LED
 *
 * It gets the state of the specified LED
 */
uint8_t	WaspUtils::getLED(uint8_t led)
{
	if(led==0)
	{
		if(GPIO_ReadOutputDataBit(GPIOE, GPIO_Pin_0)==0)
			return 1;
		else return 0;
	}
	else if(led==1)
	{
		if(GPIO_ReadOutputDataBit(GPIOE, GPIO_Pin_1)==0)
			return 1;
		else return 0;
	}
	else if(led==2)
	{
		return GPIO_ReadOutputDataBit(GPIOD, GPIO_Pin_5);
	}
	else return 0;
}


/* blinkLEDs(time) - blinks LED, with the specified time of blinking
 *
 * It bliks LED0 and LED1, with the specified time of blinking
 */
//
void WaspUtils::blinkLEDs(uint16_t time) 
{
	setLED(0,1);
	setLED(1,1);
	setLED(2,1);
	delay_ms(time);
	setLED(0,0);
	setLED(1,0);
  setLED(2,0);  
	delay_ms(time);
}


/* map(x,in_min,in_max,out_min,out_max) - maps 'x' from the read range to the specified range
 *
 * It maps 'x' from the read range to the specified range
 *
 * 'in_min' and 'in_max' are the entry range read from the sensor
 * 'out_min' and 'out_max' are the desired output range
 */
//long WaspUtils::map(long x, long in_min, long in_max, long out_min, long out_max)
//{
//	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
//}











void WaspUtils::setMuxAux1()
{

}



/* readEEPROM(address) - reads from the EEPROM specified address
 *
 * It reads from the EEPROM specified address
 * 
 * EEPROM has 512 Bytes of memory
 */
//以下这个函数没有经过验证，有可能错的
uint8_t WaspUtils::readEEPROM(int address)
{


	 int temp;
	 temp = Eeprom.readEEPROM (0xa0,address);
	 if((temp>=0)&&(temp<256))return temp;
	 else return 0xff;

}


/* writeEEPROM(address,value) - writes the specified value into the specified address
 *
 * It writes the specified value into the specified address
 * 
 * EEPROM has 512 Bytes of memory
 */
void WaspUtils::writeEEPROM(int address, uint8_t value)
{
	//eeprom_write_byte((unsigned char *) address, value);
	Eeprom.writeEEPROM (0xa0,address, value);
}

//uint8_t WaspUtils::dec2hex(uint8_t number)
//{
//	int aux=0, aux2=0;
//	aux=number/16;
//	aux2=number-(aux*16);
//	return (aux*10+aux2);
//}


void WaspUtils::hex2str(uint8_t* number, char* macDest)
{
	hex2str(number,macDest,8);
}

void WaspUtils::hex2str(uint8_t* number, char* macDest, uint8_t length)
{
	uint8_t aux_1=0;
	uint8_t aux_2=0;

	for(int i=0;i<length;i++){
		aux_1=number[i]/16;
		aux_2=number[i]%16;
		if (aux_1<10){
			macDest[2*i]=aux_1+'0';
		}
		else{
			macDest[2*i]=aux_1+('A'-10);
		}
		if (aux_2<10){
			macDest[2*i+1]=aux_2+'0';
		}
		else{
			macDest[2*i+1]=aux_2+('A'-10);
		}
	} 
	macDest[length*2]='\0';
}


/*
 Function: Converts a number stored in an array to a long
*/
long WaspUtils::array2long(char* num)
{
	int j=0;
	long resul=0;
	long aux=1;
	uint8_t counter=0;
  
	while( (num[counter]>='0') && (num[counter]<='9') ){
		counter++;
	}
	while( (num[j]>='0') && (num[j]<='9') ){
		for(int a=0;a<counter-1;a++)
		{
			aux=aux*10;
		}
		resul=resul+(num[j]-'0')*aux;
		counter--;
		j++;
		aux=1;
	}
	return resul;
}


/*
 Function: Coverts a long to a number stored in an array
*/
uint8_t WaspUtils::long2array(long num, char* numb)
{
	long aux=num;
	uint8_t i=0;
	
	if( num<0 )
	{
		num = ~(num);
		num+=1;
		numb[i]='-';
		i++;
	}
	aux=num;
	while(aux>=10)
	{
		aux=aux/10;
		i++;
	}
	numb[i+1]='\0';
	aux=num;
	while(aux>=10)
	{
		numb[i]=aux%10 + '0';
		aux=aux/10;
		i--;
	}
	numb[i]=aux + '0';
	return i;
}

/*
  Function: Converts a string to an hex number
  */
uint8_t WaspUtils::str2hex(char* str)
{
	int aux=0, aux2=0;
	
	
	if( (*str>='0') && (*str<='9') )
	{
		aux=*str++-'0';
	}
	else if( (*str>='A') && (*str<='F') )
	{
		aux=*str++-'A'+10;
	}
	if( (*str>='0') && (*str<='9') )
	{
		aux2=*str-'0';
	}
	else if( (*str>='A') && (*str<='F') )
	{
		aux2=*str-'A'+10;
	}
	return aux*16+aux2;
}


/*
  Function: Converts a string to an hex number
  */
uint8_t WaspUtils::str2hex(uint8_t* str)
{
	int aux=0, aux2=0;
	
	
	if( (*str>='0') && (*str<='9') )
	{
		aux=*str++-'0';
	}
	else if( (*str>='A') && (*str<='F') )
	{
		aux=*str++-'A'+10;
	}
	if( (*str>='0') && (*str<='9') )
	{
		aux2=*str-'0';
	}
	else if( (*str>='A') && (*str<='F') )
	{
		aux2=*str-'A'+10;
	}
	return aux*16+aux2;
}

/*
 Function: Generates a decimal number from two ASCII characters which were numbers
 Returns: The generated number
*/
uint8_t WaspUtils::converter(uint8_t conv1, uint8_t conv2)
{
	uint8_t aux=0;
	uint8_t aux2=0;
	uint8_t resul=0;

	switch(conv1)
	{
		case 48: aux=0;
		break;
		case 49: aux=1;
		break;
		case 50: aux=2;
		break;
		case 51: aux=3;
		break;
		case 52: aux=4;
		break;
		case 53: aux=5;
		break;
		case 54: aux=6;
		break;
		case 55: aux=7;
		break;
		case 56: aux=8;
		break;
		case 57: aux=9;
		break;
		case 65: aux=10;
		break;
		case 66: aux=11;
		break;
		case 67: aux=12;
		break;
		case 68: aux=13;
		break;
		case 69: aux=14;
		break;
		case 70: aux=15;
		break;
	}
	switch(conv2)
	{
		case 48: aux2=0;
		break;
		case 49: aux2=1;
		break;
		case 50: aux2=2;
		break;
		case 51: aux2=3;
		break;
		case 52: aux2=4;
		break;
		case 53: aux2=5;
		break;
		case 54: aux2=6;
		break;
		case 55: aux2=7;
		break;
		case 56: aux2=8;
		break;
		case 57: aux2=9;
		break;
		case 65: aux2=10;
		break;
		case 66: aux2=11;
		break;
		case 67: aux2=12;
		break;
		case 68: aux2=13;
		break;
		case 69: aux2=14;
		break;
		case 70: aux2=15;
		break;
		default: aux2=100;
		break;
	}
	if(aux2==100) // Only one character but we have treated two, so We have to fix it
	{
		resul=aux;
	}
	else
	{
		resul=(aux*16)+aux2;
	}
	return resul;
}


void WaspUtils::float2String (float fl, char str[], int N) {

	uint8_t neg = false;

 
	if( fl<0 ){
		neg = true;
		fl*=-1;
	}
 
	float numeroFloat=fl; 
	int parteEntera[10];
	int cifra; 
	long numero=(long)numeroFloat;  
	int size=0;
  
	while(1){
		size=size+1;
		cifra=numero%10;
		numero=numero/10;
		parteEntera[size-1]=cifra; 
		if (numero==0){
			break;
		}
	}

	int indice=0;
	if( neg ){
		indice++;
		str[0]='-';
	}
	for (int i=size-1; i>=0; i--)
	{
		str[indice]=parteEntera[i]+'0'; 
		indice++;
	}

	str[indice]='.';
	indice++;

	numeroFloat=(numeroFloat-(int)numeroFloat);
	for (int i=1; i<=N ; i++)
	{
		numeroFloat=numeroFloat*10;
		cifra= (long)numeroFloat;          
		numeroFloat=numeroFloat-cifra;
		str[indice]=char(cifra)+48;
		indice++;
	}
	str[indice]='\0';
}


WaspUtils Utils = WaspUtils();
