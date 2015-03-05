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
 *  Version:		0.1
 *  Design:		David Gasc贸n
 *  Implementation:	David Cuartielles, Alberto Bielsa
 */
 

#ifndef __WPROGRAM_H__
  #include "WaspClasses.h"
#endif

// Constructors ////////////////////////////////////////////////////////////////

WaspXBee::WaspXBee()
{
//    _uart = 3;
}

// Public Methods //////////////////////////////////////////////////////////////
//开启串口3，开模块Xbee模块供电,开启time3 为了中断1ms
void WaspXBee::begin()
{
//  beginxbee(XBEE_RATE, _uart);
  beginxbee(XBEE_RATE, XBEE_UART);
}

//只是关了Xbee模块供电，但是串口3没有关,也没有关time3
void WaspXBee::close()
{
//  closeSerial(_uart);
	closeSerial(XBEE_UART);
}
//不知道干嘛用的,没改
void WaspXBee::setMode(uint8_t mode)
{
  _pwrMode = mode;
  // set the GPS in the defined power mode
  switch (_pwrMode)
  {
  case XBEE_ON:
	begin();
//	digitalWrite(XBEE_PW,HIGH);

    break;

  case XBEE_OFF:
//	digitalWrite(XBEE_PW,LOW);
	close();
    break;

  default:
	break;
  }
}





//没有改，和以前保持一致
uint8_t WaspXBee::available()
{
//  return serialAvailable( _uart);
  return serialAvailable(XBEE_UART);
}
//没有改，和以前保持一致
int WaspXBee::read()
{
//  return serialRead(_uart);
  return serialRead(XBEE_UART);
}

//
uint8_t WaspXBee::Rx80available()
{
  return serialRx80Available();
}
//
int WaspXBee::Rx80read()
{
  return serialRx80Read();
}



//这个是作者增加的，觉得添加一个读字符串的比较好，不知道客户觉得呢？
int WaspXBee::readstr(uint8_t *str, uint8_t len)
{
//  return serialReadstr(str, len,_uart);
  return serialReadstr(str, len,XBEE_UART);
}



//没有改，和以前保持一致
void WaspXBee::flush()
{
	printf("\r\nflush\r\n");
  //serialFlush(_uart);
  serialFlush(XBEE_UART);
}
//查询方式发送一个自己
void WaspXBee::print(char c)
{
  //printByte(c,  _uart);
  printByte(c,  XBEE_UART);
}
//中断方式发送字符串
void WaspXBee::print(const char c[])
{
  //printString(c,  _uart);
  printString(c,  XBEE_UART);

  ////假如不喜欢中断方式，只要把 printString(c,  _uart)函数里面的定义改为查询方式即可，具体操作看这个函数的定义
}

//void WaspXBee::print(uint8_t b)
//{
//  printByte(b,  _uart);
//}
//
//void WaspXBee::print(int n)
//{
//  print((long) n);
//}
//
//void WaspXBee::print(unsigned int n)
//{
//  print((unsigned long) n);
//}
//
//没有改，和以前保持一致
void WaspXBee::print(long n)
{
  if (n < 0) {
    print('-');
    n = -n;
  }
  printNumber(n, 10);
}
//
//void WaspXBee::print(unsigned long n)
//{
//  printNumber(n, 10);
//}
//
//void WaspXBee::print(double n)
//{
//	printFloat(n, 10);
//}
//没有改，和以前保持一致
void WaspXBee::print(long n, int base)
{
  if (base == 0)
    print((char) n);
  else if (base == 10)
    print(n);
  else
    printNumber(n, base);
}

void WaspXBee::printstr(char *str, unsigned int len)
{
	unsigned int i;
	for(i=0;i<len;i++)
		print(*(str+i));
}

//没有改，和以前保持一致
void WaspXBee::println()
{
  print('\r');
  print('\n');  
}

//void WaspXBee::println(char c)
//{
//  print(c);
//  println();  
//}
//没有改，和以前保持一致
void WaspXBee::println(const char c[])
{
  print(c);
  println();
}

//void WaspXBee::println(uint8_t b)
//{
//  print(b);
//  println();
//}
//
//void WaspXBee::println(int n)
//{
//  print(n);
//  println();
//}
//
//void WaspXBee::println(long n)
//{
//  print(n);
//  println();  
//}
//
//void WaspXBee::println(unsigned long n)
//{
//  print(n);
//  println();  
//}
//
//没有改，和以前保持一致
void WaspXBee::println(long n, int base)
{
  print(n, base);
  println();
}
//
//void WaspXBee::println(double n)
//{
//	print(n);
//	println();
//}

// Private Methods /////////////////////////////////////////////////////////////
//这个和以前保持一致，但是里面调用的程序改了下，改成中断发送方式
void WaspXBee::printNumber(unsigned long n, uint8_t base)
{
//  printIntegerInBase(n, base,  _uart);
  printIntegerInBase(n, base,  XBEE_UART);
}

//void WaspXBee::printFloat(double number, uint8_t digits) 
//{
//  // Handle negative numbers
//  if (number < 0.0)
//  {
//     print('-');
//     number = -number;
//  }
//
//  // Round correctly so that print(1.999, 2) prints as "2.00"
//  double rounding = 0.5;
//  for (uint8_t i=0; i<digits; ++i)
//    rounding /= 10.0;
//  
//  number += rounding;
//
//  // Extract the integer part of the number and print it
//  unsigned long int_part = (unsigned long)number;
//  double remainder = number - (double)int_part;
//  print(int_part);
//
//  // Print the decimal point, but only if there are digits beyond
//  if (digits > 0)
//    print("."); 
//
//  // Extract digits from the remainder one at a time
//  while (digits-- > 0)
//  {
//    remainder *= 10.0;
//    int toPrint = int(remainder);
//    print(toPrint);
//    remainder -= toPrint; 
//  }
//}

// Preinstantiate Objects //////////////////////////////////////////////////////

WaspXBee XBee = WaspXBee();

