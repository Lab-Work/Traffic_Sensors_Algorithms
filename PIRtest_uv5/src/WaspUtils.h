/*! \file WaspUtils.h
    \brief Library containing useful general functions
    
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
  
    Version:		0.12

    Design:		David Gasc n

    Implementation:	Alberto Bielsa, David Cuartielles

*/
  
  
/*! \def Wasputils_h
    \brief The library flag
    
 */
#ifndef Wasputils_h
#define Wasputils_h

/******************************************************************************
 * Includes
 ******************************************************************************/
 
#include <inttypes.h>

//#include <avr/eeprom.h>

/******************************************************************************
 * Definitions & Declarations
 ******************************************************************************/

/*! \def MAX_ARGS
    \brief max amount of arguments in Wasp
 */
#define MAX_ARGS 20

/*! \def MAX_ARG_LENGTH
    \brief max length for Wasp arguments
 */
#define MAX_ARG_LENGTH 16

/*! \def LED_ON
    \brief sets LED ON
 */
#define	LED_ON	1

/*! \def LED_OFF
    \brief sets LED OFF
 */
#define	LED_OFF	0

/*! \def MUX_TO_HIGH
    \brief sets mux high
 */
#define	MUX_TO_HIGH	1

/*! \def MUX_TO_LOW
    \brief sets mux low
 */
#define	MUX_TO_LOW	0


/******************************************************************************
 * Class
 ******************************************************************************/
 
//! WaspUtils Class
/*!
	WaspUtils Class defines all the variables and functions used to set LEDs, multiplexor and useful general functions
 */
class WaspUtils
{
  private:

  public:
  
  //! Variable : bidimensional array of arguments in Waspmote, mainly used with the GPS
  /*!
   */ 
//  char arguments[MAX_ARGS][MAX_ARG_LENGTH];

  //! class constructor
  /*!
  It does nothing
  \param void
  \return void
  */
  WaspUtils(void);


void	initLEDs(void);
  //! It sets the specified LED to the specified state(ON or OFF)
  /*!
  \param uint8_t led : the LED to set ON/OFF
  \param uint8_t state : the state to set the LED
  \return void
  \sa getLED(uint8_t led), blinkLEDs(uint16_t time)
   */
  void	setLED(uint8_t led, uint8_t state);
  
  //! It gets the state of the specified LED
  /*!
  \param uint8_t led : the LED to get the state
  \return the state of the LED
  \sa setLED(uint8_t led, uint8_t state), blinkLEDs(uint16_t time)
   */
  uint8_t getLED(uint8_t led);
  
  //! It blinks LEDs, with the specified time for blinking
  /*!
  \param uint16_t time : time for blinking
  \return void
  \sa setLED(uint8_t led, uint8_t state), getLED(uint8_t led)
   */
  void blinkLEDs(uint16_t time);
  
  //! It maps 'x' from the read range to the specified range
  /*!
  \param long x : value to map
  \param long in_min : minimum input value for 'x'
  \param long in_max : maximum input value for 'x'
  \param long out_min : minimum output value for 'x'
  \param long out_max : maximum output value for 'x'
  \return the value 'x' mapped to the [out_min,out_max] range
   */
//  long map(long x, long in_min, long in_max, long out_min, long out_max);

  //! It sets MUX to the desired combination
  /*!
  	It sets MUX to the desired combination. Possible combinations are:
  
  	MUX_LOW = 0 & MUX_HIGH = 1 ---> GPS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 1 ---> GPRS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 0 ---> AUX1 MODULE
  	MUX_LOW = 0 & MUX_HIGH = 0 ---> AUX2 MODULE
  
  \param uint8_t MUX_LOW : low combination part
  \param uint8_t MUX_HIGH : high combination part
  \return void
  \sa setMuxGPS(), setMuxGPRS(), setMuxAux1(), setMuxAux2()
   */
//  void setMux(uint8_t MUX_LOW, uint8_t MUX_HIGH);
  
  //! It sets MUX to the desired combination (0,1) to enable GPS module
  /*!
  	Possible combinatios are:
  
  	MUX_LOW = 0 & MUX_HIGH = 1 ---> GPS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 1 ---> GPRS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 0 ---> AUX1 MODULE
  	MUX_LOW = 0 & MUX_HIGH = 0 ---> AUX2 MODULE
  
  \return void
  \sa setMux(uint8_t MUX_LOW, uint8_t MUX_HIGH), setMuxGPRS(), setMuxAux1(), setMuxAux2()
   */
//  void setMuxGPS();
  
  //! It sets MUX to the desired combination (1,1) to enable GPRS module
  /*!
  	Possible combinatios are:
  
  	MUX_LOW = 0 & MUX_HIGH = 1 ---> GPS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 1 ---> GPRS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 0 ---> AUX1 MODULE
  	MUX_LOW = 0 & MUX_HIGH = 0 ---> AUX2 MODULE
  
  \return void
  \sa setMux(uint8_t MUX_LOW, uint8_t MUX_HIGH), setMuxGPS(), setMuxAux1(), setMuxAux2()
   */
//  void setMuxGPRS();
  
  //! It sets MUX to the desired combination (1,0) to enable AUX1 module
  /*!
  	Possible combinatios are:
  
  	MUX_LOW = 0 & MUX_HIGH = 1 ---> GPS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 1 ---> GPRS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 0 ---> AUX1 MODULE
  	MUX_LOW = 0 & MUX_HIGH = 0 ---> AUX2 MODULE
  
  \return void
  \sa setMux(uint8_t MUX_LOW, uint8_t MUX_HIGH), setMuxGPS(), setMuxGPRS(), setMuxAux2()
   */
  void setMuxAux1();
  
  //! It sets MUX to the desired combination (0,0) to enable AUX2 module
  /*!
  	Possible combinatios are:
  
  	MUX_LOW = 0 & MUX_HIGH = 1 ---> GPS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 1 ---> GPRS MODULE
  	MUX_LOW = 1 & MUX_HIGH = 0 ---> AUX1 MODULE
  	MUX_LOW = 0 & MUX_HIGH = 0 ---> AUX2 MODULE
  
  \return void
  \sa setMux(uint8_t MUX_LOW, uint8_t MUX_HIGH), setMuxGPS(), setMuxGPRS(), setMuxAux1()
   */
//  void setMuxAux2();

  //! It reads a value from the specified EEPROM address
  /*!
  \param int address : EEPROM address to read from
  \return the value read from EEPROM address
  \sa writeEEPROM(int address, uint8_t value)
   */
  uint8_t readEEPROM(int address);
  
  //! It writes the specified value to the specified EEPROM address
  /*!
  \param int address : EEPROM address to write to
  \param uint8_t value: value to write to the EEPROM
  \return void
  \sa readEEPROM(int address)
   */
  void writeEEPROM(int address, uint8_t value);
  
  //! It gets a number out of a string. It gets till the 2nd decimal of the number.
  /*!
  \param char* str: string containing the number to extract
  \return the number extracted from the string
  \sa parse_degrees(char* str), gpsatol(char* str), gpsisdigit(char c), parse_latitude(char* str)
   */
//  long parse_decimal(char *str);
  
  //! It gets the degree number out of a string
  /*!
  \param char* str: string containing the number to extract
  \return the number extracted from the string
  \sa parse_decimal(char* str), gpsatol(char* str), gpsisdigit(char c), parse_latitude(char* str)
   */
//  double parse_degrees(char *str);
  
  //! It gets the integer part of a number out of a string
  /*!
  \param char* str: string containing the number to extract
  \return the number extracted from the string
  \sa parse_decimal(char* str), parse_degrees(char* str), gpsisdigit(char c), parse_latitude(char* str)
   */
//  long gpsatol(char *str);
  
  //! It checks if the char is a digit or not
  /*!
  \param char c: character to determine if it is a digit or not
  \return TRUE if is a digit, FALSE if not
  \sa parse_decimal(char* str), parse_degrees(char* str), gpsatol(char* str), parse_latitude(char* str)
   */
  bool gpsisdigit(char c) { return c >= '0' && c <= '9'; };
  
  //! It parses latitude or longitude, getting the number out of a string
  /*!
  \param char* str: string containing the latitude or longitude to parse
  \return the latitude or longitude extracted from the string
  \sa parse_decimal(char* str), parse_degrees(char* str), gpsatol(char* str), gpsisdigit(char c)
   */
//  double parse_latitude(char *str);
  
  //! It converts a decimal number into an hexadecimal number
  /*!
  \param uint8_t number: number to convert
  \return the number converted to hexadecimal
  \sa array2long(char* num), long2array(long num, char* numb), str2hex(char* str), str2hex(uint8_t* str)
   */
//  uint8_t dec2hex(uint8_t number);
  
  //! It converts a number stored in an array into a decimal number
  /*!
  \param char* num : string that contains the number to extract
  \return the number extracted
  \sa dec2hex(uint8_t num), long2array(long num, char* numb), str2hex(char* str), str2hex(uint8_t* str)
   */
  long array2long(char* num);
  
  //! It converts a decimal number into a string
  /*!
  \param long num : number to convert
  \param char* numb : string where store the converted number
  \return the number of digits of the number
  \sa array2long(char* num), dec2hex(uint8_t num), str2hex(char* str), str2hex(uint8_t* str)
   */
  uint8_t long2array(long num, char* numb);
  
  //! It converts a number stored in a string into a hexadecimal number
  /*!
  \param char* str : string where the number is stored
  \return the converted number
  \sa array2long(char* num), dec2hex(uint8_t num), long2array(long num, char* numb), str2hex(uint8_t* str)
   */
  uint8_t str2hex(char* str);
  
  //! It converts a number stored in a string into a hexadecimal number
  /*!
  \param char* str : string where thember is stored
  \return the converted number
  \sa array2long(char* num), dec2hex(uint8_t num), long2array(long num, char* numb), str2hex(char* str)
   */
  uint8_t str2hex(uint8_t* str);
  
  //! It converts a hexadecimal number stored in an array to a string (8 Byte numbers)
  /*!
  \param uint8_t* number : hexadecimal array to conver to a string
  \param const char* macDest : char array where the converted number is stored
  \return void
  \sa array2long(char* num), dec2hex(uint8_t num), long2array(long num, char* numb), str2hex(char* str), str2hex(uint8_t* str)
   */
  void hex2str(uint8_t* number, char* macDest);
  
   //! It converts a hexadecimal number stored in an array to a string (8 Byte numbers)
  /*!
  \param uint8_t* number : hexadecimal array to conver to a string
  \param const char* macDest : char array where the converted number is stored
  \param uint8_t length : length to copy
  \return void
  \sa array2long(char* num), dec2hex(uint8_t num), long2array(long num, char* numb), str2hex(char* str), str2hex(uint8_t* str)
   */
  void hex2str(uint8_t* number, char* macDest, uint8_t length);
  
  //! It converts a number stored in an array into a decimal number
  /*!
  \param const char* str : string that contains the number to extract
  \return the number extracted
  \sa sizeOf(const char* str), strCmp(const char* str1, const char* str2, size), strCp(char* str1, char* str2)
   */
//  uint32_t strtolong(const char* str);
  
  //! It gets the size of a string or array
  /*!
  \param const char* str : string to get the size from
  \return the array or string size
  \sa strtolong(const char* str), strCmp(const char* str1, const char* str2, size), strCp(char* str1, char* str2)
   */
  int sizeOf(const char* str);
  
  //! It compares two strings
  /*!
  \param const char* str1 : string to compare
  \param const char* str2 : string to compare
  \param uint8_t size : string size
  \return '0' if strings are equal, '1' if don't
  \sa strtolong(const char* str), sizeOf(const char* str), strCp(char* str1, char* str2)
   */
  uint8_t strCmp(const char* str1, const char* str2, uint8_t size);
  
  //! It copies one string into another
  /*!
  	- str1: origin
  	- str2: target
  \param char* str1 : origin string
  \param char* str2 : target string
  \return void
  \sa strtolong(const char* str), sizeOf(const char* str), strCmp(const char* str1, const char* str2, uint8_t size)
   */
//  void strCp(char* str1, char* str2);

  //! It clears the arguments[][] data matrix
  /*!
  \param void
  \return void
   */
//  void clearArguments(void);  

  
  //! It breaks a string into its arguments separated by "separators". The pieces are stored in 'arguments' array
  /*!
  \param const char* str : string to separate
  \param char separator : the separator used to separate the string in pieces
  \return void
  \sa clearArguments(),clearBuffer()
   */
//  void strExplode(const char* str, char separator);
  
  //! It generates a decimal number from two ASCII characters which were numbers
  /*!
  \param uint8_t conv1 : the ASCII number first digit to convert
  \param uint8_t conv2 : the ASCII number second digit to convert
  \return the converted number
  */
  uint8_t converter(uint8_t conv1, uint8_t conv2);
  
  //! It converts a float into a string
  /*!
  \param float fl : the float to convert
  \param char str[] : the string where store the float converted
  \param int N : the number of decimals
  \return void
   */
  void float2String(float fl, char str[], int N);
  
};

extern WaspUtils Utils;

#endif
