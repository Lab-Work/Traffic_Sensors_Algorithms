/*
  SENSORS CODE V 3.0
  In this code the medians of the 7 sensors are sent and variations are reported in a short a simple way
  Example of a packet: #D8f4886918521921800022722322918584559*
  -In this version, only one value of ambient temperature is sent, not the average
  -Also the sensors d,e,f are the ones facing the same lane as the ultrasound sensor
  -Something different from V2.0 is the fact that the variations are computed with the right median value, not the previous one
*/
#include "allboardinc.h"
#include "MLX620_API.h"
#include <stdio.h>
#include <stdlib.h>
#include "MLX620_API.h"
#include <stdlib.h >

#define BTESTALLSENSORSV4 0
#define BTESTALLSENSORSNULLV4 100
//#define EXAMPLEALLSENSORSV4 BTESTALLSENSORSNULLV4 //Uncomment this to deactivate the code
#define EXAMPLEALLSENSORSV4 BTESTALLSENSORSV4      //Uncommment this to activate the code
#if EXAMPLEALLSENSORSV4 == BTESTALLSENSORSV4

long here = 0;
uint8_t error_MAC=2;

packetXBee* paq_sent;    //Object to create the packet for XBee
uint8_t packet_counter=0;  //Counter for the packets
char MAC_address1[2];
char MAC_address2[2];
bool xbeeWakeUPFlag;
bool XBEEFlagOFF;


char packet_tosend[100];
const uint8_t array_size = 50; //Size of the arrays to store the sensors values
const uint8_t ave_array_size = (array_size/10)*6;
unsigned int slave_add;   //Slave address of the temp sensor being read      
float tempobject;         //Variable to store the temporary value of the Object temperature of the sensor being read
float tempambient;         //Variable to store the temporary value of the Ambient temperature of the sensor being read
uint8_t data_count=0;      //Counter of the values being stored in the sensors array


//Variables to read the Ultrasound sensor
int data_Ult[array_size];
uint8_t  Ult_count=0;
int Ult_median;
int valuesonar;
int flagsonar=2;	
int data_sorted_Ult[array_size];

//Variables to read the Temp sensor a
float data_tempa[array_size];
uint8_t tempa_count = 0;
float tempa_median;

//Variables to read the Temp sensor b
float data_tempb[array_size];
uint8_t tempb_count = 0;
float tempb_median;

//Variables to read the Temp sensor c
float data_tempc[array_size];
uint8_t tempc_count = 0;
float tempc_median;

//Variables to read the Temp sensor d
float data_tempd[array_size];
uint8_t tempd_count = 0;
float tempd_median;

//Variables to read the Temp sensor e
float data_tempe[array_size];
uint8_t tempe_count = 0;
float tempe_median;

//Variables to read the Temp sensor f
float data_tempf[array_size];
uint8_t tempf_count = 0;
float tempf_median;

//Variable to temporary store the sorted data
float data_sorted[array_size];

//Variable to obtain the average value and std dev of the ambient temperature
float tempambient_ave;
float Sum_temp=0;
float data_tempambient[ave_array_size];
uint8_t amb_count=0;
float std_dev;


//Variables used to handle the  Ultrasound variations 

bool cal_done = false;   //Flag to activate or deactivate the calibration
uint16_t max_height_car = 400;  //Maximum height of the car  ON THE STREET THE VALUE SHOULD BE 400
uint16_t min_height_car = 120;  //Minimum height of the car   ON THE STREET THE VALUE SHOLD BE 120
int max_variation;
int first_variation;
uint8_t first_variation_count;
uint8_t last_variation_count;
bool first_time = true;
char variations_Ult_char[14]; //Character array to store the variations report
uint8_t count_vars=0;
bool Ult_full = false;
char char_val[6];
const uint8_t size_reportvar= 7;   //Seven bytes for a variation report (3 for max_value, 2 for first_value, 2 for last_value)


//Variables used to handle the temperature variations

float temp_threshold = 0.5; // Celsius Degress as Threshold   ON THE STREET THE VALUE SHOULD BE 0.5
uint8_t count_vars_tempa=0;  //Count variations of sensor a
char variations_tempa_char[14]; //Character array to store the variations report
uint8_t count_vars_tempb=0;  //Count variations of sensor b
char variations_tempb_char[14]; //Character array to store the variations report
uint8_t count_vars_tempc=0;  //Count variations of sensor c
char variations_tempc_char[14]; //Character array to store the variations report
uint8_t count_vars_tempd=0;  //Count variations of sensor d
char variations_tempd_char[14]; //Character array to store the variations report
uint8_t count_vars_tempe=0;  //Count variations of sensor e
char variations_tempe_char[14]; //Character array to store the variations report
uint8_t count_vars_tempf=0;  //Count variations of sensor f
char variations_tempf_char[14]; //Character array to store the variations report

char value_str[10];
char sl_add_char[20];
char temps[40];
/*
  CMat *matUlt = NULL;
  CMat *matTempa = NULL;
  CMat *matTempb = NULL;
  CMat *matTempc = NULL;
  CMat *matTempd = NULL;
  CMat *matTempe = NULL;
  CMat *matTempf = NULL;
*/

uint8_t ack;      //I2C acknowledge bit
double Ta;        //Ambient Temperature
uint8_t pixIdx;   //pixel index
uint16_t trimReg, confReg;

//=====================Method to send the packet created===================================================

void send_packet(){
	
	paq_sent=(packetXBee*) calloc(1,sizeof(packetXBee));
	paq_sent->mode=BROADCAST;  // BroadCast; you need to update everyone !
	paq_sent->MY_known=0;
	paq_sent->packetID=0x52;  //Think about changing it each time you send
	paq_sent->opt=0;
	xbee802.hops=0;
	xbee802.setOriginParams(paq_sent, "5678", MY_TYPE); // Think about this in the future as well
	xbee802.setDestinationParams(paq_sent, "000000000000FFFF",packet_tosend, MAC_TYPE, DATA_ABSOLUTE);
	xbee802.sendXBee(paq_sent);
	if( !xbee802.error_TX )
	{
		printf(" ok\r\n");//	delay_ms(300);
	}
	else   
	{
		printf("WRG\r\n");//	
	}	   
	free(paq_sent);
	paq_sent=NULL; 
	delay_ms(5);
	
	
}

//=======================Methods to turn on and off the XBee to reduce consumption=========================================

void switchOFFXBee(void)
{
	//broadcastSignalagain("======I will sitch  OFF my XBEE ===========");
	//delay(100);
	GPIO_InitTypeDef  GPIO_InitStructure;
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
	//delay(40);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);
	//delay(80);
	//GPIO_SetBits(GPIOD, GPIO_Pin_10);      //set the pin to high
	GPIO_ResetBits(GPIOD, GPIO_Pin_11);   // set the pin to low.
	//delay(80);
}

void wakeAll()
{
	//====================================
	// Enable XBee port 
	GPIO_InitTypeDef  GPIO_InitStructure;
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);
	// delay(40);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);
	GPIO_SetBits(GPIOD, GPIO_Pin_11);      //set the pin to high
	//delay(80);
	//====================================
	xbeeWakeUPFlag=1;
	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	//delay(80);
	xbee802.ON();
	// I might not need the delay ?!
	//delay(100);
	XBEEFlagOFF=0;
  
	// Check if you need to add RTC.ON
	monitor_offuart3RX();
	//delay(15);
	monitor_onuart3TX();
	//delay(15);
	//beginSerial(115200, 1); 
	beginSerial(115200, 3); 
	//  RTCbianliang.getTime();
	// XBee.println("XBEE:W  at time: ");
	// XBee.println(RTCbianliang.getTime());
 
}


//=======================Method to create the random number to include in the packet=======================================

void setSeed()
{
	xbee802.getOwnMacLow(); // Get 32 lower bits of MAC Address
	xbee802.getOwnMacHigh(); // Get 32 upper bits of MAC Address
  
	uint16_t seed_Pram=0X00;
	seed_Pram=xbee802.sourceMacLow[2]* 0x100;
	seed_Pram=seed_Pram+xbee802.sourceMacLow[3];
	srand(seed_Pram);
    
}


uint16_t random_delay(uint16_t Limit){
	
	uint16_t randomNumber=0;

	randomNumber=((int)(Limit*(rand()/((double)RAND_MAX + 1))));

	if(randomNumber<0){
		randomNumber=randomNumber*(-1);
	}
	return randomNumber;

}


//===================Method to create the packet to be sent================================================

void do_packet(){
	
	char value_char[6]; //Array of characters to store the value number converte into string
	uint8_t packet_pointer=0;
	char tempArray[4]; 
		

		
	//Headers for the packet that will be sent
	packet_tosend[packet_pointer]='#';
	packet_pointer++;
	packet_tosend[packet_pointer]='D';   //Frame ID = 'D' from Sensors
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address1[0];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address1[1];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address2[0];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address2[1];
	packet_pointer++;
	
	//Method to create a random number
	uint16_t packetID;
	packetID = random_delay(999);

	sprintf(tempArray,"%d",packetID);

	if(packetID <10){

        tempArray[2]=tempArray[0];
        tempArray[0]='0';
        tempArray[1]='0';   
	}

	if(9 < packetID && packetID< 100){
		tempArray[2]=tempArray[1];
		tempArray[1]=tempArray[0];
		tempArray[0]='0';
	}

     
	for(int i=0;i<3;i++){
		packet_tosend[packet_pointer]=tempArray[i];
		packet_pointer++;
	}
	
	sprintf(value_char, "%d", Ult_median);
	if(Ult_median<100){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;		
	}else{
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;
	}
	//printf("String ul: %s\n",value_char);
	Utils.float2String(tempa_median,value_char,2);
	if(tempa_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(tempb_median,value_char,2);
	if(tempb_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(tempc_median,value_char,2);
	if(tempc_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(tempd_median,value_char,2);
	if(tempd_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(tempe_median,value_char,2);
	if(tempe_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(tempf_median,value_char,2);
	if(tempf_median<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	slave_add = 0x2e;
	tempambient=readMlx90614AmbientTemp(slave_add);
	Utils.float2String(tempambient,value_char,2);
	if(tempambient_ave<10){
		packet_tosend[packet_pointer] = '0';
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;	
	}else{		
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[1];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	//printf("String: %s\n",value_char);
	Utils.float2String(std_dev,value_char,2);
	if(std_dev>10){
		packet_tosend[packet_pointer] = '9';
		packet_pointer++;
		packet_tosend[packet_pointer] = '9';
		packet_pointer++;
		packet_tosend[packet_pointer] = '9';
		packet_pointer++;	
	}else{
		packet_tosend[packet_pointer] = value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[2];
		packet_pointer++;
		packet_tosend[packet_pointer] = value_char[3];
		packet_pointer++;
	}
	
	//printf("String: %s\n",value_char);
	
	if(count_vars>0){
		
		packet_tosend[packet_pointer] = 'u';
		packet_pointer++;
		
		for(uint8_t j=0;j<count_vars;j++){
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+0];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+1];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+2];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+3];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+4];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+5];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_Ult_char[j*size_reportvar+6];
			packet_pointer++;
			
		}
	}
	
	if(count_vars_tempd>0){
		
		packet_tosend[packet_pointer] = 'd';
		packet_pointer++;
		
		for(uint8_t j=0;j<count_vars_tempd;j++){
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+0];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+1];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+2];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+3];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+4];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+5];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempd_char[j*size_reportvar+6];
			packet_pointer++;
			
			
		}
			
	}
	
	if(count_vars_tempe>0){
		
		packet_tosend[packet_pointer] = 'e';
		packet_pointer++;
		
		for(uint8_t j=0;j<count_vars_tempe;j++){
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+0];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+1];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+2];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+3];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+4];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+5];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempe_char[j*size_reportvar+6];
			packet_pointer++;
			
			
		}
			
	}	

	if(count_vars_tempf>0){
		
		packet_tosend[packet_pointer] = 'f';
		packet_pointer++;
		
		for(uint8_t j=0;j<count_vars_tempf;j++){
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+0];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+1];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+2];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+3];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+4];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+5];
			packet_pointer++;
			packet_tosend[packet_pointer] = variations_tempf_char[j*size_reportvar+6];
			packet_pointer++;
			
			
		}
			
	}

	if((count_vars<1)&&(count_vars_tempd<1)&&(count_vars_tempe<1)&&(count_vars_tempf<1)){


		if(count_vars_tempa>0){
		
			packet_tosend[packet_pointer] = 'a';
			packet_pointer++;
		
			for(uint8_t j=0;j<count_vars_tempa;j++){
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+0];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+1];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+2];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+3];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+4];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+5];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempa_char[j*size_reportvar+6];
				packet_pointer++;
			
			
			}
			
		}
	
		if(count_vars_tempb>0){
		
			packet_tosend[packet_pointer] = 'b';
			packet_pointer++;
		
			for(uint8_t j=0;j<count_vars_tempb;j++){
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+0];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+1];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+2];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+3];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+4];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+5];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempb_char[j*size_reportvar+6];
				packet_pointer++;
			
			
			}
			
		}	

		if(count_vars_tempc>0){
		
			packet_tosend[packet_pointer] = 'c';
			packet_pointer++;
		
			for(uint8_t j=0;j<count_vars_tempc;j++){
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+0];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+1];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+2];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+3];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+4];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+5];
				packet_pointer++;
				packet_tosend[packet_pointer] = variations_tempc_char[j*size_reportvar+6];
				packet_pointer++;
			
			
			}
			
		}


	}	
	
	sprintf(value_char, "%d", packet_counter);
	if(packet_counter<10){
		packet_tosend[packet_pointer]='0';
		packet_pointer++;
		packet_tosend[packet_pointer]=value_char[0];
		packet_pointer++;	 
	}else{
		packet_tosend[packet_pointer]=value_char[0];
		packet_pointer++;
		packet_tosend[packet_pointer]=value_char[1];
		packet_pointer++;
	}
	packet_tosend[packet_pointer]='*';
	packet_pointer++;
	packet_tosend[packet_pointer]='\0';
	packet_pointer++;
	
	
	//printf("Packet: %s\n",packet_tosend);
	
	for(uint8_t j=0;j<14;j++){
		variations_Ult_char[j]='\0';
	}
			
	for(uint8_t j=0;j<14;j++){
		variations_tempa_char[j]='\0';
	}	
		
    for(uint8_t j=0;j<14;j++){
		variations_tempb_char[j]='\0';
	}

    for(uint8_t j=0;j<14;j++){
		variations_tempc_char[j]='\0';
	}
    
    for(uint8_t j=0;j<14;j++){
		variations_tempd_char[j]='\0';
	}
    
    for(uint8_t j=0;j<14;j++){
		variations_tempe_char[j]='\0';
	}

    for(uint8_t j=0;j<14;j++){
		variations_tempf_char[j]='\0';
	}				
	
	//count_vars=0;
	
}

//====================Method to create the packet sent only in the setup===================================

void do_setuppacket(){
	
	uint8_t packet_pointer=0;
	char tempArray[4]; 
		
		
	//Headers for the packet that will be sent
	packet_tosend[packet_pointer]='#';
	packet_pointer++;
	packet_tosend[packet_pointer]='D';   //Frame ID = 'S' from Sensors
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address1[0];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address1[1];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address2[0];
	packet_pointer++;
	packet_tosend[packet_pointer]=MAC_address2[1];
	packet_pointer++;
	
	//Method to create a random number
	uint16_t packetID;
	packetID = random_delay(999);

	sprintf(tempArray,"%d",packetID);

	if(packetID <10){

        tempArray[2]=tempArray[0];
        tempArray[0]='0';
        tempArray[1]='0';   
	}

	if(9 < packetID && packetID< 100){
		tempArray[2]=tempArray[1];
		tempArray[1]=tempArray[0];
		tempArray[0]='0';
	}

     
	for(int i=0;i<3;i++){
		packet_tosend[packet_pointer]=tempArray[i];
		packet_pointer++;
	}
		 
	for(uint8_t i=0;i<27;i++){
		packet_tosend[packet_pointer]='0';
		packet_pointer++;
	}
	
	
	packet_tosend[packet_pointer]='0';
	packet_pointer++;
	packet_tosend[packet_pointer]='0';
	packet_pointer++;
		 
	packet_tosend[packet_pointer]='*';
	packet_pointer++;
	packet_tosend[packet_pointer]='\0';
	packet_pointer++;
	
	
}

//===================Additional Functions to handle the sensors data=======================================

//Method to handle variations
void handle_var_Ult(){
	
	count_vars=0;
	bool Ult_full=false;
	
	
	for(uint8_t i=0;i<array_size;i++){
		
		if(Ult_full){
			break;
		}
		
		if((data_Ult[i]>Ult_median-max_height_car)&&(data_Ult[i]<Ult_median-min_height_car)){
			
			if(first_time){
				max_variation = data_Ult[i];
				last_variation_count = i;
				first_variation_count= i;
				first_time = false;
			}else{
				last_variation_count = i;
				if(data_Ult[i]>max_variation){
					max_variation=data_Ult[i];
				}						 
				
			}
			
		}else if(!first_time){
		
			// variations_Ult_char[0]='u';		REMEMBER TO PUT THE ''u' IN THE DO_PACKET METHOD!!	
			sprintf(char_val,"%d",max_variation);
			if(max_variation<100){
				variations_Ult_char[count_vars*size_reportvar] = '0';
				variations_Ult_char[count_vars*size_reportvar +1] = char_val[0];
				variations_Ult_char[count_vars*size_reportvar +2] = char_val[1];	
			}else{
				variations_Ult_char[count_vars*size_reportvar] = char_val[0];
				variations_Ult_char[count_vars*size_reportvar +1] = char_val[1];
				variations_Ult_char[count_vars*size_reportvar +2] = char_val[2];
			}
			for(uint8_t i=0;i<5;i++){
				char_val[i]=' ';	
			}
		
			sprintf(char_val,"%d",first_variation_count);
			if(first_variation_count<10){
				variations_Ult_char[count_vars*size_reportvar +3] = '0';
				variations_Ult_char[count_vars*size_reportvar +4] = char_val[0];
			}else{
				variations_Ult_char[count_vars*size_reportvar +3] = char_val[0];
				variations_Ult_char[count_vars*size_reportvar +4] = char_val[1];
			}
			for(uint8_t i=0;i<5;i++){
				char_val[i]=' ';	
			}
		
			sprintf(char_val,"%d",last_variation_count);
			if(last_variation_count<10){
				variations_Ult_char[count_vars*size_reportvar +5] = '0';
				variations_Ult_char[count_vars*size_reportvar +6] = char_val[0];
			}else{
				variations_Ult_char[count_vars*size_reportvar +5] = char_val[0];
				variations_Ult_char[count_vars*size_reportvar +6] = char_val[1];
			}
			for(uint8_t i=0;i<5;i++){
				char_val[i]=' ';	
			}
		
			count_vars++;
			first_time = true;
		
			if(count_vars>1){
				Ult_full = true;
			
			}
			
			
			
		}
		
	}
	
	

	
	//printf("%s\n",variations_Ult_char);
	
}
	
//Method to sort an array 
void sort_array(int array[]){
	
	int temp;
		
	for(uint16_t i=0;i<array_size;i++){
		data_sorted_Ult[i] = array[i];
	}
	
	for(uint16_t u=0;u<array_size;u++){
		for(uint16_t v=u+1;v<array_size;v++){
			if(data_sorted_Ult[u]>data_sorted_Ult[v]){
				temp = data_sorted_Ult[v];
				data_sorted_Ult[v] = data_sorted_Ult[u];
				data_sorted_Ult[u] = temp;
			}
		}
	}
	
}

//Method 	to create the median packet
void 	do_median_Ult(){
	
	sort_array(data_Ult);
	if(array_size%2==0){
		Ult_median = (data_sorted_Ult[array_size/2 -1] + data_sorted_Ult[array_size/2])/2;
	}else{
		Ult_median = data_sorted_Ult[(array_size-1)/2];
	}
	
}

//Main method to read the Ultrasound and take decisions based on the value
void read_Ultrasound(){

	flagsonar = 2;
	while(flagsonar!=0){
		flagsonar=readMb7076(&valuesonar);
		if(flagsonar==3){
			break;
		}
	}

	
	//Condition to limit the distance value to maximum of 999	
	if(valuesonar>999){        
		valuesonar=999;
	}
	

	data_Ult[data_count] = valuesonar;
	 	
}

//================================Additional functions to handle the temperature data===================================================	

//Method to handle the variations of the temperature sensor

uint8_t handle_var_temp(float data_temp[],float  temp_median, char variations_temp_char[]){
	
	uint8_t count_vars_temp=0;
	bool temp_full=false;
	bool first_time_temp=true;
	float max_variation_temp;
	uint8_t first_variation_count_temp;
	uint8_t last_variation_count_temp;
	char char_value[6];
	
	
	for(uint8_t i=0;i<array_size;i++){
		
		if(temp_full){
			break;
		}
		
		
		if((data_temp[i]>temp_median+temp_threshold)||(data_temp[i]<temp_median-temp_threshold)){
			
			if(first_time_temp){
				max_variation_temp = data_temp[i];
				last_variation_count_temp = i;
				first_variation_count_temp= i;
				first_time_temp = false;
			}else{
				last_variation_count_temp = i;
				if(data_temp[i]>max_variation_temp){
					max_variation_temp=data_temp[i];
				}						 
				
			}
			
		}else if(!first_time_temp){
		
			// variations_Ult_char[0]='u';		REMEMBER TO PUT THE ''u' IN THE DO_PACKET METHOD!!	
			
			Utils.float2String(max_variation_temp,char_value,2);
			if(max_variation_temp<10){
				variations_temp_char[count_vars_temp*size_reportvar] = '0';
				variations_temp_char[count_vars_temp*size_reportvar +1] = char_value[0];
				variations_temp_char[count_vars_temp*size_reportvar +2] = char_value[2];		
				
			}else{
				variations_temp_char[count_vars_temp*size_reportvar] = char_value[0];
				variations_temp_char[count_vars_temp*size_reportvar +1] = char_value[1];
				variations_temp_char[count_vars_temp*size_reportvar +2] = char_value[3];
			}
		
			for(uint8_t i=0;i<5;i++){
				char_value[i]=' ';	
			}
		
			sprintf(char_value,"%d",first_variation_count_temp);
			if(first_variation_count_temp<10){
				variations_temp_char[count_vars_temp*size_reportvar +3] = '0';
				variations_temp_char[count_vars_temp*size_reportvar +4] = char_value[0];
			}else{
				variations_temp_char[count_vars_temp*size_reportvar +3] = char_value[0];
				variations_temp_char[count_vars_temp*size_reportvar +4] = char_value[1];
			}
			for(uint8_t i=0;i<5;i++){
				char_value[i]=' ';	
			}
		
			sprintf(char_value,"%d",last_variation_count_temp);
			if(last_variation_count_temp<10){
				variations_temp_char[count_vars_temp*size_reportvar +5] = '0';
				variations_temp_char[count_vars_temp*size_reportvar +6] = char_value[0];
			}else{
				variations_temp_char[count_vars_temp*size_reportvar +5] = char_value[0];
				variations_temp_char[count_vars_temp*size_reportvar +6] = char_value[1];
			}
			for(uint8_t i=0;i<5;i++){
				char_value[i]=' ';	
			}
		
			count_vars_temp++;
			first_time_temp = true;
		
			if(count_vars_temp>1){
				temp_full = true;
			
			}
			
			
			
		}
		
	}
	
	
	return count_vars_temp;
	
}

//Method to obtain the std deviation

float get_std_dev(){
	
	float sum=0.0;
	float meanOfSquares, result, squareOfMean;
	
	for(uint8_t i=0;i<ave_array_size;i++){
		
		sum += data_tempambient[i]*data_tempambient[i];
	}
	
	meanOfSquares = sum/(ave_array_size-1);
	
	squareOfMean = (tempambient_ave*tempambient_ave*ave_array_size)/(ave_array_size-1);
	
	result = sqrt(meanOfSquares - squareOfMean);
	
	return result;
	
}

//Method to sort an array of floats
void sort_array_float(float array[]){
	
	float temp;
	
	for(uint16_t i=0;i<array_size;i++){
		data_sorted[i] = array[i];
	}
	
	for(uint16_t u=0;u<array_size;u++){
		for(uint16_t v=u+1;v<array_size;v++){
			if(data_sorted[u]>data_sorted[v]){
				temp = data_sorted[v];
				data_sorted[v] = data_sorted[u];
				data_sorted[u] = temp;
			}
		}
	}
	
}

float do_median_temp(float array_temp[]){
	
	float median;
	sort_array_float(array_temp);
	if(array_size%2==0){
		median = (data_sorted[array_size/2 -1] + data_sorted[array_size/2])/2;
	}else{
		median = data_sorted[(array_size-1)/2];
	}
	return median;
}
	
void read_Tempsensor(float data_temp[]){
	
	tempobject=readMlx90614ObjectTemp(slave_add);
	data_temp[data_count] = tempobject;

	
	if(data_count%10 == 0){
		tempambient=readMlx90614AmbientTemp(slave_add);
		data_tempambient[amb_count] = tempambient;
		Sum_temp += tempambient;
		amb_count++;
	}		
}

//===================================================================================================

/**
 * \file MLX620_Demo.C
 * \brief MLX90620 API functions demonstration usage.
 * \details This demonstration usage of the API functions performing the most necessary operations in order to get full frame of the sensor printed out using standard 'printf' function.\n
 * The users of this demo should implement the 'printf' function for the particular MCU and compiler that is used. If the printing functionality is not needed, it should be disabled.
 * This demo is performing the following operations:
 * - initializing the sensor and reporting (printing) the error
 * - printing the values of the "Configuration" and "Trimming" Registers, after the initialization is done
 * - reading raw IR temperature data for each pixel, as well as reading Ambient Temperature sensor
 * - compensating the printing the Ambient Temperature [Kelvin]
 * - compensating the printing the Object's Temperature for each pixel from 1 to 64
 */

/** \fn uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf)
 * \brief Initialize the sensor.
 * \details It should be used only when the sensor is supplied.
 * \param[out] *trim Trimming register value after the configuration is done.
 * \param[out] *conf Configuration register value after the configuration is done.
 * \retval ack I2C acknowledge bit.
 */
uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf);

/** \fn uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta)
 * \brief Read measurement data from the sensor and calculate ambient and Infra Red (object's) temperature in Kelvin.
 * \details The temperature results for each pixel is saved in .\n
 * \param[in] *pIRtempK Pointer to buffer where the temperature results for each pixel would be stored
 * \param[out] *Ta Ambient temperature in Kelvin
 * \retval ack I2C acknowledge bit.
 */
uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta);


uint8_t MLX90620_InitializeSensor(uint16_t *trim, uint16_t *conf)
{
	uint8_t ack;

	ack = MLX620_Initialize();      //initialize the sensor
	if (ack == MLX620_ACK)
	{
		ack = MLX620_ReadTrim(trim);    //read the Trimming register and return it
		ack |= MLX620_ReadConfig(conf); //read the Configuration register and return it
	}

	return ack;
}

uint8_t MLX90620_MeasureTemperature(double *pIRtempK, double *Ta)
{
	uint8_t ack;
	int16_t ptat, tgc;

	//get RAW (not compensated) ambient temperature sample (PTAT sensor)
	ack = MLX620_ReadRAM(MLX620_RAM_PTAT, 0, 1, (uint8_t*)&ptat);

	if (ack == MLX620_ACK)
	{
		//compensate ambient temperature; get absolute temperature in Kelvin
		*Ta = MLX620_GetTaKelvin (ptat);

		ack = MLX620_ReadRAM(MLX620_RAM_TGC, 0, 1, (uint8_t*)&tgc);

		if (ack == MLX620_ACK)
		{
			MLX620_CalcTGC(tgc);

			ack = MLX620_ReadRAM(MLX620_RAM_IR_BEG, 1, MLX620_IR_SENSORS, (uint8_t*)MLX620_RAMbuff);

			if (ack == MLX620_ACK)
			{
				MLX620_CompensateIR((int16_t*)MLX620_RAMbuff, MLX620_RAM_IR_BEG, 1, MLX620_IR_SENSORS, pIRtempK);

			}
		}
	}
	return ack;
}

//====================================================================================================
// Setup and loop methods	

void setup()
{
	SysClkWasp=32000000;  //Clk changed to 32MHz. You can also try other frequency but better higher than 30MHz 
	SysPreparePara();
	zppSystemInit(); 

	Utils.setLED(0, 0);
	Utils.setLED(1, 0);
	Utils.setLED(2, 0);

	//Mux_poweron();
	monitor_onuart3TX();  monitor_offuart3RX();
	beginSerial(115200, PRINTFPORT);//
	beginMb7076();	
	SMBus_Init();
	SMBus_Apply();
	
	//Timer3_Init(1,1000);
   	xbee802.init(XBEE_802_15_4,FREQ2_4G,PRO);
	xbee802.ON();
	error_MAC = xbee802.getOwnMac(); 	

	//MAC address
	sprintf(MAC_address1,"%x",xbee802.sourceMacLow[2]);
	sprintf(MAC_address2,"%x",xbee802.sourceMacLow[3]);
	
	//printf("%s",MAC_address1);
	//printf("%s\n",MAC_address2);
	
	setSeed();
	
	do_setuppacket();
	send_packet();
	switchOFFXBee();
	
	delay_ms(20);
	IWDG_Init(IWDG_Prescaler_64,4000);
	IWDG_Feed();
	
	ack = MLX90620_InitializeSensor(&trimReg, &confReg);

	if (ack == MLX620_ACK)
	{
		//printf("Sensor initialized successfully\n")
		//printf("Triming Register = %X\n, trimReg");
		//printf("Configuration Register = %X\n, confReg");
	}
	else
	{
		//printf("ERROR: Sensor initiazation failed!\n");
	}
	//Method to turn on the battery monitor
	//PWR2745.switchesOFF(PWR_XBEE|PWR_SD|PWR_SENS_3V3|PWR_SENS1_5V|PWR_SENS2_5V|PWR_SENS3_5V|PWR_MUX_UART6);
	
	//Functions to turn on the battery monitor
	/*
	  PWR2745.initPower(PWR_BAT|PWR_RTC);
	  PWR2745.switchesON(PWR_BAT|PWR_RTC);
	
	  PWR2745.initBattery();

	  int16_t tempb;

	  tempb=PWR2745.getBatteryVolts();
	  printf("\r\n");
	  printf("Voltage=%dmv  ",tempb);
	  delay_ms(50);
	
	  tempb=PWR2745.getBatteryCurrent();
	  printf("Current=%dma  \n",tempb);
	*/
}


void loop()
{
	if(ack == MLX620_ACK)
	{
	    ack = MLX90620_MeasureTemperature(IRtempK, &Ta);
	    if(ack == MLX620_ACK)
	    {
			printf("Ambient Temperature = %4.2 \n", Ta);
			for(pixIdx = 0; pixIdx < MLX620_IR_SENSORS; pixIdx++)
				printf("Infrared Temperature pixel[%u] = %4.2 \n", pixIdx, IRtempK[pixIdx]);
	    }
	    else
			printf("ERROR: Reading data from the sensor failed!\n");
	}
	else
	    return 0;
	MLX620_I2C_Driver_NOPdelay(10000);   //wait some time	
}
#endif //end EEPROM
