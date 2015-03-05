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
 *  Version:		0.2
 *  Design:		David Gascón
 *  Implementation:	Alberto Bielsa
 */


#ifndef __WPROGRAM_H__
	#include "WaspClasses.h"
#endif

void	WaspXBee802::init(uint8_t protocol_used, uint8_t frequency, uint8_t model_used)
{
	protocol=protocol_used;
	freq=frequency;
	model=model_used;

	totalFragmentsReceived=0;
	pendingPackets=0;
	pos=0;
	discoveryOptions=0x00;
	awakeTime[0]=AWAKE_TIME_802_15_4_H;
	awakeTime[1]=AWAKE_TIME_802_15_4_L;
	sleepTime[0]=SLEEP_TIME_802_15_4_H;
	sleepTime[1]=SLEEP_TIME_802_15_4_L;
	scanTime[0]=SCAN_TIME_802_15_4;
	scanChannels[0]=SCAN_CHANNELS_802_15_4_H;
	scanChannels[1]=SCAN_CHANNELS_802_15_4_L;
	encryptMode=ENCRYPT_MODE_802_15_4;
	powerLevel=POWER_LEVEL_802_15_4;
	timeRSSI=TIME_RSSI_802_15_4;
	sleepOptions=SLEEP_OPTIONS_802_15_4;
	retries=0;
	delaySlots=0;
	macMode=0;
	energyThreshold=0x2C;
	counterCCA[0]=0x00;
	counterCCA[1]=0x00;
	counterACK[0]=0x00;
	counterACK[1]=0x00;

        counter=0;
	data_length=0;
	it=0;
	start=0;
	finish=0;
	add_type=0;
	mode=0;
	frag_length=0;
	TIME1=0;
	nextIndex1=0;
	frameNext=0;
	replacementPolicy=XBEE_OUT;
	indexNotModified=1;
	error_AT=2;
	error_RX=2;
	error_TX=2;
	clearFinishArray();
	clearCommand();
}


/*
 Function: Set the maximum number of retries to execute in addition to the
           three retries defined in the 802.15.4 protocol
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the RR command
 Parameters:
   retry: number of retries (0-6)
*/
//uint8_t WaspXBee802::setRetries(uint8_t retry)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(set_retries_802,retry);
//    gen_checksum(set_retries_802);
//    error=gen_send(set_retries_802);
//    
//
//    if(!error)
//    {
//        retries=retry;
//    }
//    return error;
//}

/*
 Function: Get the retries that specifies the RR command
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "retries" variable the number of retries
*/
//uint8_t WaspXBee802::getRetries()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_retries_802);
//    error=gen_send(get_retries_802);
//    
//
//    if(!error)
//    {
//        retries=data[0];
//    }
//    return error;
//}

/*
 Function: Set the minimun value of the back-off exponent in CSMA/CA
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the RN command
 Parameters:
   exponent: value of the back-off exponential (0-3)
*/
//uint8_t WaspXBee802::setDelaySlots(uint8_t exponent)
//{
//	int8_t error=2;
//     
//	error_AT=2;
//	gen_data(set_delay_slots_802,exponent);
//	gen_checksum(set_delay_slots_802);
//	error=gen_send(set_delay_slots_802);
//	
//    if(!error)
//    {
//        delaySlots=exponent;
//    }
//    
//    return error;
//}

/*
 Function: Get the minimum value of the back-off exponent in CSMA/CA
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "delaySlots" variable the back-off exponent
*/
//uint8_t WaspXBee802::getDelaySlots()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_delay_slots_802);
//    error=gen_send(get_delay_slots_802);
//    
//    if(!error)
//    {
//        delaySlots=data[0];
//    }
//    return error;
//}

/*
Function: Set the Mac Mode, choosen between the 4 options (0/1/2/3)
Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
Values: Change the MM command
Parameters:
  mac: set the mac mode to use (0-3)
*/
uint8_t WaspXBee802::setMacMode(uint8_t mac)
{
    int8_t error=2;
     
    error_AT=2;
    gen_data(set_mac_mode_802,mac);
    gen_checksum(set_mac_mode_802);
    error=gen_send(set_mac_mode_802);
    
    if(!error)
    {
        macMode=mac;
    }
    return error;
}

/*
Function: Get the Mac Mode
Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
Values: Stores in global "macMode" variable the Mac Mode
*/
uint8_t WaspXBee802::getMacMode()
{
	int8_t error=2;
	error_AT=2;
	gen_data(get_mac_mode_802);
	error=gen_send(get_mac_mode_802);
	if(!error)
	{
		macMode=data[0];
	}
	return error;
}

/*
 Function: Set the CA threshold in the CCA process to detect energy on the channel
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the CA command
 Parameters:
   threshold: CA threshold in the CCA process (0x00-0x50)
*/
//uint8_t WaspXBee802::setEnergyThreshold(uint8_t threshold)
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(set_energy_thres_802,threshold);
//    gen_checksum(set_energy_thres_802);
//    error=gen_send(set_energy_thres_802);
//    
//    if(!error)
//    {
//        energyThreshold=threshold;
//    }
//    return error;
//}

/*
 Function: Get the Energy Threshold used in the CCA process
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "energyThreshold" variable any error happened while execution
*/
//uint8_t WaspXBee802::getEnergyThreshold()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_energy_thres_802);
//    error=gen_send(get_energy_thres_802);
//    
//    if(!error)
//    {
//        energyThreshold=data[0];
//    } 
//    return error;
//}

/*
 Function: It gets the number of times too much energy has been found on the channel
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "counterCCA" variable number of times too much energy has been found
*/
//uint8_t WaspXBee802::getCCAcounter()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_CCA_802);
//    error=gen_send(get_CCA_802);
//    
//    if(!error)
//    {
//        counterCCA[0]=data[0];
//        delay(20);
//        counterCCA[1]=data[1];
//        delay(20);   
//    } 
//    return error;
//}

/*
 Function: Reset the CCA counter
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the EC command
*/
//uint8_t WaspXBee802::resetCCAcounter()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(reset_CCA_802);
//    error=gen_send(reset_CCA_802);
//    
//    if(!error)
//    {
//        counterCCA[0]=0;
//        counterCCA[1]=0;
//    }
//    return error;
//}

/*
 Function: Get the number of times there has been an ACK failure
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Stores in global "counterACK" variable the number of times there has been an ACK failure
*/
//uint8_t WaspXBee802::getACKcounter()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(get_ACK_802);
//    error=gen_send(get_ACK_802);
//    
//    if(!error)
//    {
//        counterACK[0]=data[0];
//        delay(20);
//        counterACK[1]=data[1];
//        delay(20);   
//    } 
//    return error;
//}

/*
 Function: Reset the ACK counter
 Returns: Integer that determines if there has been any error 
   error=2 --> The command has not been executed
   error=1 --> There has been an error while executing the command
   error=0 --> The command has been executed with no errors
 Values: Change the EA command
*/
//uint8_t WaspXBee802::resetACKcounter()
//{
//    int8_t error=2;
//     
//    error_AT=2;
//    gen_data(reset_ACK_802);
//    error=gen_send(reset_ACK_802);
//    
//    if(!error)
//    {
//        counterACK[0]=0;
//        counterACK[1]=0;
//    }
//    return error;
//}

WaspXBee802	xbee802 = WaspXBee802();
