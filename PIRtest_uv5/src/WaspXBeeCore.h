/*! \file WaspXBeeCore.h
    \brief Library for managing the XBee modules
    
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
  
    Version:		0.13

    Design:		David Gasc¨®n

    Implementation:	Alberto Bielsa

 */
 
/*! \def WaspXBeeCore_h
    \brief The library flag
    
 */
#ifndef WaspXBeeCore_h
#define WaspXBeeCore_h

/******************************************************************************
 * Includes
 ******************************************************************************/
 
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
//#include "WConstants.h"
//#include "WaspXBee.h"
//#include "WaspUSB.h"

#include <inttypes.h>

#ifndef __WASPXBEECONSTANTS_H__
  #include "WaspXBeeConstants.h"
#endif

#define DEBUGPRINT

#ifdef DEBUGPRINT
#define DBG USB.print
#define DBGLN USB.println
#else
#define DBG(...)
#define DBGLN(...)
#endif

/******************************************************************************
 * Definitions & Declarations
 ******************************************************************************/
 
//! Structure : used for storing the information returned by other nodes when a Node Discovery is performed
/*!    
 */
typedef struct strNode
{
	//! Structure Variable : 16b Network Address
	/*!    
 	*/
	uint8_t MY[2];
	
	//! Structure Variable : 32b Lower Mac Source
	/*!    
	 */
	uint8_t SH[4];
	
	//! Structure Variable : 32b Higher Mac Source
	/*!    
	 */
	uint8_t SL[4];
	
	//! Structure Variable : Node Identifier
	/*!    
	 */
	char NI[20];
	
	//! Structure Variable : Parent 16b Network Address (ZigBee)
	/*!    
	 */
	uint8_t PMY[2];
	
	//! Structure Variable : Device Type: 0=End 1=Router 2=Coord (ZigBee)
	/*!    
	 */
	uint8_t DT;
	
	//! Structure Variable : Status: Reserved (ZigBee)
	/*!    
	 */
	uint8_t ST;
	
	//! Structure Variable : Profile ID (ZigBee)
	/*!    
	 */
	uint8_t PID[2];
	
	//! Structure Variable : Manufacturer ID (ZigBee)
	/*!    
	 */
	uint8_t MID[2];
	
	//! Structure Variable : Receive Signal Strength Indicator
	/*!    
	 */
	uint8_t RSSI;
}Node;

//! Structure : used for storing the information needed to send or receive a packet, such as the addresses and data
/*!    
 */
typedef struct strpacketXBee
{
  private:
  public:
  /************ IN ***********/
	//! Structure Variable : 32b Lower Mac Destination
	/*!    
 	*/
	uint8_t macDL[4];
	
	//! Structure Variable : 32b Higher Mac Destination
	/*!    
	 */
	uint8_t macDH[4];
	
	//! Structure Variable : Sending Mode -> 0=unicast ; 1=broadcast ; 2=cluster ; 3=synchronization
	/*!    
	 */
	uint8_t mode;
	
	//! Structure Variable : Address Type -> 0=16B ; 1=64B
	/*!    
	 */
	uint8_t address_type;
	
	//! Structure Variable : 16b Network Address Destination
	/*!    
	 */
	uint8_t naD[2];
	
	//! Structure Variable : Data. All the data here, even when it is > payload
	/*!    
	 */
	char data[MAX_DATA];
	
	//! Structure Variable : Real data length.
	/*!    
	 */
	uint16_t data_length;
	
	//! Structure Variable : Fragment length. Used to send each fragment
	/*!    
	 */
	uint16_t frag_length;
	
	//! Structure Variable : Source Endpoint (ZigBee)
	/*!    
	 */
	uint8_t SD;
	
	//! Structure Variable : Destination Endpoint (ZigBee)
	/*!    
	 */
	uint8_t DE;
	
	//! Structure Variable : Cluster Identifier (ZigBee)
	/*!    
	 */
	uint8_t CID[2];
	
	//! Structure Variable : Profile Identifier (ZigBee)
	/*!    
	 */
	uint8_t PID[2];
	
	//! Structure Variable : Specifies if Network Address is known -> 0=unknown net address ; 1=known net address
	/*!    
	 */
	uint8_t MY_known;
	
	//! Structure Variable : Sending options (depends on the XBee module)
	/*!    
	 */
	uint8_t opt;
	
	/******** APLICATION *******/
	
	//! Structure Variable : Application Level ID
	/*!    
	 */
	uint8_t packetID;
	
	//! Structure Variable : 32b Lower Mac Source
	/*!    
	 */
	uint8_t macSL[4];
	
	//! Structure Variable : 32b Higher Mac Source
	/*!    
	 */
	uint8_t macSH[4];
	
	//! Structure Variable : 16b Network Address Source
	/*!    
	 */
	uint8_t naS[2];
	
	//! Structure Variable : 32b Lower Mac Origin Source
	/*!    
	 */
	uint8_t macOL[4];
	
	//! Structure Variable : 32b Higher Mac Origin Source
	/*!    
	 */
	uint8_t macOH[4];
	
	//! Structure Variable : 16b Network Address origin
	/*!    
	 */
	uint8_t naO[2];
	
	//! Structure Variable : Node Identifier Origin. To use in transmission, it must finish in "#".
	/*!    
	 */
	char niO[21];
	
	//! Structure Variable : Receive Signal Strength Indicator
	/*!    
	 */
	uint8_t RSSI;
	
	//! Structure Variable : Address Source Type
	/*!    
	 */
	uint8_t address_typeS;
	
	//! Structure Variable : Source Type ID
	/*!    
	 */
	uint8_t typeSourceID;
	
	//! Structure Variable : Fragment number for ordering the global packet
	/*!    
	 */
	uint8_t numFragment;
	
	//! Structure Variable : Specifies if it is the last fragment of a packet
	/*!    
	 */
	uint8_t endFragment;
	
	//! Structure Variable : Specifies the time when the first fragment of the packet was received
	/*!    
	 */
	long time;
	
	
	/******** OUT **************/
	
	//! Structure Variable : Delivery Status
	/*!    
	 */
	uint8_t deliv_status;
	
	//! Structure Variable : Discovery Status
	/*!    
	 */
	uint8_t discov_status;
	
	//! Structure Variable : Network Address where the packet has been set
	/*!    
	 */
	uint8_t true_naD[2];
	
	//! Structure Variable : Retries needed to send the packet
	/*!    
	 */
	uint8_t retries;
}packetXBee;

//! Structure : used for storing the received fragments
/*!    
 */
typedef struct  strmatrix
{
    private:
    public:
	    
	//! Structure Variable : Fragment data
	/*!    
	 */
        char    data[DATA_MATRIX];
	
	//! Structure Variable : Fragment data length
	/*!    
	 */
        uint16_t frag_length;
	
	//! Structure Variable : Fragment number for ordering the packet
	/*!    
	 */
        uint8_t numFragment;
	
	//! Structure Variable : Specifies if it is the last fragment
	/*!    
	 */
        uint8_t endFragment;
}matrix;


//! Structure : used for storing the needed information about the received packets
/*!    
 */
typedef struct strindex
{
  private:
  public:
	  
	//! Structure Variable : Application Level ID
	/*!    
	 */
        uint8_t packetID;
	
	//! Structure Variable : 32b Lower Mac Source
	/*!    
	 */
        uint8_t macSL[4];
	
	//! Structure Variable : 32b Higher Mac Source
	/*!    
	 */
        uint8_t macSH[4];
	
	//! Structure Variable : 16b Network Address Source
	/*!    
	 */
        uint8_t naS[2];
	
	//! Structure Variable : 32b Higher Mac Origin Source
	/*!    
	 */
        uint8_t macOH[4];
	
	//! Structure Variable : 32b Lower Mac Origin Source
	/*!    
	 */
        uint8_t macOL[4];
	
	//! Structure Variable : 16b Network Address origin
	/*!    
	 */
        uint8_t naO[2];
	
	//! Structure Variable : Node Identifier Origin. To use in transmission, it must finish in "#".
	/*!    
	 */
        char niO[20];
	
	//! Structure Variable : Source Type ID -> 0=NetAdrress ; 1=MacAddress ; 2=NodeIdentifier
	/*!    
	 */
        uint8_t typeSourceID;
	
	//! Structure Variable : Source Address Type -> 0=16B ; 1=64B
	/*!    
	 */
        uint8_t address_typeS;
	
	//! Structure Variable : Sending Mode -> 0=unicast ; 1=broadcast ; 2=cluster ; 3=synchronization
	/*!    
	 */
        uint8_t mode;
	
	//! Structure Variable : Source Endpoint
	/*!    
	 */
        uint8_t SD;
	
	//! Structure Variable : Destination Endpoint
	/*!    
	 */
        uint8_t DE;
	
	//! Structure Variable : Cluster Identifier
	/*!    
	 */
        uint8_t CID[2];
	
	//! Structure Variable : Profile Identifier
	/*!    
	 */
        uint8_t PID[2];
	
	//! Structure Variable : Sending Options (depends on the XBee module)
	/*!    
	 */
        uint8_t opt;
	
	//! Structure Variable : Receive Signal Strength Indicator
	/*!    
	 */
        uint16_t RSSI;
	
	//! Structure Variable : Time in miliseconds at the first fragment was received
	/*!    
	 */
        long time;
	
	//! Structure Variable : Specifies if the global packet is complete -> 1=Complete ; 0=Uncomplete ; -1=Empty
	/*!    
	 */
        uint8_t complete;
	
	//! Structure Variable : Specifies the total number of fragments that are expected -> -1=Empty
	/*!    
	 */
        uint8_t totalFragments;
	
	//! Structure Variable : Specifies the number of fragments received till now. -1=Empty
	/*!    
	 */
        uint8_t recFragments;
	
	//! Structure Variable : Real Data Length
	/*!    
	 */
        uint16_t data_length;
}index;


//! Structure : used to store information about a new firmware
/*!    
 */
typedef struct strfirmware_struct
{
	private:
	public:
		//! Structure Variable : New firmware ID
		/*!    
		 */
		char ID[33];
		
		//! Structure Variable : New firmware Date
		/*!    
		 */
		char DATE[13];
		
		//! Structure Variable : New firmware MAC where it was sent
		/*!    
		 */
		uint8_t mac_programming[8];
		
		//! Structure Variable : Number of packets received
		/*!    
		 */
		uint16_t packets_received;
		
		//! Structure Variable : Time in which last OTA packet arrived
		/*!    
		 */
		long	time_arrived;
		
		//! Structure Variable : New firmware ID
		/*!    
		 */
		char name_file[10];
		
		//! Structure Variable : Number of packet just received
		/*!    
		 */
		uint8_t data_count_packet;
		
		//! Structure Variable : Number of packet received before
		/*!    
		 */
		uint8_t data_count_packet_ant;
		
		//! Structure Variable : Specifies if a packet has been lost
		/*!    
		 */
		uint8_t paq_disordered;
		
		//! Structure Variable : Specifies if the function new_firmware_received has been executed previously
		/*!    
		 */
		uint8_t already_init;
		
		//! Structure Variable : channel to set after re-programming
		/*!    
		 */
		uint8_t channel;
		
		//! Structure Variable : Auth-key to set after re-programming
		/*!    
		 */
		char authkey[8];
		
		//! Structure Variable : Encryption-key to set after re-programming
		/*!    
		 */
		char encryptionkey[16];
		
		//! Structure Variable : multicast type (0:802.15.4 - 1:DigiMesh/ZB - 2:868/900)
		/*!    
		 */
		uint8_t multi_type;
}firmware_struct;


/******************************************************************************
 * Class
 ******************************************************************************/
 
 //! WaspXBeeCore Class
/*!
	WaspXBeeCore Class defines all the variables and functions used for managing the XBee modules. In this library, all the common functions are defined to be used in the other XBee libraries
 */
class WaspXBeeCore
{
  public:
	  
	//! class constructor
  	/*!
	  It does nothing
	  \param void
	  \return void
	 */
        WaspXBeeCore(){};
	
	//! It initializes the necessary variables
  	/*!
        It initalizes all the necessary variables
        \param uint8_t protocol_used : specifies the protocol used in the XBee module (depends on the XBee module)
        \param uint8_t frequency : specifies the frequency used in the XBee module (depends on the XBee module)
        \param uint8_t model_used : specifies the XBee model used (depends on the XBee module)
        \return void
         */
//        void	init(uint8_t protocol_used, uint8_t frequency, uint8_t model_used);
	
	//! It gets the own lower 32b MAC
  	/*!
        It stores in global 'sourceMacLow' variable the own lower 32b MAC
        \return '0' on success, '1' otherwise
         */
        uint8_t getOwnMacLow();
	
	//! It gets the own higher 32b MAC
  	/*!
        It stores in global 'sourceMacHigh' variable the own higher 32b MAC
        \return '0' on success, '1' otherwise
         */
        uint8_t getOwnMacHigh();
	
	//! It gets the own MAC
  	/*!
        It stores in global 'sourceMacHigh' and 'sourceMacLow' variables the 64b MAC
        \return '0' on success, '1' otherwise
         */
        uint8_t getOwnMac();
	
	//! It sets the own 16b Network Address
  	/*!
        \param uint8_t NA_H : higher Network Address byte (range [0x00-0xFF])
        \param uint8_t NA_L : lower Network Address byte (range [0x00-0xFF])
        \return '0' on success, '1' otherwise
         */
        uint8_t setOwnNetAddress(uint8_t NA_H, uint8_t NA_L);
	   
	//! It gets the 16b Network Address
  	/*!
        It stores in global 'sourceNA' the 16b Network Address
        \return '0' on success, '1' otherwise
         */
        uint8_t getOwnNetAddress();
	
	//! It sets the baudrate
  	/*!
        \param uint8_t baud_rate : the baudrate to set the XBee to (range [0-5])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setBaudrate(uint8_t baud_rate);
	   
	//! It sets API enabled or disabled
  	/*!
        \param uint8_t api_value : the API mode (range [0-2])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setAPI(uint8_t api_value);
	
	//! It sets API options
  	/*!
        \param uint8_t api_options : the API options (range [0-1])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setAPIoptions(uint8_t api_options);
	
	//! It sets the PAN ID
  	/*!
        \param uint8_t* PANID : the PAN ID (64b - ZigBee ; 16b - Other protocols)
        \return '0' on success, '1' otherwise
         */
        uint8_t setPAN(uint8_t* PANID);
	
	//! It gets the PAN ID
  	/*!
        It stores in global 'PAN_ID' the PAN ID
        \return '0' on success, '1' otherwise
         */
//        uint8_t getPAN();
	
	//! It sets the sleep mode
  	/*!
        \param uint8_t sleep : the sleep mode (range [0-5])
        \return '0' on success, '1' otherwise
         */
        uint8_t setSleepMode(uint8_t sleep);
	
	//! It gets the sleep mode
  	/*!
        It stores in global 'sleepMode' the sleep mode
        \return '0' on success, '1' otherwise
         */
//        uint8_t getSleepMode();
	
	//! It sets the time awake before sleeping
  	/*!
        \param uint8_t* awake : the time awake before sleeping (range depends on the XBee module)
        \return '0' on success, '1' otherwise
         */
//        uint8_t setAwakeTime(uint8_t* awake);
	
	//! It sets time the module is slept
  	/*!
        \param uint8_t* sleep : the module is slept (range depends on the XBee module)
        \return '0' on success, '1' otherwise
         */
//        uint8_t setSleepTime(uint8_t* sleep);
	
	//! It sets the channel frequency where module is working on
  	/*!
        \param uint8_t _channel : the channel frequency where module is working on (range depends on the XBee module)
        \return '0' on success, '1' otherwise
         */
        uint8_t setChannel(uint8_t _channel);
	
	//! It gets the channel frequency where module is working on
  	/*!
        It stores in global 'channel' the channel frequency where module is working on
        \return '0' on success, '1' otherwise
         */
        uint8_t getChannel();
	
	//! It sets the Node Identifier
  	/*!
        \param char* node : the NI must be a 20 character max string
        \return '0' on success, '1' otherwise
         */
//        uint8_t setNodeIdentifier(const char* node);
	
	//! It gets the Node Identifier
  	/*!
        It stores in global 'nodeID' the Node Identifier
        \return '0' on success, '1' otherwise
         */
//        uint8_t getNodeIdentifier();
	
	//! It scans for brothers in the same channel and same PAN ID
  	/*!
        It stores the given info (SH,SL,MY,RSSI,NI) in global array "scannedBrothers" variable
        It stores in global "totalScannedBrothers" the number of found brothers
        \return '0' on success, '1' otherwise
         */
        uint8_t scanNetwork();
	
	//! It scans for a brother in the same channel and same PAN ID
  	/*!
        It stores the given info (SH,SL,MY,RSSI,NI) in global array "scannedBrothers" variable
        It stores in global "totalScannedBrothers" the number of found brothers
        \param char* node : 20-byte max string containing NI of the node to search
        \return '0' on success, '1' otherwise
         */
//        uint8_t scanNetwork(const char* node);
	
	//! It sets the time the Node Discovery is scanning
  	/*!
        \param uint8_t* time : the time the Node Discovery is scanning (range [0x01-0xFC])
        \return '0' on success, '1' otherwise
         */
        uint8_t setScanningTime(uint8_t* time);
	
	//! It gets the time the Node Discovery is scanning
  	/*!
        It stores in global 'scanTime' the time the Node Discovery is scanning
        \return '0' on success, '1' otherwise
         */
//        uint8_t getScanningTime();
	
	//! It sets the options for the network discovery command
  	/*!
        \param uint8_t options : the options for the network discovery command (range [0x00-0x03])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setDiscoveryOptions(uint8_t options);
	
	//! It gets the options for the network discovery command
  	/*!
        It stores in global 'discoveryOptions' the options for the network discovery command
        \return '0' on success, '1' otherwise
         */
//        uint8_t getDiscoveryOptions();
	
	//! It performs a quick search of a specific node. Depending on the XBee module it stores different information
  	/*!
        It stores the given info (SH,SL,MY,RSSI,NI) in global array "scannedBrothers" variable
        It stores in global "totalScannedBrothers" the number of found brothers
        \param char* node : 20-byte max string containing NI of the node to search
        \param struct packetXBee* paq : a packetXBee structure to store the node information
        \return '0' on success, '1' otherwise
         */
//        uint8_t nodeSearch(const char* node, struct packetXBee* paq);
	
	//! It sets the list of channels to scan when performing an energy scan 
  	/*!
        \param uint8_t channel_H : higher channel list byte (range [0x00-0xFF])
        \param uint8_t channel_L : lower channel list byte (range [0x00-0xFF])
        \return '0' on success, '1' otherwise
         */
        uint8_t setScanningChannels(uint8_t channel_H, uint8_t channel_L);
	
	//! It gets the list of channels to scan when performing an energy scan 
  	/*!
        It stores in global 'scanChannels' the list of channels to scan when performing an energy scan 
        \return '0' on success, '1' otherwise
         */
        uint8_t getScanningChannels();
	
	//! It sets the time the Energy Scan will be performed
  	/*!
        It stores the energy on each channel in global 'energyChannel' variable
        \param uint8_t duration : time the energy scan will be performed (range [0-6])
        \return '0' on success, '1' otherwise
         */
        uint8_t setDurationEnergyChannels(uint8_t duration);
	
	//! It gets the time the Energy Scan will be performed
  	/*!
        It stores in global 'timeEnergyChannel' the time the Energy Scan will be performed
        \return '0' on success, '1' otherwise
         */
        uint8_t getDurationEnergyChannels();
	
	//! It sets the link key to use in the 128b AES algorithm
  	/*!
        \param char* key : the 128-bit AES encryption key (range [0-0xFFFFFFFFFFFFFFFF])
        \return '0' on success, '1' otherwise
         */
        uint8_t setLinkKey(const char* key);

//	   uint8_t getLinkKey(void);
	
	//! It sets the encryption mode ON/OFF
  	/*!
        \param uint8_t mode : the encryption mode (range [0-1])
        \return '0' on success, '1' otherwise
         */
        uint8_t encryptionMode(uint8_t mode);


//	   uint8_t getencryptionMode(void);
	
	//! It sets the power level at which RF module transmits
  	/*!
        \param uint8_t mode : power level at which RF module transmits (depends on the XBee module - range [0-4])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setPowerLevel(uint8_t value);
	
	//! It gets the Received Signal Strength Indicator
  	/*!
        It stores in global 'valueRSSI' the Received Signal Strength Indicator
        \return '0' on success, '1' otherwise
         */
//        uint8_t getRSSI();
	
	//! It gets the Hardware Version
  	/*!
        It stores in global 'hardVersion' the Hardware Version
        \return '0' on success, '1' otherwise
         */
//        uint8_t getHardVersion();
	
	//! It gets the Software Version
  	/*!
        It stores in global 'softVersion' the Software Version
        \return '0' on success, '1' otherwise
         */
//        uint8_t getSoftVersion();
	
	//! It sets the RSSI time
  	/*!
        \param uint8_t time : amount of time to do the PWM (range [0x00-0xFF])
        \return '0' on success, '1' otherwise
         */
//        uint8_t setRSSItime(uint8_t time);
	
	//! It gets the RSSI time
  	/*!
        It stores in global 'timeRSSI' the RSSI time
        \return '0' on success, '1' otherwise
         */
//        uint8_t getRSSItime();
	
	//! It writes the parameters changed into non-volatil memory, being applied when the XBee is set OFF
  	/*!
        \return '0' on success, '1' otherwise
         */
        uint8_t writeValues();
	
	//! It writes the parameters changed into non-volatil memory and applies them immediately
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t applyChanges();
	
	//! It resets the XBee firmware
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t reset();
	
	//! It resets the XBee parameters to factory defaults
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t resetDefaults();
	
	//! It sets the sleep options
  	/*!
        \param uint8_t soption : the sleep options (range depends on the XBee module)
        \return '0' on success, '1' otherwise
         */
//        uint8_t setSleepOptions(uint8_t soption);
	
	//! It gets the sleep options
  	/*!
        It stores in global 'sleepOptions' the sleep options
        \return '0' on success, '1' otherwise
         */
//        uint8_t getSleepOptions();
	
	//! It sends a packet to others XBee modules
  	/*!
        \param struct packetXBee* packet : it is filled with the information needed to be able to send the packet
        \return '0' on success, '1' otherwise
         */
        uint8_t sendXBee(packetXBee* packet);

	 uint8_t analyserecframe(char * sendstr, uint8_t len);
	 uint8_t sendhandleresponse(char * sendstr, uint8_t len);
	 
	 uint8_t findRxframe(char * str, uint16_t *len); 

	 uint8_t findRxframeinRx80Buffer(char * str, uint16_t *len);

	
	//! It sends a packet to others XBee modules
  	/*!
        \param char* address : address where to send the packet to
        \param uint8_t* data : data to send	
        \return '0' on success, '1' if error, '-1' if no memory
         */
//        int8_t send(char* address, uint8_t* data);
	
	//! It sends a packet to others XBee modules
  	/*!
        \param uint8_t* address : address where to send the packet to
        \param uint8_t* data : data to send	
        \return '0' on success, '1' if error, '-1' if no memory
         */
//        int8_t send(uint8_t* address, uint8_t* data);
	
	//! It sends a packet to others XBee modules
  	/*!
        \param uint8_t* address : address where to send the packet to
        \param char* data : data to send	
        \return '0' on success, '1' if error, '-1' if no memory
         */
//        int8_t send(uint8_t* address, char* data);
	
	//! It sends a packet to others XBee modules
  	/*!
        \param char* address : address where to send the packet to
        \param char* data : data to send	
        \return '0' on success, '1' otherwise
         */
//        int8_t send(char* address, char* data);
	
	//! It sends a packet to others XBee modules
  	/*!
        \param char* address : address where to send the packet to
        \param char* data : data to send
        \param uint8_t type : 0==string transmission | 1=byte transmission
        \param uint8_t dataMax : length to send in byte transmission
        \return '0' on success, '1' otherwise
         */
//        int8_t send(char* address, char* data, uint8_t type, uint8_t dataMax);
	
	//! It treats the data from XBee UART
  	/*!
        \return '0' on success, '1' otherwise
         */
        int8_t treatData();
	
	//! It frees the outcoming data buffer in the XBee
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t freeXBee();
	
	//! It synchronizes two nodes in a PAN
  	/*!
        \param struct packetXBee* paq : it is filled with the information needed to be able to synchronize the two nodes
        \return '0' on success, '1' otherwise
         */
//        uint8_t synchronization(struct packetXBee* paq);
	
	//! The user introduces an AT command within a string and the function executes it without knowing its meaning
  	/*!
        \param char* atcommand : the command to execute. It must finish with a '#'
        \return '0' on success, '1' otherwise
         */
//        uint8_t sendCommandAT(const char *atcommand);
	
	//! It connects XBee, activating switch and opens the UART
  	/*!
        \return '0' on success, '1' otherwise
         */
        uint8_t ON();
	
	//! It disconnects XBee, switching it off and closing the UART
  	/*!
        \return '0' on success, '1' otherwise
         */
        uint8_t OFF();
	
	//! It sets the XBee to sleep, asserting PIN 9
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t sleep();
	
	//! It wakes up the XBee, de-asserting PIN 9
  	/*!
        \return '0' on success, '1' otherwise
         */
//        uint8_t wake();
	
	//! It sets the destination parameters, such as the receiver address and the data to send
  	/*!
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param uint8_t* address : the receiver MAC
        \param char* data : the data to send
        \param uint8_t type : origin identification type (using this function call, it only can be used MAC_TYPE)
        \param uint8_t off_type : DATA_ABSOLUTE or DATA_OFFSET. It specifies if 'data' are absolute or if they must be added at the end of the packet
        \return '1' on success
         */
        int8_t setDestinationParams(packetXBee* paq, uint8_t* address, const char* data, uint8_t type, uint8_t off_type);
	
	//! It sets the destination parameters, such as the receiver address and the data to send
  	/*!
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param uint8_t* address : the receiver MAC
        \param int data : the data to send
        \param uint8_t type : origin identification type (using this function call, it only can be used MAC_TYPE)
        \param uint8_t off_type : DATA_ABSOLUTE or DATA_OFFSET. It specifies if 'data' are absolute or if they must be added at the end of the packet
        \return '1' on success
         */
        int8_t setDestinationParams(packetXBee* paq, uint8_t* address, int data, uint8_t type, uint8_t off_type);
	
	//! It sets the destination parameters, such as the receiver address and the data to send
  	/*!
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param char* address : the receiver MAC
        \param char* data : the data to send
        \param uint8_t type : origin identification type (MAC_TYPE,MY_TYPE or NI_TYPE)
        \param uint8_t off_type : DATA_ABSOLUTE or DATA_OFFSET. It specifies if 'data' are absolute or if they must be added at the end of the packet
        \return '1' on success
         */
        int8_t setDestinationParams(packetXBee* paq, const char* address, const char* data, uint8_t type, uint8_t off_type);
	
	//! It sets the destination parameters, such as the receiver address and the data to send
  	/*!
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param char* address : the receiver MAC
        \param int data : the data to send
        \param uint8_t type : origin identification type (MAC_TYPE,MY_TYPE or NI_TYPE)
        \param uint8_t off_type : DATA_ABSOLUTE or DATA_OFFSET. It specifies if 'data' are absolute or if they must be added at the end of the packet
        \return '1' on success
         */
        int8_t setDestinationParams(packetXBee* paq, const char* address, int data, uint8_t type, uint8_t off_type);
	
	//! It sets the origin parameters, such as the sender address
  	/*!
        It gets the origin identification from the node.
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param uint8_t type : origin identification type (MAC_TYPE,MY_TYPE or NI_TYPE)
        \return '1' on success
         */
        int8_t setOriginParams(packetXBee* paq, uint8_t type);
	
	//! It sets the origin parameters, such as the sender address
  	/*!
        \param packetXBee* paq : a packetXBee structure where some parameters should have been filled before calling this function. After this call, this structure is filled with the corresponding address and data
        \param char* address : origin identification (Netowrk Address, MAC Address or Node Identifier)
        \param uint8_t type : origin identification type (MAC_TYPE,MY_TYPE or NI_TYPE)
        \return '1' on success
         */
        int8_t setOriginParams(packetXBee* paq, const char* address, uint8_t type);
	
	//! It clears the variable 'command'
  	/*!
         */
        void clearCommand();
	
	//! It checks the new firmware upgrade
  	/*!
	\return void
	 */
	void checkNewProgram();

	//! It checks if timeout is up while sending program packets
  	/*!
	\return '1'--> Timeout is up.  '0'--> The function was executed with no errors 
	 */
	uint8_t checkOtapTimeout();



	
	//! Variable : it stores if the XBee module is ON or OFF (0-1)
	/*!    
	 */
	uint8_t XBee_ON;
	
	//! Variable : 32b Lower Mac Source 
	/*!    
	 */
	uint8_t sourceMacLow[4];
	
	//! Variable : 32b Higher Mac Source
	/*!    
	 */
   	uint8_t sourceMacHigh[4];
	
	//! Variable : 16b Network Address
	/*!    
	 */
   	uint8_t sourceNA[2];
	
	//! Variable : Baudrate, speed used to communicate with the XBee module (0-5)
	/*!    
	 */
        uint8_t baudrate;
	
	//! Variable : Api value selected (0-2)
	/*!    
	 */
        uint8_t apiValue;
	
	//! Variable : 64b PAN ID
	/*!    
	 */
	uint8_t PAN_ID[8];
	
	//! Variable : Current Sleep Mode (0-5)
	/*!    
	 */
	uint8_t sleepMode;
	
	//! Variable : bidimensional array for storing the received fragments
	/*!    
	 */
	matrix *packet_fragments[MAX_FINISH_PACKETS][MAX_FRAG_PACKETS];
	
	//! Variable : the number of fragments received
	/*!    
	 */
	uint8_t totalFragmentsReceived;
	
	//! Variable : array for storing the information related with each global packet, being able to order the different fragments
	/*!    
	 */
	index *pendingFragments[MAX_FINISH_PACKETS];
	
	//! Variable : number of packets pending of being treated
	/*!    
	 */
	uint8_t pendingPackets;
	
	//! Variable : array for storing the packets received completely
	/*!    
	 */
	packetXBee *packet_finished[MAX_FINISH_PACKETS];
	
	//! Variable : real number of complete received packets
	/*!    
	 */
	uint8_t totalPacketsReceived;
	
	//! Variable : indicates the position in 'packet_finished' array of each packet
	/*!    
	 */
	uint8_t	pos;
	
	//! Variable : array for storing the information answered by the nodes when a Node Discovery is performe
	/*!    
	 */
	Node scannedBrothers[MAX_BROTHERS];
	
	//! Variable : number of brothers found
	/*!    
	 */
	int8_t totalScannedBrothers;
	
	//! Variable : time to be idle before start sleeping
	/*!    
	 */
	uint8_t awakeTime[3];
	
	//! Variable : Cyclic sleeping time
	/*!    
	 */
	uint8_t sleepTime[3];
	
	//! Variable : Channel frequency where the module is currently working on
	/*!    
	 */
	uint8_t channel;
	
	//! Variable : Node Identifier
	/*!    
	 */
	char nodeID[20];
	
	//! Variable : time meanwhile the Node Discovery is scanning
	/*!    
	 */
	uint8_t scanTime[2];
	
	//! Variable : options for the network discovery command 
	/*!    
	 */
        uint8_t discoveryOptions;
	
	//! Variable : list of channels to scan when performing an energy scan
	/*!    
	 */
	uint8_t scanChannels[2];
	
	//! Variable : energy found on each channel
	/*!    
	 */
	uint8_t energyChannel[20];
	
	//! Variable : time the Energy Scan is going to be performed
	/*!    
	 */
	uint8_t timeEnergyChannel;
	
	//! Variable : 128b AES Link key
	/*!    
	 */
	char linkKey[16];
	
	//! Variable : encryption mode (ON/OFF) (0-1)
	/*!    
	 */
	uint8_t encryptMode;
	
	//! Variable : power level at which the RF transmits
	/*!    
	 */
	uint8_t powerLevel;
	
	//! Variable : time meanwhile the PWM output is active after receiving a packet
	/*!    
	 */
	uint8_t timeRSSI;
	
	//! Variable : software Version
	/*!    
	 */
	uint8_t softVersion[2];
	
	//! Variable : hardware Version
	/*!    
	 */
	uint8_t hardVersion[2];
	
	//! Variable : received Signal Strength Indicator
	/*!    
	 */
	uint8_t valueRSSI[2];
	
	//! Variable : sleep Options
	/*!    
	 */
	uint8_t sleepOptions;
	
	//! Variable : max number of hops a packet can travel
	/*!    
	 */
	uint8_t hops;
	
	//! Variable : Answer received after executing "sendCommandAT" function
	/*!    
	 */
        uint8_t commandAT[100];
	
	//! Variable : It stores the last Modem Status indication received
	/*!    
	 */
	uint8_t modem_status;
	
	//! Variable : It specifies the replacement ploicty to implement (FIFO, LIFO or OUT)
	/*!    
	 */
	uint8_t replacementPolicy;
	
	//! Variable : It stores if the last call to an AT command has generated an error
	/*!    
	 */
	int8_t error_AT;
	
	//! Variable : It stores if the last received packet has generated an error
	/*!    
	 */
	int8_t error_RX;
	
	//! Variable : It stores if the last sent packet has generated an error
	/*!    
	 */
	int8_t error_TX;
	
	//! Variable : specifies the firmware information
  	/*!
	 */
	firmware_struct firm_info;
	
	//! Variable : specifies if the re-programming process is running
  	/*!
	 */
	uint8_t programming_ON;


	uint8_t sendTx64Simple(char datastr[],unsigned char lendata,unsigned long addressH,unsigned long addressL);

  protected:
	
	  
	//! It reads a packet from other XBee module
  	/*!
      It should be called when data is available from the XBee. If the available data is not a packet it will handle it and will return the appropriate message
      \return '0' on success, '1' when error, '-1' when no more memory is available
         */
      int8_t readXBee(uint8_t* data);
	
	//! It sends a packet to other XBee modules
  	/*!
      \param struct packetXBee* packet : the function gets the needed information to send the packet from it
      \return '0' on success, '1' otherwise
         */
      uint8_t sendXBeePriv(packetXBee* packet);
	
	//! It generates the API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \param uint8_t param : input parameter to set using the AT command
      \return void
         */
      void gen_data(const char* data, uint8_t param);
	
	//! It generates the API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \return void
         */
      void gen_data(const char* data);
	
	//! It generates the API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \param uint8_t param1 : higher part of the input parameter to set using the AT command
      \param uint8_t param2 : lower part of the input parameter to set using the AT command	
      \return void
         */
      void gen_data(const char* data, uint8_t param1, uint8_t param2);
	
	//! It generates the API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \param uint8_t* param : input parameter to set using the AT command
      \return void
         */
      void gen_data(const char* data, uint8_t* param);
	
	//! It generates the API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \param char* param : input parameter to set using the AT command
      \return void
         */	
      void gen_data(const char* data, const char* param);

	//! It generates the checksum API frame to send to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \return the checksum generated
         */
      uint8_t gen_checksum(const char* data);	
	
	//! It sends the API frame stored in 'command' variable to the XBee module
  	/*!
      \param const char* data : the string that contains part of the API frame
      \return '0' if no error, '1' if error
         */
      uint8_t gen_send(const char* data);
	
	//! It generates the API frame when a TX is done
  	/*!
      \param struct packetXBee* _packet : packet for storing the data to send
      \param uint8_t* TX_array : array for storing the data
      \param uint8_t start_pos: start position
         */
      void gen_frame(packetXBee* _packet, uint8_t* TX_array, uint8_t start_pos);
	
	//! It generates the frame using eschaped characters
  	/*!
      \param struct packetXBee* _packet : packet for storing the data to send
      \param uint8_t* TX_array : array for storing the data
      \param uint8_t &protect: variable used for storing if some protected character
      \param uint8_t type: variable used for knowing the frame length
         */
      void gen_frame_ap2(packetXBee* _packet, uint8_t* TX_array, uint8_t &protect, uint8_t type);
	
	//! It parses the answer received by the XBee module, calling the appropriate function
  	/*!
      \param uint8_t* frame : an array that contains the API frame that is expected to receive answer from if it is an AT command
      \return '0' if no error, '1' if error
         */
      int8_t parse_message(uint8_t* frame);
	
	//! It generates the correct API frame from an eschaped one
  	/*!
      \param uint8_t* data_in : the string that contains the eschaped API frame AT command
      \param uint16_t end : the end of the frame
      \param uint16_t start : the start of the frame
      \return '0' if no error, '1' if error
         */
      void des_esc(uint8_t* data_in, uint16_t end, uint16_t start);
	
	//! It parses the AT command answer received by the XBee module
  	/*!
      \param uint8_t* data_in : the string that contains the eschaped API frame AT command
      \param uint8_t* frame : an array that contains the API frame that is expected to receive answer from if it is an AT command
      \param uint16_t end : the end of the frame
      \param uint16_t start : the start of the frame
      \return '0' if no error, '1' if error
         */
      uint8_t atCommandResponse(uint8_t* data_in, uint8_t* frame, uint16_t end, uint16_t start);
	
	//! It parses the Modem Status message received by the XBee module
  	/*!
      \param uint8_t* data_in : the string that contains the eschaped API frame AT command
      \param uint16_t end : the end of the frame
      \param uint16_t start : the start of the frame
      \return '0' if no error, '1' if error
         */
      uint8_t modemStatusResponse(uint8_t* data_in, uint16_t end, uint16_t start);
	
	//! It parses the TX Status message received by the XBee module
  	/*!
      \param uint8_t* ByteIN : array to store the received answer
      \return '0' if no error, '1' if error
         */
      uint8_t txStatusResponse();	

	//! It parses the ZB TX Status message received by the XBee module
  	/*!
      \param uint8_t* ByteIN : array to store the received answer
      \return '0' if no error, '1' if error
         */
      uint8_t txZBStatusResponse();
	
	//! It parses the RX Data message received by the XBee module
  	/*!
      \param uint8_t* data_in : the string that contains the eschaped API frame AT command
      \param uint16_t end : the end of the frame
      \param uint16_t start : the start of the frame
      \return '0' if no error, '1' if error
         */
      int8_t rxData(uint8_t* data_in, uint16_t end, uint16_t start);
	
	//! It parses the ND message received by the XBee module
  	/*!
	It stores in 'scannedBrothers' variable the data extracted from the answer
	 */
	void treatScan();		
	
	//! It checks the checksum is good
  	/*!
        \param uint8_t* data_in : the string that contains the eschaped API frame AT command
        \param uint16_t end : the end of the frame
        \param uint16_t start : the start of the frame
        \return '0' if no error, '1' if error
         */
        uint8_t checkChecksum(uint8_t* data_in, uint16_t end, uint16_t start);	
    //! It gets the TX frame checksum 
  	/*!
    \param uint8_t* TX : the pointer to the generated frame which checksum has 
    to be calculated
	\return calculated checksum
	*/	
	uint8_t getChecksum(uint8_t* TX);		
	//! It frees a position in index array
  	/*!
         */
        void freeIndex();
	
	//! It frees index array and matrix
  	/*!
         */
        void freeAll();
	
	//! It gets the next index where store the finished packet
  	/*!
        \return the index where store the packet
         */
        uint8_t getFinishIndex();
	
	//! It clears the finished packets array
  	/*!
         */
        void clearFinishArray();
	
	//! It gets the index in 'packet_finished' where store the new packet, according to a FIFO policy
  	/*!
        \return the index where store the packet
         */
        uint8_t getIndexFIFO();
	
	//! It gets the index in 'packet_finished' where store the new packet, according to a LIFO policy
  	/*!
        \return the index where store the packet
         */
        uint8_t getIndexLIFO();
	
	//! It frees the index array and the matrix row corresponding to the position is sent as an input parameter
  	/*!
        \param uint8_t position : the position to free
         */
        void freeIndexMatrix(uint8_t position);
	
	//! It receives the first packet of a new firmware
  	/*!
	\return 1 if error, 0 otherwise
	 */
	uint8_t new_firmware_received();
	
	//! It receives the data packets of a new firmware
  	/*!
	\return void
	 */
	void new_firmware_packets();

	//! It receives the last packet of a new firmware
  	/*!
	\return void
	 */
	void new_firmware_end();

	//! It uploads the new firmware
  	/*!
	\return void
	 */
	void upload_firmware();

	//! It answers the ID requested
  	/*!
	\return void
	 */
	void request_ID();

	//! It answers the boot list file
  	/*!
	\return void
	 */
	void request_bootlist();
		
	//! It deletes the firmware required
  	/*!
	\return void
	 */
	void delete_firmware();
	
	//! It sets the previous configuration in multicast transmissions
  	/*!
	\return void
	 */
	void setMulticastConf();

	//! Variable : protocol used (depends on de the XBee module)
  	/*!
	 */
	uint8_t protocol;
	
	//! Variable : frequency used (depends on de the XBee module)
  	/*!
	 */
	uint8_t freq;
	
	//! Variable : model used (depends on de the XBee module)
  	/*!
	 */
	uint8_t model;
	
	//! Variable : it stores the UART where the dara are sent to (UART0 or UART1)
	/*!    
	 */
	uint8_t uart;
	
	//! Variable : internal variable used to store the data length
  	/*!
	 */
   	uint16_t data_length;
	
	//! Variable : it stores the data received in each frame
  	/*!
	 */
   	uint8_t data[50];
	
	//! Variable : internal counter
  	/*!
	 */
	uint16_t it;
	
	//! Variable : byte for starting getting the data
  	/*!
	 */
	uint16_t start;
	
	//! Variable : byte for stopping getting the data
  	/*!
	 */
	uint16_t finish;
	
	//! Variable : address type
  	/*!
	 */
	uint8_t add_type;
	
	//! Variable : sending mode
  	/*!
	 */
	uint8_t mode;
	
	//! Variable : fragment length
  	/*!
	 */
	uint16_t frag_length;
	
	//! Variable : time used to manage timeouts
  	/*!
	 */
        long TIME1;
	
	//! Variable : array to store the AT commands
  	/*!
	 */
	uint8_t command[30];
	
	//! Variable : index data to make a packet from many different data
  	/*!
	 */
	uint16_t data_ind;
	
	//! Variable : delivery packet status
  	/*!
	 */
	uint8_t delivery_status;
	
	//! Variable : discovery process status
  	/*!
	 */
	uint8_t discovery_status;
	
	//! Variable : true 16b Network Address where the packet has been sent
  	/*!
	 */
	uint8_t true_naD[2];
	
	//! Variable : retries done during the sending
  	/*!
	 */
	uint8_t retries_sending;
	
	//! Variable : specifies the next index where storing the next received fragment
  	/*!
	 */
	uint8_t nextIndex1;
	
	//! Variable : flag to indicate if a frame was truncated
  	/*!
	 */
	uint8_t frameNext;
	
	//! Variable : flag to indicate if the variable 'nextIndex1' has been modified during the last execution of 'readXBee'
  	/*!
	 */
	uint8_t indexNotModified;
	
	//! Variable : specifies if APS encryption is enabled or disabled
  	/*!
	 */
	uint8_t apsEncryption;

	//! Variable : object to the SD card
  	/*!
	 */
	/*
	
	
	Sd2Card card;



	*/

	//! Variable : flag to indicate sd card is on
  	/*!
	 */
	uint8_t sd_on;
};


#endif

