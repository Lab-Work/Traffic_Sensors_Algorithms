#include "dingke_sysclkchange.h"
//#include "wiring_serial.h"



#define zppVECT_TAB_OFFSET  0x80000 

//#define zppPLL_M      8		 //4     //
//#define zppPLL_N     336	   //168    // 
//#define zppPLL_P      2
//#define zppPLL_Q      7

#define CHOOSESYS 0//0原来的sysinit方式 1 直接关PLL重配后开PLL
unsigned long SysClkWasp=168000000;
unsigned long DelayMsWasp=168000;
unsigned long DelayUsWasp=168;

unsigned long SysPllN=336,SysPllM=8,SysPllP=2,SysPllQ=7;
unsigned long SysFlashWait=5;

void SysChsngeDelay(u32 count)
{
	while(count--);
}
void SysPreparePara(void)
{
	unsigned long hz;
  if(SysClkWasp>336000000)SysClkWasp=336000000;	//最大应该是频率168M，目前允许让其超频到336的
  else if(SysClkWasp<2000000)SysClkWasp=2000000;//不是M级别的暂时不处理
  
  if(SysClkWasp>168000000)SysFlashWait=5;
  else SysFlashWait=SysClkWasp/30000000;

  SysClkWasp=(SysClkWasp/1000000)*1000000;

  DelayUsWasp=SysClkWasp/1000000;DelayMsWasp=SysClkWasp/1000;	

  hz= DelayUsWasp;
  if(hz>=64)     {SysPllN=hz;   SysPllM=2; SysPllP=4;}	// 64- 336
  else if(hz>=32){SysPllN=2*hz; SysPllM=2; SysPllP=8;}// 32-64
  else if(hz>=9) {SysPllN=7*hz; SysPllM=7; SysPllP=8;}//  9-31
  else if(hz>=2) {SysPllN=50*hz;SysPllM=50;SysPllP=8;}//  2-8

  //对于USB那些要考虑PLLQ
  if(hz%12==0)
  {
  	if((hz>=72)&&(hz<=180)){SysPllQ=hz/12;}
	else if(hz==60)        {SysPllN=2*hz; SysPllM=4; SysPllP=4;SysPllQ=hz/12;} //OK
	else                   {SysPllQ=7;}	
  }
  else
  	SysPllQ=7;

//  if((hz%12==0)&&(hz>=72)&&(hz<=180))
//  {
//  	SysPllQ=hz/12;
//  }
//  else if(hz==60)
//  {
//  	SysPllN=2*hz; SysPllM=4; SysPllP=4;	//实验得知，好像这个配的不成功
//  }
//  else
//  	SysPllQ=7;

}


static void zppSetSysClock(void)
{
/******************************************************************************/
/*            PLL (clocked by HSE) used as System clock source                */
/******************************************************************************/
  __IO uint32_t StartUpCounter = 0, HSEStatus = 0;
  



  //对于外部晶振为8M的情况	   //配出来的PLL最低2M，配1M的好像就死掉了，所以这里干脆强制为2M
  //


  /* Enable HSE */
  RCC->CR |= ((uint32_t)RCC_CR_HSEON);
 
  /* Wait till HSE is ready and if Time out is reached exit */
  do
  {
    HSEStatus = RCC->CR & RCC_CR_HSERDY;
    StartUpCounter++;
  } while((HSEStatus == 0) && (StartUpCounter != HSE_STARTUP_TIMEOUT));

  if ((RCC->CR & RCC_CR_HSERDY) != RESET)
  {
    HSEStatus = (uint32_t)0x01;
  }
  else
  {
    HSEStatus = (uint32_t)0x00;
  }

  if (HSEStatus == (uint32_t)0x01)
  {
    /* Enable high performance mode, System frequency up to 168 MHz */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_PMODE;  

    /* HCLK = SYSCLK / 1*/
    RCC->CFGR |= RCC_CFGR_HPRE_DIV1;
      
    /* PCLK2 = HCLK / 2*/
    RCC->CFGR |= RCC_CFGR_PPRE2_DIV2;
    
    /* PCLK1 = HCLK / 4*/
    RCC->CFGR |= RCC_CFGR_PPRE1_DIV4;

    /* Configure the main PLL */
    RCC->PLLCFGR = SysPllM | (SysPllN << 6) | (((SysPllP >> 1) -1) << 16) |
                   (RCC_PLLCFGR_PLLSRC_HSE) | (SysPllQ << 24);

    /* Enable the main PLL */
    RCC->CR |= RCC_CR_PLLON;

    /* Wait till the main PLL is ready */
    while((RCC->CR & RCC_CR_PLLRDY) == 0)
    {
    }
   
    /* Configure Flash prefetch, Instruction cache, Data cache and wait state */
//    FLASH->ACR = FLASH_ACR_ICEN |FLASH_ACR_DCEN |FLASH_ACR_LATENCY_5WS;
    FLASH->ACR = FLASH_ACR_ICEN |FLASH_ACR_DCEN |SysFlashWait;
    /* Select the main PLL as system clock source */
    RCC->CFGR &= (uint32_t)((uint32_t)~(RCC_CFGR_SW));
    RCC->CFGR |= RCC_CFGR_SW_PLL;

    /* Wait till the main PLL is used as system clock source */
    while ((RCC->CFGR & (uint32_t)RCC_CFGR_SWS ) != RCC_CFGR_SWS_PLL);
    {
    }
  }
  else
  { /* If HSE fails to start-up, the application will have wrong clock
         configuration. User can add here some code to deal with this error */
  }

}
void zppSystemInit(void)
{
#if CHOOSESYS==0
  /* Reset the RCC clock configuration to the default reset state ------------*/
  /* Set HSION bit */
  RCC->CR |= (uint32_t)0x00000001;

  /* Reset CFGR register */
  RCC->CFGR = 0x00000000;

  /* Reset HSEON, CSSON and PLLON bits */
  RCC->CR &= (uint32_t)0xFEF6FFFF;

  /* Reset PLLCFGR register */
  RCC->PLLCFGR = 0x24003010;

  /* Reset HSEBYP bit */
  RCC->CR &= (uint32_t)0xFFFBFFFF;

  /* Disable all interrupts */
  RCC->CIR = 0x00000000;

#ifdef DATA_IN_ExtSRAM
  SystemInit_ExtMemCtl(); 
#endif /* DATA_IN_ExtSRAM */
         
  /* Configure the System clock source, PLL Multiplier and Divider factors, 
     AHB/APBx prescalers and Flash settings ----------------------------------*/
  zppSetSysClock();

  /* Configure the Vector Table location add offset address ------------------*/
#ifdef VECT_TAB_SRAM
  SCB->VTOR = SRAM_BASE | VECT_TAB_OFFSET; /* Vector Table Relocation in Internal SRAM */
#else
  SCB->VTOR = FLASH_BASE | zppVECT_TAB_OFFSET; /* Vector Table Relocation in Internal FLASH */
#endif

#else//	上面是#if CHOOSESYS==0

//    RCC->CR &= ~RCC_CR_PLLON;

    /* Configure the main PLL */	//M8  N336  P2
//    RCC->PLLCFGR = PLL_M | (PLL_N << 6) | (((PLL_P >> 1) -1) << 16) |
//                   (RCC_PLLCFGR_PLLSRC_HSE) | (PLL_Q << 24);

//    RCC->PLLCFGR = 8 | (84 << 6) | (((2 >> 1) -1) << 16) |
//                   (RCC_PLLCFGR_PLLSRC_HSE) | (7 << 24);
//
//
//    /* Enable the main PLL */
//    RCC->CR |= RCC_CR_PLLON;
//
//    /* Wait till the main PLL is ready */
//    while((RCC->CR & RCC_CR_PLLRDY) == 0)
//    {
//    }
//   
//    /* Configure Flash prefetch, Instruction cache, Data cache and wait state */
//    FLASH->ACR = FLASH_ACR_ICEN |FLASH_ACR_DCEN |FLASH_ACR_LATENCY_5WS;
//
//    /* Select the main PLL as system clock source */
//    RCC->CFGR &= (uint32_t)((uint32_t)~(RCC_CFGR_SW));
//    RCC->CFGR |= RCC_CFGR_SW_PLL;
//
//    /* Wait till the main PLL is used as system clock source */
//    while ((RCC->CFGR & (uint32_t)RCC_CFGR_SWS ) != RCC_CFGR_SWS_PLL);
//    {
//    }

#endif
}
