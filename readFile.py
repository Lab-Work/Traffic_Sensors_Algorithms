'''
Author: Fangyu Wu
Date: 04/05/2015

The module reads temperature data in from an existing csv file and stores them into a PIRTemp obejct.
   @Parameters:  CSV file
   @Returns:     temp object
   @Example: temp = readFiles(*.csv)
'''

from PIRTemp import PIRTemp

PIR = PIRTemp('dataset.csv')
#PIR.logisticRegression()
#PIR.linearRegression()
#PIR.display(0)
#PIR.timeSeries()
#PIR.heatMap()
PIR.heatTopo()
