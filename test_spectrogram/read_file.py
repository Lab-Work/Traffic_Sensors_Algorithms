'''
Author: Yanning Li
Date: 05/28/2015

The script create a database which can perform operations on the data
   @Parameters:  CSV file
   @Returns:     database object, which can plot data, plot spectrogram, and others.
   @Example: temp = readFiles(*.csv)
'''

from PIR3_spectro import PIR3_spectro

PIR3 = PIR3_spectro('unfilteredAllData_num.csv')

PIR3.plot_data('all')
