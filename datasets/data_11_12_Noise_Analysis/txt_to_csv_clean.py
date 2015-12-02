__author__ = 'wbarbour1'

import csv
import os
from pir_convert import *
import time


translate_list = []
for f_name in os.listdir('/Users/wbarbour1/Documents/Dropbox/CEE/PIR Array/data_11_12'):
    print f_name
    if f_name[-4:] == '.txt' or f_name[-4:] == '.asc':
        translate_list.append(f_name)

print translate_list

skip_flag = False
for txt_file in translate_list:
    print "Working on", txt_file, txt_file[0]
    if txt_file[0] == '4':
        converter = PIR3_converter()
    elif txt_file[0] == '2':
        converter = PIR2x16_converter()
    else:
        raise ImportError

    trial_fname = txt_file
    output_fname = 'csv_' + trial_fname[0:-4] + '.csv'
    print "output file: ", output_fname

    trans_file = open(trial_fname, 'r')
    csv_file = open(output_fname, 'w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)

    for line in trans_file:
        csv_line = []
        try:
            csv_line = converter.convert(line)
        except:
            raise ValueError

        csv_writer.writerow(csv_line)

    trans_file.close()
    csv_file.close()
    print csv_file.name, " complete. Pausing for 5 seconds."
    time.sleep(5)
