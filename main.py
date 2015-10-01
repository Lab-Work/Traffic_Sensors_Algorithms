import sys
sys.path.append('./tools')
from _parser import parse
from crosscorrelation import find_pixels_shifts

print "Executing MAIN..."

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
#find_pixels_shifts(PIR_data, [1050,1070,0,191], center_cc=False, time_cc=False)
find_pixels_shifts(PIR_data, [1350,1570,0,191], center_cc=False, time_cc=False)

