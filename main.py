import sys
sys.path.append('./tools')
from _parser import parse
from crosscorrelation import find_pixels_shifts

print "This is the main function."

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
find_pixels_shifts(PIR_data, [962,990,0,191], center_cc=False, time_cc=False)

