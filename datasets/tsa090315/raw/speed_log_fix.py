import csv
import os

fix_list = []
for f_name in os.listdir("./"):
    if ("speed_log" in f_name) and (".csv" in f_name) and not ("fixed" in
            f_name):
        fix_list.append(f_name)
print "Fixing:"

for item in fix_list:
    print item
    with open("fixed_"+item, 'w') as fixed:
        with open(item, 'r') as original:
            #reader = csv.reader(file)
            counter = 1
            for line in original:
                if counter > 1:
                    line = line.replace('"','')
                    if counter%2 != 1:
                        line = line.replace('\r','').replace('\n','')
                    else:
                        line = line.replace('\r','')
                else:
                    line = line.replace('\r','')
                fixed.write(line)
                counter += 1
