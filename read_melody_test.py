# For test/dev
from util import *
import csv

def test_reading():
    name = "AimeeNorwich_Child"
    pitches = list(read_melody(name))
    dir = "../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/"
    csv_file = dir+name+"_MELODY1.csv"
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        even = 0
        for row in reader:
            curr_frequency = float(list(row.values())[0])
            if even == 0:
              if  curr_frequency > 0 and getBinFromFrequency(curr_frequency)!= pitches[count]:
                  return False
              count += 1
              even = 1
            else:
              even == 0
    return True

print (test_reading())
