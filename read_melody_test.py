from read_melody import *
from pitch_contour import *

def test_reading():
    name = 'testfile'
    pitches = read_melody(name)
    dir = "data/"
    # dir = "../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/"
    # csv_file = dir+folder_name+"_MELODY1.csv"
    csv_file = dir + "test.csv"
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            curr_frequency = float(row.values()[0])
            if  curr_frequency > 0 and getBinFromFrequency(curr_frequency)!= getBinFromFrequency(pitches[count][1]):
                return False
            count += 1
    return True

print test_reading()
