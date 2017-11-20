import csv
from pitch_contour import *


def read_melody(folder_name, dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/"):

    csv_file = dir+folder_name+"_MELODY1.csv"
    pitch_list = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count%2:
                count+=1
                continue
            # print(row)
            newFrequency = 0.0
            freq = list(row.values())[0]
            pitch_bin = 0
            # Note: comparing float 0.0 to 0 results in **False**
            if freq != '0.0':
                # Use pitch_bin as (non-ordinal) class labels, rather than frequency
                pitch_bin = getBinFromFrequency(float(freq))
                # print(pitch_bin)
                # newFrequency = getFrequencyFromBin(getBinFromFrequency(float(freq)))
                # print(newFrequency)
            pitch_list.append(pitch_bin)
            count+=1
#        if True:
#          print(len(pitch_list))
#          print(pitch_list[:3])
    return pitch_list


if __name__ == '__main__':
    pitch_list = read_melody("AimeeNorwich_Child")
