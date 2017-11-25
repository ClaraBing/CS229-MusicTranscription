import csv
from pitch_contour import *

def read_melody(folder_name, dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/", sampling_rate = 2):

    csv_file = dir+folder_name+"_MELODY1.csv"
    pitch_bin_list = []
    pitch_freq_list = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            #Using a sampling rate of two times the original sampling.
            if count%sampling_rate:
                count+=1
                continue
            # print(row)
            newFreq = float(list(row.values())[0])
            # Note: comparing float 0.0 to 0 results in **False**
            if newFreq > 0:
                pitch_bin_list.append(getBinFromFrequency(newFreq))
            else:
                pitch_bin_list.append(0)
            pitch_freq_list.append(newFreq)
            count+=1
    return pitch_bin_list, pitch_freq_list


# if __name__ == '__main__':
#     pitch_bin_list, pitch_freq_list = read_melody("AimeeNorwich_Child")
