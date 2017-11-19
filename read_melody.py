import numpy as np
import sys
import os
import time
import csv

def read_melody(folder_name):
	dir = "../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/"
	csv_file = dir+folder_name+"_MELODY1.csv"
	pitch_list = []
	with open(csv_file) as f:
		reader = csv.DictReader(f)
		count = 0
		for row in reader:
#			print(row,row.keys())
			pitch_list.append([count, float(row.values()[0])])
			count+=1
#		pitch_list = list(reader)
        if True:
          print(len(pitch_list))
          print(pitch_list[:3])
	return pitch_list


if __name__ == '__main__':
	pitch_list = read_melody("AimeeNorwich_Child")
#	print(len(pitch_list))
#	print(pitch_list)
