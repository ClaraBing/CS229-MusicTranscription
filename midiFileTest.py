from util import outputMIDI, read_melody
import numpy as np

dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/"
name = 'AClassicEducation_NightOwl'
pitch_list, _ = read_melody(name, dir)
outputMIDI(len(pitch_list), pitch_list, name,  duration_sec = 2.0/441, tempo=60)
print (len(pitch_list))
print (pitch_list[1:10])
