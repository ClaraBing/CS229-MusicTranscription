from util import outputMIDI, read_melody
import numpy as np

dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/"
name = 'AClassicEducation_NightOwl'
# name = 'AimeeNorwich_Child'
pitch_list, _ = read_melody(name, dir)
outputMIDI(len(pitch_list), pitch_list, name,  duration_sec = 0.0058, tempo=120)
print (len(pitch_list))
print (pitch_list[1:10])
