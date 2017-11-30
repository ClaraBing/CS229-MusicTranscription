from util import outputMIDI, read_melody
import numpy as np

name = 'AimeeNorwich_Child'
pitch_list = read_melody(name)

outputMIDI(len(pitch_list), pitch_list, 'test',  duration = 1)
