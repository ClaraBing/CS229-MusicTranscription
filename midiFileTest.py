from util import outputMIDI, read_melody
import numpy as np

name = 'AimeeNorwich_Child'
pitch_list, _= read_melody(name)

outputMIDI(len(pitch_list), pitch_list, 'test',  duration = 0.01)
