from util import outputMIDI, read_melody
import numpy as np

name = 'AimeeNorwich_Child'
_, pitch_freq_list = read_melody(name)

outputMIDI(len(pitch_freq_list), pitch_freq_list, 'test',  duration = 1)
