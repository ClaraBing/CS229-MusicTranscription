from outputMidi import *
from read_melody import *
import numpy as np

name = 'AimeeNorwich_Child'
pitch_list = read_melody(name)

outputMIDI(len(pitch_list), pitch_list, 'test',  duration = 1)
