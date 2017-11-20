from outputMidi import *
from read_melody import *
import numpy as np

name = 'test.mid'
pitch_list = np.array(read_melody(name))

outputMIDI(len(pitch_list), pitch_list[:, 1], 'test',  duration = 1)
