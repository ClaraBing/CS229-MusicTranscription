from wav2spec_2 import *

import numpy as np
import sys
import os
import time

dir = "../MedleyDB_selected/Audios/"

output = [dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]


for folder in output:
	path = os.path.join(dir,folder)
	wav_file = path+"/"+folder+"_MIX.wav"
	graph_spectrogram(wav_file)
