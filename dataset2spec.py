# packages
import numpy as np
import sys
import os
import time
# our code
from util import *

common_parent = '/root/'
dir = common_parent + "MedleyDB_selected/Audios/"

output = [dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]

train_set = [line.strip() for line in open('/root/data/train.txt', 'r').readlines()]

parent_dir = common_parent + 'data/'
for folder in train_set:
	path = os.path.join(dir, folder)
	audioName, file_ext = folder+"_MIX", 'wav'
	outdir = parent_dir+''+folder+'/'
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		wav2spec_data(path+'/', audioName, file_ext, outdir)
