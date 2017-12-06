import csv
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display
import matplotlib.pylab as plt
from midiutil.MidiFile import MIDIFile
import numpy as np
import os, math

# Function list:

# Data Processing:
# extract_pitch_max: keep only max (magnitude) pitches at each time interval
# load_txt
# read_melody: read in annotations as pitch bin / freq
# read_melodies: read in annotations as binary vectors of size 109
# getBinFromFrequency / getFrequencyFromBin

# Post-processing:
# outputMIDI

# Evaluation:
# eval_accuracy

# Misc:
# plot: pitches to plot
# smooth: take a vector as input
# wav2spec_data / wav2spec_demo: generate spectrograms from wav files, for data (slices) or demo (entire audio)
# result2input



#####################
#  Data processing  #
#####################

def subsample(array1, array2):
# uniformly subsample array
	n1 = array1.shape[0]
	n2 = array2.shape[0]
	scale = n1/float(n2)
	new_array = []
	for i in range(n2):
		j = int(scale*i)
		new_array.append(array1[j])
	new_array = np.asarray(new_array)
	return new_array


def extract_pitch_max(pitches, magnitudes, timerange):
# Input:
# 	pitches: P x T matrix of pitches, each column contains the frequencies played a time t
#   magnitudes: P x T matrix of magnitudes, each column contains the magniturdes a time t
#	timerange: T
# Output:
#	new_pitches: T x 1 array containing the highest magnitude pitch at each timestamp
#	new_magnitures: T x 1 array containing the value of the highest magnitude at each timestamp
	new_pitches = []
	new_magnitudes = []
	for i in range(timerange):
		maxMagn = max(magnitudes[:,i])
		index = np.argmax(magnitudes[:,i])
		new_pitches.append(pitches[index,i])
		new_magnitudes.append(maxMagn)
	return (new_pitches,new_magnitudes)

def load_txt(f_name):
# Load data from text
	fin = open(f_name, 'r')
	return [line.strip().split(',') for line in fin.readlines()]

def getBinFromFrequency(frequency, base = 440.0):
# Get bin number from frequency using formula
# n = floor(log(2^(57/12)*f/f0)/ log(\sqrt[12]{2})
# why adding 2^(57/12): to make the output non-negative
    if frequency == 0.0:
        return 0
    else:
        return round((math.log(frequency/base) / math.log(math.pow(2.0, 1/ 12.0)))) + 58

# Get frequency from bin using the formula
def getFrequencyFromBin(bin, base = 440.0):
    if bin == 0:
        return 0.0
    else:
        return base * math.pow(2.0, (bin - 58) / 12.0)

def read_melody(folder_name, dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/", sr_ratio = 2):

    csv_file = dir+folder_name+"_MELODY1.csv"
    pitch_bin_list = []
    pitch_freq_list = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # sr_ratio: ratio between the sampling rate (sr) of the annotation and the sr of the spectrogram.
            # Currently the ratio is 2, i.e. a spectrogram corresponds to every other line in the annotation.
            if count % sr_ratio:
                count+=1
                continue
            # print(row)
            newFreq = float(list(row.values())[0])
            # Note: comparing float 0.0 to 0 results in **False**
            if newFreq > 0:
                pitch_bin_list.append(getBinFromFrequency(newFreq))
            else:
                pitch_bin_list.append(0)
            pitch_freq_list.append(newFreq)
            count+=1
    return pitch_bin_list, pitch_freq_list

def read_melodies(folder_name, dir="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/", sr_ratio = 2):

    csv_file = dir+folder_name+"_MELODY1.csv"
    pitch_bin_list = []
    pitch_freq_list = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # sr_ratio: ratio between the sampling rate (sr) of the annotation and the sr of the spectrogram.
            # Currently the ratio is 2, i.e. a spectrogram corresponds to every other line in the annotation.
            if count % sr_ratio:
                count+=1
                continue
            # print(row)
            newFreq = [float(x) for x in list(row.values())]
            newFreqBin = [getBinFromFrequency(x) for x in newFreq]
            bin_vec = [1 if i in newFreqBin else 0 for i in range(109)] # binary vector of size 109: bin_vec[i]=1 if i is a note present in the current segment
            # Note: comparing float 0.0 to 0 results in **False**
            pitch_bin_list.append(bin_vec)
            pitch_freq_list.append(newFreq)
            count+=1
    return pitch_bin_list, pitch_freq_list


###################
# Post-processing #
###################

# Input: N number of notes,
# bins: array of size N with bins
# output_name: name of the file to be saved
# duration of each notes in s.
def outputMIDI(N, bins, output_name, duration_sec=1, tempo=60):
	# Creates a MIDI file with one track
	track    = 0
	channel  = 0
	time     = 0    # In beats
	volume   = 100  # 0-127, as per the MIDI standard

	MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
	                      # automatically)
	MyMIDI.addTempo(track, time, tempo)
	lastBin = 0
	currentfreq_length = 0
	for i in range(N):
		# Ignore frequencies 0, this means there's a silence
		if bins[i] == lastBin:
			currentfreq_length+=1
		else:
			noteduration_s = 2 * currentfreq_length * duration_sec
			noteduration_beats = noteduration_s / 60 * tempo
			if lastBin > 0:
				lastfrequency = getFrequencyFromBin(lastBin)
				midiNote = int(round(21 + 12 * np.log(lastfrequency/ 27.5) / np.log(2)))
				MyMIDI.addNote(track, channel, midiNote, time, noteduration_beats, 100)
			print (lastBin, time)
			time += noteduration_beats
			lastBin = bins[i]
			currentfreq_length = 1
	binfile = open('midiOutput/'+output_name + ".mid", 'wb')
	MyMIDI.writeFile(binfile)
	binfile.close()





####################
##   Evaluation   ##
####################

def eval_accuracy(output, target, N):
    print('N: ', N)
    print('out shape: ', np.asarray(output).shape)
    print('target shape: ', np.asarray(target).shape)
    print(len(target))
    cnt = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            cnt += 1
    return sum([int(a==b) for a,b in zip(output, target)]), cnt



############
##  Misc  ##
############

def plot(vector, name, xlabel=None, ylabel=None):
# Plots data from vector into file pitch_plots/name
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.savefig('pitch_plots/'+name)


def smooth(x,window_len=11,window='hanning'):
# Smooths the data from vector x
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]


def wav2spec_data(data_dir, audioName, fext, outDir):
# Generate time slices of spectrogram, which will be used for CNN training
# Note: outDir needs to have a trailing '/': e.g. 'my_home/' rather than 'my_home'
    sr = 44100 # sampling rate
    y, sr = librosa.load(data_dir+audioName+'.'+fext, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000, n_mels=256)
    spec = librosa.power_to_db(S, ref=np.max)

    for i in range(spec.shape[1]):
        plt.figure(figsize=(2.56, 2.56)) # 100 * 400 pixels
        plt.subplot(111)
        librosa.display.specshow(spec[:, i:i+1], sr=sr)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis('off')
        plt.savefig('{:s}spec_{:s}_{:d}.png'.format(outDir, audioName, i))
        plt.close()


def wav2spec_demo(data_dir, audioName, fext, outDir):
# Generate a spectrogram for the entire audio for demo
# Note: outDir needs to have a trailing '/': e.g. 'my_home/' rather than 'my_home'
    sr = 44100
    y, sr = librosa.load(data_dir+audioName+'.'+fext, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000, n_mels=256)
    spec = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(20, 4)) # 2000 * 400 pixels
    plt.subplot(111)
    librosa.display.specshow(spec, sr=sr, y_axis='mel', x_axis="time")
    # display the legend: color vs db
    plt.colorbar(format='%+2.0f dB')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(outDir + 'spec_' + audioName + '.png')
    plt.close()


def result2input(infile, outfile):
# Format convert: cnn output (N*109*2) to lstm input (N*109)
    mtrx = np.load(infile)
    ret = []
    for i in range(mtrx.shape[0]):
        prob_vec = sorted(mtrx[i], key=lambda x: x[1])
        prob_vec = [p[0] for p in prob_vec]
        ret += prob_vec,
    np.save(outfile, np.asarray(ret))
