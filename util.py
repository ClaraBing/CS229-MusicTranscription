import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import librosa
import librosa.display

# Function list:
# extract_pitch_max: keep only max (magnitude) pitches at each time interval
# plot: pitches to plot
# smooth: take a vector as input
# load_txt
# wav2spec_data / wav2spec_demo: generate spectrograms from wav files, for data (slices) or demo (entire audio)



# Input:
# 	pitches: P x T matrix of pitches, each column contains the frequencies played a time t
#   magnitudes: P x T matrix of magnitudes, each column contains the magniturdes a time t
#	timerange: T
# Output:
#	new_pitches: T x 1 array containing the highest magnitude pitch at each timestamp
#	new_magnitures: T x 1 array containing the value of the highest magnitude at each timestamp
def extract_pitch_max(pitches, magnitudes, timerange):
	new_pitches = []
	new_magnitudes = []
	for i in range(timerange):
		maxMagn = max(magnitudes[:,i])
		index = np.argmax(magnitudes[:,i])
		new_pitches.append(pitches[index,i])
		new_magnitudes.append(maxMagn)
	return (new_pitches,new_magnitudes)


# Plots data from vector into file pitch_plots/name
def plot(vector, name, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.savefig('pitch_plots/'+name)


# Smooths the data from vector x
def smooth(x,window_len=11,window='hanning'):
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


# Load data from text
def load_txt(f_name):
	fin = open(f_name, 'r')
	return [line.strip().split(',') for line in fin.readlines()]


# Generate time slices of spectrogram, which will be used for CNN training
# Note: outDir needs to have a trailing '/': e.g. 'my_home/' rather than 'my_home'
def wav2spec_data(data_dir, audioName, fext, outDir):
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


# Generate a spectrogram for the entire audio for demo
# Note: outDir needs to have a trailing '/': e.g. 'my_home/' rather than 'my_home'
def wav2spec_demo(data_dir, audioName, fext, outDir):
    sr = 44100
    y, sr = librosa.load(fname, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000, n_mels=256, y_axis='mel', x_axis="time")
    spec = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(20, 4)) # 2000 * 400 pixels
    plt.subplot(111)
    librosa.display.specshow(spec[:, i:i+1], sr=sr)
    # display the legend: color vs db
    plt.colorbar(format='%+2.0f dB')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(outDir + 'spec_' + audioName + '.png')
    plt.close()

