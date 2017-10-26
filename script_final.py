#!/usr/bin/env ipython

# Authors: martijn.millecamp@student.kuleuven.be [Martijn Millecamp] & miro.masat@gmail.com [Miroslav Masat]

import librosa
import numpy
import matplotlib.pylab as plt



def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(numpy.max(pitches[:,i]))
        new_magnitudes.append(numpy.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def smooth(x,window_len=11,window='hanning'):
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=numpy.ones(window_len,'d')
        else:
                w=eval('numpy.'+window+'(window_len)')
        y=numpy.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def plot(vector, name, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.savefig(name)

def set_variables(sample_f,duration,window_time,fmin,fmax,overlap):
    total_samples = sample_f * duration
    #There are sample_f/1000 samples / ms
    #windowsize = number of samples in one window
    window_size = sample_f/1000 * window_time
    hop_length = total_samples / window_size
    #Calculate number of windows needed
    needed_nb_windows = total_samples / (window_size - overlap)
    n_fft = needed_nb_windows * 2.0
    return total_samples, window_size, needed_nb_windows, n_fft, hop_length

def analyse(y,sr,n_fft,hop_length,fmin,fmax):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, S=None, n_fft= n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, threshold=0.75)
    shape = numpy.shape(pitches)
    #nb_samples = total_samples / hop_length
    nb_samples = shape[0]
    #nb_windows = n_fft / 2
    nb_windows = shape[1]
    pitches,magnitudes = extract_max(pitches, magnitudes, shape)

    pitches1 = smooth(pitches,window_len=10)
    pitches2 = smooth(pitches,window_len=20)
    pitches3 = smooth(pitches,window_len=30)
    pitches4 = smooth(pitches,window_len=40)

    plot(pitches1, 'pitches1')
    plot(pitches2, 'pitches2')
    plot(pitches3, 'pitches3')
    plot(pitches4, 'pitches4')
    plot(pitches, 'pitches')
    plot(magnitudes, 'magnitudes')
    plot( y, 'audio')

def main():
    #Set all wanted variables

    #we want a sample frequency of 16 000
    sample_f = 16000
    #The duration of the voice sample
    duration = 10
    #We want a windowsize of 30 ms
    window_time = 60
    fmin = 80
    fmax = 250
    #We want an overlap of 10 ms
    overlap = 20
    total_samples, window_size, needed_nb_windows, n_fft, hop_length = set_variables(sample_f, duration, window_time, fmin, fmax, overlap)


    # y = audio time series
    # sr = sampling rate of y
    y, sr = librosa.load('data/Audio/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav', sr=sample_f, duration=duration)
    #y1, sr1 = librosa.load('./1', sr=sample_f, duration=duration)

    analyse(y, sr, n_fft, hop_length, fmin, fmax)
    #analyse(y1, sr1, n_fft, hop_length, fmin, fmax)



if __name__ == "__main__":
    main()
