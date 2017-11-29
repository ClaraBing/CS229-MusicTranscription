"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Generate time slices of spectrogram, which will be used for CNN training
def wav2spec_data(audioName, fext):
    sr = 44100 # sampling rate
    y, sr = librosa.load(audioName+'.'+fext, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000, n_mels=256)
    spec = librosa.power_to_db(S, ref=np.max)

    for i in range(spec.shape[1]):
        plt.figure(figsize=(1, 4)) # 100 * 400 pixels
        plt.subplot(111)
        librosa.display.specshow(spec[:, i:i+1], sr=sr)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis('off')
        plt.savefig('spec_{:s}_{:d}.png'.format(audioName, i))
        plt.close()

# Generate a spectrogram for the entire audio for demo
def wav2spec_demo(audioName, fext):
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
    plt.savefig('spec_' + audioName + '.png')
    plt.close()

if __name__ == '__main__':
    audioName, file_ext = "Phoenix_ScotchMorris_MIX", 'wav'
    wav2spec_data(audioName, file_ext)