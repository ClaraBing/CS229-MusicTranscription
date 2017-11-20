"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""


import os
import wave

import pylab

save_dir = "../Spectrogram/"


def graph_spectrogram(wav_file):
    parts = wav_file.split("/")
    file_name = parts[len(parts)-1].split('.')[0]
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(save_dir+file_name+'.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    frames = frames[:int(len(frames)/50)]
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


if __name__ == '__main__':
    wav_file = 'data/Audio/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav'
    graph_spectrogram(wav_file)
