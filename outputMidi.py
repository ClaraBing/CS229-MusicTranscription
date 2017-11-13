from midiutil.MidiFile import MIDIFile
import numpy as np

# Input: N number of notes,
# frequencies: array of size N with frequencies in Hz
# output_name: name of the file to be saved
# duration of each notes in s. 
def outputMIDI(N, frequencies, output_name,  duration = 1):
    # Creates a MIDI file with one track
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    MyMIDI.addTrackName(track, time, output_name)
    MyMIDI.addTempo(track,time,120)
    for i in range(N):
        midiNote = int(round(21 + 12 * np.log(frequencies[i]/ 27.5) / np.log(2)))
        MyMIDI.addNote(track, 0, midiNote, time, duration, 100)
        time += duration

    binfile = open(output_name+ ".mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

# outputMIDI(5, [261.3, 293.66, 329.63, 349.23, 392.0], "doremifasol")
