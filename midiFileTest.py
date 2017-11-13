from midiutil.MidiFile import MIDIFile

MyMIDI = MIDIFile(1)
track = 0
time = 0
MyMIDI.addTrackName(track,time,"Sample Track")
MyMIDI.addTempo(track,time,120)
MyMIDI.addNote(track,0,100,0,1,100)

binfile = open("output.mid", 'wb')
MyMIDI.writeFile(binfile)
binfile.close()
