# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:36:27 2021

@author: DELL
"""
"""
###### TRAIN için
import sounddevice as sd
import soundfile as sf


##### TRAIN için 60 sn lik veri kaydetme
train_audio_path = 'C:/Users/buseb/Desktop/Audio Recognition/codes/'

#'D:/Spyder-Kodlar/biyomedikal/data/train/'

samplerate = 44100
duration = 15
filename = train_audio_path + 'sesler.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)


"""
##### TEST için 2 _sn lik veri kaydetme
import sounddevice as sd
import soundfile as sf

train_audio_path = 'C:/Users/buseb/Desktop/Audio Recognition/codes/'

samplerate = 44100
duration = 2
filename = train_audio_path + 'move1.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)