# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:59:58 2022

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy import signal
import librosa, librosa.display
import pandas as pd
from scipy.fftpack import fft

x, sr = librosa.load("D:/Spyder-Kodlar/biyomedikal/data/train/sesler.wav")


# Normalleştirme yapıyoruz
x = x / np.power(2, 15)

############# Time Domain - 1
time_axis = 1 * np.arange(0, len(x), 1) / float(sr)
plt.plot(time_axis, x, color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Time Domain Signal')
plt.show()

############# Time Domain - 2 Plot
N = (60 - 0) * sr
time = np.linspace(0, 60, N)
plt.plot (time, x)
plt.title ('Time Domain Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()

############# Frequency Domain Plot
frequency = np.linspace (0.0, sr/2, int (N/2))
freq_data = fft(x)
y = 2/N * np.abs (freq_data [0:int (N/2)])
plt.plot(frequency[1000:1200], y[1000:1200])
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel("Amplitude")
plt.show()

#################### Extract features ##############################
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

######## Zooming in
n0 = 100
n1 = 200
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

####### zero crossing
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

####### spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')

####### spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

####### MFCC — Mel-Frequency Cepstral Coefficients
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title("MFCC")


#################### Frequency domain features ##############################
y, sr = librosa.core.load('D:/Spyder-Kodlar/biyomedikal/data/train/sesler.wav') # load samples and determine sampling rate

### Chroma related features
# using a power spectrum
chroma_d = librosa.feature.chroma_stft(y=y, sr=sr)
plt.figure(figsize=(15, 4))
librosa.display.specshow(chroma_d, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Power spectrum chromagram')
plt.tight_layout()
plt.show()

# using an energy (magnitude) spectrum
S = np.abs(librosa.stft(y)) # apply short-time fourier transform
chroma_e = librosa.feature.chroma_stft(S=S, sr=sr)
plt.figure(figsize=(15, 4))
librosa.display.specshow(chroma_e, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Energy spectrum chromagram')
plt.tight_layout()
plt.show()

# using a pre-computed power spectrogram with a larger frame
S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma_p = librosa.feature.chroma_stft(S=S, sr=sr)
plt.figure(figsize=(15, 4))
librosa.display.specshow(chroma_p, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Pre-computed power spectrogram with larger frame size')
plt.tight_layout()
plt.show()


########################### Time - Frequency features ######################
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

frequency_sampling, audio_signal = wavfile.read('D:/Spyder-Kodlar/biyomedikal/data/train/sesler.wav')

# Note that here we are taking first 15000 samples for analysis.
audio_signal = audio_signal[:15000]
# Use the MFCC techniques and execute the following command to extract the MFCC features
features_mfcc = mfcc(audio_signal, frequency_sampling)


print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# Now, plot and visualize the MFCC features using the commands given below 
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# In this step, we work with the filter bank features as shown −
# Extract the filter bank features
filterbank_features = logfbank(audio_signal, frequency_sampling)

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

# Now, plot and visualize the filterbank features.
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()


