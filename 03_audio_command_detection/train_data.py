# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:47:05 2022

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

frequency_sampling, audio_signal = wavfile.read("C:/Users/buseb/Desktop/Audio Recognition/codes/sounds.wav")

# Normalleştirme yapıyoruz
audio_signal = audio_signal / np.power(2, 15)


############# Time Domain - 1
time_axis = 1 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling)
plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Time Domain Signal')
plt.show()

############# Time Domain - 2 Plot
N = (60 - 0) * frequency_sampling
time = np.linspace(0, 60, N)
plt.plot (time, audio_signal)
plt.title ('Time Domain Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()

############# Frequency Domain Plot
frequency = np.linspace (0.0, frequency_sampling/2, int (N/2))
freq_data = fft(audio_signal)
y = 2/N * np.abs (freq_data [0:int (N/2)])
plt.plot(frequency[1000:1200], y[1000:1200])
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel("Amplitude")
plt.show()


############# Sinyalin STFT sini buluyoruz ve çizdiriyoruz.
sr = 44100
hop_length = 512
n_fft = 2048
X = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
S = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15, 5))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')

############# zero array oluşturuyoruz. Labellar için.
arr = np.zeros( (len(S[1]), ) , dtype=np.int64)
#5168/60 = 86.1 , stft grafiğinde nokta belirle(x) , sonra 86 ile çarp

#Buse'nin sesi ## Datayı labellıyoruz. Manuel olarak.
arr[254:324]=0 
arr[457:521]=1 
arr[663:724]=2 
arr[844:897]=3 
arr[1038:1085]=4 
arr[1206:1261]=5 

#Kardelen'in sesi ## Datayı labellıyoruz. Manuel olarak.
arr[1473:1530]=0 
arr[1655:1710]=1 
arr[1886:1929]=2 
arr[2081:2143]=3 
arr[2268:2325]=4 
arr[2428:2482]=5 

# Seher'in sesi ## Datayı labellıyoruz. Manuel olarak.
arr[2850:2936]=0 
arr[3074:3125]=1 
arr[3272:3327]=2 
arr[3444:3499]=3 
arr[3588:3649]=4 
arr[3733:3797]=5 

#Ece'nin sesi ## Datayı labellıyoruz. Manuel olarak.
arr[4073:4150]=0 
arr[4309:4354]=1 
arr[4492:4534]=2 
arr[4650:4707]=3 
arr[4813:4865]=4 
arr[4992:5042]=5 

################## Gereksiz yerleri siliyoruz. 40 saniyeden sonrasını
copy_S = S
a_del = np.delete(copy_S, slice(5080,5168), 1)
print(a_del)

############## Gereksiz yerleri siliyoruz. 40 saniyeden sonrasını
copy_arr = arr
a1_del = np.delete(copy_arr, slice(5080,5168), 0)
print(a_del)
a_del_transpose = np.transpose(a_del)
## X ve y'mizi tanımlıyoruz.
X = a_del_transpose
y = a1_del

# data kaydetme
"""
df_data = pd.DataFrame(data=X)
df_data.to_csv(r'D:\Spyder-Kodlar\biyomedikal\final-project\dataset.csv',index=False,header=True)
"""

######################## MACHINE LEARNING ########################
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(5)

validation_size = 0.25
seed = 5

# Datayı %20 test %80 train olacak şekilde ayırıyoruz.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = validation_size,random_state = seed)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test) # y_test.values

class Model(nn.Module):
    def __init__(self, in_features=1025, out_features=6):
        super().__init__()
        self.fc1=nn.Linear(in_features, 1600)
        self.fc2=nn.Linear(1600, 64)
        self.out=nn.Linear(64, out_features)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x



model=Model()

criterion=nn.CrossEntropyLoss()  # loss function seçtik
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)  # optim adlı optimizerı seçtik

epochs=100
losses=[]
for i in range(epochs):
    y_pred=model.forward(X_train)
    loss=criterion(y_pred, y_train)
    losses.append(loss)
    if i%10==0:
        print(f'Epoch {i} and loss is: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    y_eval=model.forward(X_test)
    loss=criterion(y_eval, y_test)
    
correct=0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val=model.forward(data)
        print(f'{i+1}. {str(y_val)}   {y_test[i]}')
        
        if y_val.argmax().item()==y_test[i]:
            correct+=1
            
print(f'Doğru sayısı: {correct}')

percent = (correct/1270)*100
print(percent)

torch.save(model.state_dict(), "audio_recognition.pt")
