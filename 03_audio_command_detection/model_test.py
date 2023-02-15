# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:40:42 2022

@author: DELL
"""
import pickle
import sounddevice as sd
import numpy as np
import librosa, librosa.display
from sklearn.preprocessing import StandardScaler

###################### CANLI KAYIT TEST ###########################
# liste içinde en çok tekrar eden sayıyı buluyoruz.
def most_frequent(List):
    return max(set(List), key = List.count)

# modeli yüklüyoruz.
loaded_model = pickle.load(open('D:/Spyder-Kodlar/biyomedikal/final-project/codes/KNN_scaled_model.sav', 'rb'))
#loaded_model = pickle.load(open('D:/Spyder-Kodlar/biyomedikal/final-project/models/RF_model.sav', 'rb'))
#loaded_model = pickle.load(open('D:/Spyder-Kodlar/biyomedikal//model/denemeler/knn_model.sav', 'rb'))
samplerate = 44100
duration = 2
# 2 saniyelik kayıt başlatıyoruz bu süre zarfında konuşmanız gerekir.(ileri,geri,sağ,sol gibi)
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
print("--------------")
audio_signal = np.reshape(mydata, (len(mydata),))

# STFT uyguluyoruz.
sr = 44100
hop_length = 512
n_fft = 2048
X = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
S = librosa.amplitude_to_db(abs(X))
# ML algoritması için transposunu alıyoruz
s_transpose = np.transpose(S)

"""
# test verisinin çıkarımı
scaler = StandardScaler().fit(s_transpose)
X_test_rescaled = scaler.transform(s_transpose)
"""

# model tahminleme yapıyor
y_predict = loaded_model.predict(s_transpose)


# 0 dışındaki değerleri listeye atıyoruz.
none_zero = []
for i in y_predict: #  and i!=6
    if i !=0:
        none_zero.append(i)   
none_zero.append(0)
# predictionlar içinde en çok tekrar eden değerimiz bizim sonucumuz oluyor. Yani komut'umuz.
command = most_frequent(none_zero)
# komut değeri 1,2,3,4,5,6 ise aşşağıdaki değerleri alıyor. Değil ise tekrar deneyin yazdırıyoruz.
if command == 1:
    print("ileri")
elif command == 2:
    print("geri")
elif command == 3:
    print("sağ")
elif command == 4:
    print("sol")
elif command == 5:
    print("dur")
elif command == 6:
    print("çalış")
else:
    print("Lütfen Tekrar Söyleyin")


print(np.mean(none_zero))


##################### KAYDEDİLMİŞ VERİ İLE TEST ###########################
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler

frequency_sampling_1, audio_signal_ileri = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/ileri.wav")
frequency_sampling_1, audio_signal_geri = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/geri.wav")
frequency_sampling_1, audio_signal_sag = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/sağ.wav")
frequency_sampling_1, audio_signal_sol = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/sol.wav")
frequency_sampling_1, audio_signal_dur = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/dur.wav")
frequency_sampling_1, audio_signal_calıs = wavfile.read("D:/Spyder-Kodlar/biyomedikal/data/test/çalış.wav")

test_datas = []
test_datas.append(('ileri',audio_signal_ileri))
test_datas.append(('geri',audio_signal_geri))
test_datas.append(('sağ',audio_signal_sag))
test_datas.append(('sol',audio_signal_sol))
test_datas.append(('dur',audio_signal_dur))
test_datas.append(('çalış',audio_signal_calıs))

known_results = ["ileri","geri","sağ","sol","dur","çalış"]
results = []
for dataName, data in test_datas:
    # Normalleştirme yapıyoruz
    data = data / np.power(2, 15)
    # STFT uyguluyoruz.
    sr = 44100
    hop_length = 512
    n_fft = 2048
    X = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X))
    # ML algoritması için transposunu alıyoruz
    s_transpose = np.transpose(S)
    
    # model tahminleme yapıyor
    y_predict = loaded_model.predict(s_transpose)
    # 0 dışındaki değerleri listeye atıyoruz.
    none_zero = []
    for i in y_predict: #  and i!=6
        if i !=0:
            none_zero.append(i)   
    none_zero.append(0)
    # predictionlar içinde en çok tekrar eden değerimiz bizim sonucumuz oluyor. Yani komut'umuz.
    command = most_frequent(none_zero)
    # komut değeri 1,2,3,4,5,6 ise aşşağıdaki değerleri alıyor. Değil ise tekrar deneyin yazdırıyoruz.
    
    if command == 1:
        results.append("ileri")
    elif command == 2:
        results.append("geri")
    elif command == 3:
        results.append("sağ")
    elif command == 4:
        results.append("sol")
    elif command == 5:
        results.append("dur")
    elif command == 6:
        results.append("çalış")
    else:
        results.append("boş")
    











