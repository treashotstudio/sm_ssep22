import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

file_path = 'soy.mp3' # 실습에 사용할 음악 파일
wav, sr = librosa.load(file_path)

fft = np.fft.fft(wav) 

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

n_fft = 2048 
hop_length = 512 

stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

fig = plt.figure(figsize = (14,5))
librosa.display.specshow(log_spectrogram, 
                         sr=sr, 
                         hop_length=hop_length,
                         x_axis='time',
                         y_axis='log')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.show()