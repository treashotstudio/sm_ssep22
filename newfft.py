import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

file_path = '4m 1500Hz.m4a' # 실습에 사용할 음악 파일
wav, sr = librosa.load(file_path)

fft = np.fft.fft(wav) 

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

fig = plt.figure(figsize = (14,5))
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()