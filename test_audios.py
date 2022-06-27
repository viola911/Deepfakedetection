import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import glob
import numpy as np
from scipy.io.wavfile import read
import os

real_wavs = []
for filename in glob.glob('Original_audios/*.wav'):

    real_wavs.append(filename)

fake_wavs = []
for filename in glob.glob('Fake_audios/*.wav'):

    fake_wavs.append(filename)

spectrogram_fake = np.zeros((50,129,1968))
spectrogram_real =np.zeros((50,129,1968))
train=np.zeros((70,129,1968))
test=np.zeros((30,129,1968))
train_label=np.zeros((70))
test_label=np.zeros((30))
for i in range(len(real_wavs)):

    #Reading Real
    print(real_wavs[i])
    sample_rate, samples = wavfile.read(real_wavs[i])
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    np.append(spectrogram_real,spectrogram)

    #Reading Fake
    print(fake_wavs[i])
    sample_rate, samples = wavfile.read(fake_wavs[i])
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    np.append(spectrogram_fake,spectrogram)
for i in range(35):

    #train

    np.append(train,spectrogram_real[i])
    np.append(train_label, 1)


for i in range(35):
        # train

        np.append(train, spectrogram_fake[i])
        np.append(train_label, 0)
for i in range(35,50,1):

    #test

    np.append(test,spectrogram_real[i])
    np.append(test_label, 1)
for i in range(35,50,1):

    #test

    np.append(test,spectrogram_fake[i])
    np.append(test_label, 0)





plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram_real[i])
plt.imshow(spectrogram_fake[i])
#print(spectrogram_real.shape)
print(spectrogram_real[i])
print(train[i])

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()