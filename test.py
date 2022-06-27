import numpy as np
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Dropout, Flatten
#from keras.layers.core import Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201
from keras.models import Model
#from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.inception_v3 import InceptionV3

from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing import image
from keras import optimizers
import cv2
#from sklearn.utils import class_weight
import tensorflow as tf
#from openpyxl import load_workbook
import datetime
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import glob
import numpy as np

real_wavs = []
for filename in glob.glob('Original_audios/*.wav'):

    real_wavs.append(filename)

fake_wavs = []
for filename in glob.glob('Fake_audios/*.wav'):

    fake_wavs.append(filename)

spectrogram_fake = np.zeros((50,129,1968))
spectrogram_real =np.zeros((50,129,1968))
train=np.zeros((80,129,1968))
test=np.zeros((20,129,1968))
train_label=np.zeros((80))
test_label=np.zeros((20))
for i in range(len(real_wavs)):
    #Reading Real
    print(real_wavs[i])
    sample_rate, samples = wavfile.read(real_wavs[i])
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    spectrogram_real[i] = spectrogram

    #Reading Fake
    print(fake_wavs[i])
    sample_rate, samples = wavfile.read(fake_wavs[i])
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    spectrogram_fake[i] = spectrogram
for i in range(40):

    #train

    train[i] = spectrogram_real[i]
    train_label[i] = 1
    train[i] = spectrogram_fake[i]
 


for i in range(40):
        # train

       # train[i] = spectrogram_fake[i]
        train_label[i] = 0

j = 0
for i in range(40,50,1):

    #test
    test[j] = spectrogram_real[i]
    test_label[j] = 1
    j += 1

for i in range(40,50,1):

    #test

    test[j] = spectrogram_fake[i]
    test_label[j] = 0
    j += 1


baseModel = VGG16(weights=None, include_top=False, input_tensor=None, input_shape=(129, 1968,1))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
   layer.trainable = False


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train, train_label, epochs=5)

test_loss, test_acc = model.evaluate(test, test_label)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

pred = model.predict(test)
print(pred)
max_predictions = tf.argmax(pred, axis=1)
print(max_predictions)
print(tf.math.confusion_matrix(labels=test_label, predictions=max_predictions))
