import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.utils import np_utils
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras import metrics
import plotly.plotly as py
import plotly.tools as tls


train = pd.read_csv('./input/train.csv')

TestX = (pd.read_csv('./input/test.csv').values).astype(float)
labels = train.ix[1:,0].values
TrainX = (train.ix[1:,1:].values).astype(float)
TrainY = np_utils.to_categorical(labels)

TrainX /= 255
TrainX -= np.std(TrainX)


from sklearn.decomposition import PCA
pca = PCA(n_components=256)
pca.fit(TrainX)
TrainX = pca.transform(TrainX)

print(TrainX.shape)





TrainX= TrainX.reshape(TrainX.shape[0],16,16,1)

model = Sequential()
# 4x4 window
model.add(Conv2D(64, kernel_size=(4,4),
                 activation='relu',
                 input_shape=(16,16,1)))
# Dropout for regularization
model.add(Dropout(0.20))
# Reduce kernel size
model.add(Conv2D(32, kernel_size=(2, 2),activation='relu',))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(TrainY.shape[1],activation  = 'softmax'))
early_stopping_monitor = EarlyStopping(monitor="acc",min_delta=0.3,patience=2)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.fit(TrainX,TrainY, epochs = 64, verbose = 2, batch_size =256)




TestX /= 255
TestX -= np.std(TrainX)
TestX = pca.transform(TestX)
TestX= TestX.reshape(TestX.shape[0],16,16,1)
Prediction = model.predict_classes(TestX , verbose = 2)



pd.DataFrame({"ImageId": list(range(1,len(Prediction)+1)), "Label": Prediction}).to_csv("submissionC.csv", index=False, header=True)
