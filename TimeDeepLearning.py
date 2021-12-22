import pickle
import keras
import tensorflow
import keras
from keras.models import Sequential #顺序模型
from keras.layers import Activation, Dense
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM

with open("time_train_x.pkl", "rb") as f:
    train_x = pickle.load(f)
f.close()

with open("time_train_y.pkl", "rb") as f:
    train_y = pickle.load(f)
f.close()


#train_X = []
#for i in train_x:
#    train_X.append(i.flatten())
#train_X = np.array(train_X)
# print(train_X.shape)
model = Sequential([
    LSTM(32, input_shape = train_x.shape[1:]),
          Dense(256,'relu'),
          Dropout(0.3),
          Dense(1, "sigmoid"),
          ])

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=16, epochs=100,  validation_split = 0.1, verbose = 1)

