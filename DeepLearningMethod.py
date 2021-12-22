import pickle
import keras
import tensorflow
import keras
from keras.models import Sequential #顺序模型
from keras.layers import Activation, Dense
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tqdm import tqdm 
from sklearn.model_selection import KFold

with open("train_x.pkl", "rb") as f:
    train_x = pickle.load(f)
f.close()

with open("train_y.pkl", "rb") as f:
    train_y = pickle.load(f)
f.close()

#train_X = []
#for i in train_x:
#    train_X.append(i.flatten())
#train_X = np.array(train_X)
# print(train_X.shape)
def get_model(input_shape):
    model = Sequential([
            Flatten(input_shape=(input_shape)),
            Dense(256,'relu'),
            Dropout(0.3),
            Dense(1, "sigmoid"),
            ])

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


#model = get_model(train_x.shape[1:])
#model.fit(train_x, train_y, batch_size=16, epochs=100,  validation_split = 0.1, verbose = 1)
# history = model.fit(train_x, train_y, batch_size=16, epochs=100,  validation_split = 0.1, verbose = 1)
n_split = 10
split_max_accruacy = []
for train_index, test_index in tqdm(KFold(n_split).split(train_x)):
    model = get_model(train_x.shape[1:])
    x_train, x_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=100, batch_size=16,
                                verbose=0)
    # split_min_loss.append(np.min(history.history["val_loss"]))
    split_max_accruacy.append(np.max(history.history["val_accuracy"]))

print(split_max_accruacy)
print(np.mean(split_max_accruacy))
