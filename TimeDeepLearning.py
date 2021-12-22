import pickle
import keras
import tensorflow
import keras
from keras.models import Sequential #顺序模型
from keras.layers import Activation, Dense
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM
from sklearn.model_selection import KFold
from tqdm import tqdm

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
def get_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape = input_shape),
            Dense(256,'relu'),
            Dropout(0.3),
            Dense(1, "sigmoid"),
            ])

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


n_split = 10
split_max_accruacy = []
for train_index, test_index in tqdm(KFold(n_split).split(train_x)):
    model = get_model(train_x.shape[1:])
    x_train, x_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True, epochs=150, batch_size=32,
                                verbose=0)
    # split_min_loss.append(np.min(history.history["val_loss"]))
    split_max_accruacy.append(np.max(history.history["val_accuracy"]))

print(split_max_accruacy)
print(np.mean(split_max_accruacy))
