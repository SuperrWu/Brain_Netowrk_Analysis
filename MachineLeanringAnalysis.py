import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import math
from collections import Counter

with open("train_x.pkl", "rb") as f:
    train_x = pickle.load(f)
f.close()

with open("train_y.pkl", "rb") as f:
    train_y = pickle.load(f)
f.close()


def format_train(data):
    train_X = []
    for i in data:
        train_X.append(i.flatten())
    train_X = np.array(train_X)
    return train_X

print(train_x.shape)
train_x = train_x[:, 17:42, 17:42]
print(train_x.shape)
train_x = format_train(train_x)
print(train_x.shape)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.333, random_state=27)
clf = RandomForestClassifier(max_depth=10, random_state=36)
clf.fit(X_train, y_train)
importance_channel = []
top_100_index = clf.feature_importances_.argsort()[::-1][:100]
for i in top_100_index / 25:
    importance_channel.append(math.floor(i))
# print(importance_channel)
results = Counter(importance_channel)
print(results)
