import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("./ml_training_data.csv")

X = []
for i in range(len(df)):
    c = df['data'][i]
    for char in '[]\n\r':
        c = c.replace(char, '')

    result = np.array(c.split(' '), dtype=str)
    result = result[result != '']
    result = [_ for _ in map(lambda x: float(x), result)]

    X.append(result)

min_length = min([len(d) for d in X])
X = [x[:min_length] for x in X if len(x) >= 13]
print(min_length)

df['data'] = X

df.loc[df.bin == -1.0, 'bin'] = df[df.bin == -1.0]['bin'] * 0

n = int(len(df) * 0.8)
train = df[:n]
test = df[n:]

# train[train.bin == -1.0] = pd.Series(0, index=train.index)
# train[train.bin == 1.0] = pd.Series(1, index=train.index)

model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(1, 50)),
    # keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])



new_train = np.array([[val] for val in train['data']])
new_y = [val for val in train['bin']]

# print(new_train)
model.fit(new_train, new_y, epochs=100)


new_test = np.array([[val] for val in test['data']])
test_loss, test_acc = model.evaluate(new_test, test['bin'].values)