import tensorflow as tf 
# from tensorflow import *
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow.keras.models import model_from_json



# df = pd.read_csv("./data/ml_training_data1.7mil.csv")
df = pd.read_csv("./data/ml_training_data_1000000.csv")

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
X = [x[:50] for x in X if len(x) >= 50]
print(min_length)

df['data'] = X

df.loc[df.bin == -1.0, 'bin'] = df[df.bin == -1.0]['bin'] * 0

n = int(len(df) * 0.8)
train = df[:n]
test = df[n:]

model = keras.Sequential([
    keras.layers.Dense(40, input_shape=(50,)),
    keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='mean_squared_error',
                metrics=['accuracy'])



model.load_weights('model.h5')
# model = keras.Sequential([
#     keras.layers.Dense(50, input_shape=(50,)),
#     keras.layers.LSTM(4),
#     #, activation=tf.nn.softmax
#     # keras.layers.Dense(2, activation=tf.nn.softmax),
#     # keras.layers.TimeDistributed(keras.layers.Dense(50))
#     # keras.layers.Dense()
# ])

# vocabulary = 50
# hidden_size = 500
# model = keras.Sequential()
# # model.add(keras.layers.Embedding(vocabulary, hidden_size, input_length=50))
# model.add(keras.layers.Dense(50, input_shape=(50,)))
# model.add(keras.layers.LSTM(500, return_sequences=True))
# model.add(keras.layers.Dense(1))
# model.add(keras.layers.TimeDistributed(keras.layers.Dense(100)))
# model.add(keras.layers.Activation('softmax'))



new_train = np.array([val for val in train['data']])
# new_train = new_train.resh
new_y = [val for val in train['bin']]


model.fit(new_train, new_y, epochs=15)


new_test = np.array([val for val in test['data']])
new_test_y = [val for val in test['bin']]
test_loss, test_acc = model.evaluate(new_test, new_test_y)

print(test_acc, test_loss)

model.save_weights("model.h5")

