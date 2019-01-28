import tensorflow as tf 
# from tensorflow import *
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow.keras.models import model_from_json



# df = pd.read_csv("./data/ml_training_data1.7mil.csv")
df = pd.read_csv("./data/ml_training_data_100.csv")

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

df.loc[df.bin == -1.0, 'bin'] = pd.Series(0.0, index=df.index[df.bin == -1.0])

n = int(len(df) * 0.8)
train = df[:n]
test = df[n:]

# model = keras.Sequential([
#     keras.layers.Dense(40, input_shape=(50,)),
#     keras.layers.Dense(25, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.softmax)
# ])

import tensorflow.keras.backend as K
def mean_error(y_true, y_pred):
    real_ret = y_true[-1] / y_true[0]
    pred_ret = y_pred[-1] / y_true[0]
    return abs(pred_ret - real_ret)


def mean_error_single(y_true, y_pred):
    return abs(y_true - y_pred)

co = 45
model = keras.Sequential([
    keras.layers.Embedding(100, 50, input_shape=(co,)),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.LSTM(32),
    # keras.layers.Dense(49, input_shape=(co,)),
    # keras.layers.Dense(1000),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(50-co)
])

model.summary()




# model.load_weights('model.h5')
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# vocabulary = 50
# hidden_size = 500
# model = keras.Sequential()
# # model.add(keras.layers.Embedding(vocabulary, hidden_size, input_length=50))
# model.add(keras.layers.Dense(50, input_shape=(50,)))
# model.add(keras.layers.LSTM(500, return_sequences=True))
# model.add(keras.layers.Dense(1))
# model.add(keras.layers.TimeDistributed(keras.layers.Dense(100)))
# model.add(keras.layers.Activation('softmax'))


new_train = np.array([val[:co] for val in train['data']])
new_y = np.array([val[co:] for val in train['data']])
# new_train = new_train.resh
# new_y = [val for val in train['bin'].values]

model.fit(new_train, new_y, epochs=5, batch_size=64)


new_test = np.array([val[:co] for val in test['data']])
new_test_y = np.array([val[co:] for val in test['data']])
test_loss, test_acc = model.evaluate(new_test, new_test_y)

predictions = model.predict(new_test, verbose=1)
for i, p in enumerate(predictions):
    print("Prev", np.sign(new_test[i][-1] - new_test[i][-5]),"Prediction: ", p, "Actual: ", new_test_y[i], p[0] - new_test_y[0])
    # print("Prediction Delta: ", p[-1] - new_test[i][-1], "Actual: ", new_test_y[i][-1] - new_test[i][-1])

print(test_acc, test_loss)

model.save_weights("model.h5")

