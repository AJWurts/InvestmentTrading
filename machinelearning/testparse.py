import pandas as pd
import numpy as np


df = pd.read_csv('./data/ml_training_0009.csv')
X = []
for i in range(len(df)):
    c = df['data'][i]
for char in '[]\n\r':
    c = c.replace(char, '')

result = np.array(c.split(' '), dtype=str)
result = result[result != '']
result = list(map(lambda x: float(x), result))

X.append(result)

min_length = min([len(d) for d in X])
X = [x[:min_length] for x in X]


print(result)
