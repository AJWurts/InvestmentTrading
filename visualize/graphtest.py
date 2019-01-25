import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import pandas as pd
# Create random data with numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from machinelearning.fracdiff import getFracDiffCSV, fracDiff


df = pd.read_csv('./data/ml_training_0009.csv')
# df = getFracDiffCSV('./data/ml_training_0009.csv')



X = []
for i in range(len(df)):
  c = df['data'][i]
  for char in '[]\n\r':
    c = c.replace(char, '')
  
  result = np.array(c.split(' '), dtype=str)
  result = result[result != '']
  result = list(map(lambda x: float(x), result))

  X.append(np.array(result))
X = np.array(X)
grad = (X[:,1:] - X[:,:-1]) / X[:,:-1]
# abovePoint1 = df
# abovePoint1.plot(x='initret', y='ret', kind='scatter')
clusters = 5
clf = KMeans(n_clusters=clusters)

kmeans = clf.fit(X)

predictions = kmeans.predict(X)
unique = np.unique(predictions, return_counts=True)

data = [[0,0] for i in range(clusters)]
for i, p in enumerate(predictions):
  if df['ret'][i] > 0:
    data[p][0] += 1
  elif df['ret'][i] <= 0:
    data[p][1] += 1

print(data)


# print(unique[1])
# for i, p in enumerate(predictions[:10]):
#   print("Class: ", p)
#   plt.subplot(10,1,i+1)
#   plt.plot(grad[i])
#   plt.title(p)

# plt.show()

# countAbove = len(abovePoint1[abovePoint1.ret > 0])
# belowCount = len(abovePoint1[abovePoint1.ret < 0])
# print("Above Count: ", countAbove)
# print('below count: ', belowCount)
# plt.show()