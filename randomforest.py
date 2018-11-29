from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump


def forestgenerator(bins=None):
  if bins is not None:
    df = bins
  else:
    df = pd.read_csv('ml_training_data.csv')
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


  split = 0.9
  train_X = X[:int(len(X)*split)]
  train_y = df['bin'][:int(len(df)*split)].values

  test_X = X[int(len(X)*split)+1:]
  test_y = df['bin'][int(len(df)*split)+1:].values


  forest = RandomForestClassifier(n_estimators=1, criterion='entropy', class_weight='balanced_subsample', max_features='auto')

  bc = BaggingClassifier(base_estimator=forest, n_estimators=1000)

  bc.fit(train_X, train_y)

  dump(bc, 'randomforest.joblib')


  print(bc.score(test_X, test_y))


  return bc

if __name__ == "__main__":
  forestgenerator()