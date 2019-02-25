from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump


def forestgenerator(bins=None):
  if bins is not None:
    df = bins
  else:
    df = pd.read_csv('../data/training_UNH.csv')
  X = []
  for i in range(len(df)):
    c = df['data'][i]
    for char in '[]\n\r':
      c = c.replace(char, '')
    
    result = np.array(c.split(' '), dtype=str)
    result = result[result != '']
    result = list(map(lambda x: float(x), result))

    # result = [result[i:i+4] for i in range(0, len(result), 4)]
    X.append(result)

  min_length = min([len(d) for d in X])
  X = [x[:min_length] for x in X]


  testing_size = 60
  train_X = X
  train_y = df['bin'].values

  # test_X = X[int(len(X)*split)+1:]
  # test_y = df['bin'][int(len(df)*split)+1:].values


  forest = RandomForestClassifier(n_estimators=3, criterion='entropy', class_weight='balanced_subsample', max_features='auto')

  bc = BaggingClassifier(base_estimator=forest, n_estimators=50)

  bc.fit(train_X, train_y)

  dump(bc, './saved_classifiers/randomforest_unh.joblib')


  # print(bc.score(test_X, test_y))


  return bc

if __name__ == "__main__":
  forestgenerator()