import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
import sys

def use_all_data():
  tickers = ['AMD', 'SPY', "NVDA", "AAPL", "ASNA", "BP", "CZR", "DE", "DF", "ETSY", "FTR", "GE", "GPK", "JCP", "JD", "MDB", "MHLD", "MSFT", "PEIX", "SPG", "TA", "TSLA" ]
  X = []
  y = []
  for ticker in tickers:

    df = pd.read_csv('./data/training_' + ticker + '.csv')
    for i in range(len(df)):
      c = df['data'][i]
      if type(c) == float:
        continue
      for char in '[]\n\r':
        c = c.replace(char, '')
      
      result = np.array(c.split(' '), dtype=str)
      result = result[result != '']
      result = list(map(lambda x: float(x), result))

      X.append(result)
      y.append(df['bin'][i])

  min_length = min([len(d) for d in X])
  with open('./mlsize/' + ticker + "mlsize.txt", 'w') as output:
    output.write(str(min_length))
  X = [x[:min_length] for x in X]

  train_X = X
  train_y = y


  forest = RandomForestClassifier(n_estimators=5000, criterion='entropy', class_weight='balanced_subsample', max_features='auto')

  bc = BaggingClassifier(base_estimator=forest, n_estimators=20)

  bc.fit(train_X, train_y)

  dump(bc, './machinelearning/saved_classifiers/randomforest_' + ticker + '.joblib')


  # print(bc.score(test_X, test_y))


  return bc


def forestgenerator(ticker=None, bins=None):
  if bins is not None:
    df = bins
  elif ticker is not None:
    df = pd.read_csv('./data/training_' + ticker + '.csv')
  else:
    exit("Need to give a valid Stock ticker")
  X = []
  y = []
  for i in range(len(df)):
    c = df['data'][i]
    if type(c) == float:
      continue
    for char in '[]\n\r':
      c = c.replace(char, '')
    
    result = np.array(c.split(' '), dtype=str)
    result = result[result != '']
    result = list(map(lambda x: float(x), result))

    X.append(result)
    y.append(df['bin'][i])

  min_length = min([len(d) for d in X])
  with open('./mlsize/' + ticker + "mlsize.txt", 'w') as output:
    output.write(str(min_length))
  X = [x[:min_length] for x in X]

  train_X = X
  train_y = y



  forest = RandomForestClassifier(n_estimators=2000, criterion='entropy', class_weight='balanced_subsample', max_features='auto')

  bc = BaggingClassifier(base_estimator=forest, n_estimators=4)

  bc.fit(train_X, train_y)

  dump(bc, './machinelearning/saved_classifiers/randomforest_' + ticker + '.joblib')


  # print(bc.score(test_X, test_y))


  return bc

if __name__ == "__main__":
  if len(sys.argv) > 1:
    print("Training Random Forest")
    use_all_data()
    # forestgenerator(sys.argv[1])