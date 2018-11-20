from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


train_X = X[:int(len(X)*.8)]
train_y = df['bin'][:int(len(df)*.8)]

test_X = X[int(len(X)*.8)+1:]
test_y = df['bin'][int(len(df)*.8)+1:]


forest = RandomForestClassifier(n_estimators=1, criterion='entropy', class_weight='balanced_subsample', max_features='auto')

bc = BaggingClassifier(base_estimator=forest, n_estimators=1000)

bc.fit(train_X, train_y)

correct = 0
total = 0

print(bc.score(test_X, test_y))

# print(bc.score(train)
# results = bc.predict(test_X) == test_y
# correct = len(results[results == True]) / len(results)
# print("Test Accuracy", correct)

# results = bc.predict(train_X) == train_y
# correct = len(results[results == True]) / len(results)
# print("Train Accuracy: ", correct)
