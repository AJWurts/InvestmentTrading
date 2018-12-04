from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump

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
print(min_length)
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

correct = 0
total = 0

print(bc.score(test_X, test_y))

# print(bc.score(train_X, train_y))
print(bc.predict(train_X))

# print(bc.score(train)
# results = bc.predict(test_X) == test_y
print(bc.predict(test_X))
print(test_y)
# correct = len(results[results == True]) / len(results)
# print("Test Accuracy", correct)

# results = bc.predict(train_X) == train_y
# correct = len(results[results == True]) / len(results)
# print("Train Accuracy: ", correct)
