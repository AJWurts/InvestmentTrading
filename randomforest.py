from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('full_data.csv')

train_X = df.iloc[:int(len(df)*.8), 1:-2]
train_y = df.iloc[:int(len(df)*.8), -2]

test_X = df.iloc[int(len(df)*.8)+1:, 1:-2]
test_y = df.iloc[int(len(df)*.8)+1:, -2]


forest = RandomForestClassifier()

forest.fit(train_X, train_y)

print(forest.feature_importances_)