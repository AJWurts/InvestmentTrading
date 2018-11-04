from sklearn.linear_model import LinearRegression, LogisticRegression

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('full_data.csv')

train_X = df.iloc[:int(len(df)*.8), 1:-2]
train_y = df.iloc[:int(len(df)*.8), -2]

test_X = df.iloc[int(len(df)*.8)+1:, 1:-2]
test_y = df.iloc[int(len(df)*.8)+1:, -2]


linreg = LogisticRegression()

linreg.fit(train_X, train_y)

print(linreg.score(train_X, train_y))
print(linreg.score(test_X, test_y))


print(len(train_X), len(train_y))
# Plot outputs

# plt.scatter(train_X, train_y,  color='black')
# plt.plot(train_X, pred_X, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()