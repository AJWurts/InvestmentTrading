import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sampling import cumsum

def f(x):
    if x > 0.5:
        return 1
    else:
        return 0

df = pd.read_csv('full_data.csv')

train_X = df.iloc[:int(len(df)*.8), 1:-2]
train_y = df.iloc[:int(len(df)*.8), -2]

test_X = df.iloc[int(len(df)*.8)+1:, 1:-2]
test_y = df.iloc[int(len(df)*.8)+1:, -2]

# Regression
# logreg = LinearRegression()

# logreg.fit(train_X, train_y)

# pred_X = logreg.predict(test_X)

# Random Forest
forest = RandomForestClassifier()

forest.fit(train_X, train_y)

pred_X = forest.predict(test_X)

pred_and_index = [[pred_X[i], df['index'][i]] for i in range(len(test_X))]

sell_p = np.array(list(filter(lambda x: x[0] <= 0.5, pred_and_index)))
buy_p = np.array(list(filter(lambda x: x[0] > 0.5, pred_and_index)))

df = pd.read_csv('SPY.csv')
# buy = pd.read_csv('training_data_buy.csv')
# sell = pd.read_csv('training_data_sell.csv')




# # Ideal Data
# x = [x for x in map(lambda x: x[1], buy_p)]
# y = [df['Close'][arr[1]] for arr in buy_p]
# plt.scatter(x, y, color='green')

# x_sell = [x for x in map(lambda x: x[1], sell_p)]
# y_sell = [df['Close'][arr[1]] for arr in sell_p]
plt.plot([i for i in range(len(df))], df['Close'],  color='black', linewidth=3)


points = cumsum(df, 0.01, returnI=True)
x = [x for x in points]
y = [df['Close'][x] for x in points]
plt.scatter(x, y, color='red')




# plt.scatter(x_sell, y_sell, color='red')

# # Predicted Data
# plt.scatter(buy['index'], [df['Close'][index] for index in buy['index']], color='green')
# plt.scatter(sell['index'], [df['Close'][index] for index in sell['index']], color='red')


# Plot outputs


plt.xticks(())
plt.yticks(())


plt.show()