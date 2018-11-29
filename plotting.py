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

# df = pd.read_csv('full_data.csv')

# train_X = df.iloc[:int(len(df)*.8), 1:-2]
# train_y = df.iloc[:int(len(df)*.8), -2]

# test_X = df.iloc[int(len(df)*.8)+1:, 1:-2]
# test_y = df.iloc[int(len(df)*.8)+1:, -2]

# Regression
# logreg = LinearRegression()

# logreg.fit(train_X, train_y)

# pred_X = logreg.predict(test_X)

# Random Forest
# forest = RandomForestClassifier()

# forest.fit(train_X, train_y)

# pred_X = forest.predict(test_X)a

# pred_and_index = [[pred_X[i], df['index'][i]] for i in range(len(test_X))]

# sell_p = np.array(list(filter(lambda x: x[0] <= 0.5, pred_and_index)))
# buy_p = np.array(list(filter(lambda x: x[0] > 0.5, pred_and_index)))


from fracdiff import fracDiff
df = pd.read_csv('./data/SPY.csv')
df = df.set_index('Date')
df = df.drop(['Volume', 'Adj Close', 'High', 'Open', 'Low'], axis=1)
ds = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]
_, axs = plt.subplots(len(ds), 1, sharex=True)
for i, algo in enumerate(ds):
    df_ = fracDiff(df, d=algo)
    # print(result, history)
    axs[i].set_title("d: " + str(algo))
    df_.plot(ax=axs[i])
plt.show()
# buy = pd.read_csv('training_data_buy.csv')
# sell = pd.read_csv('training_data_sell.csv')




# # Ideal Data
# x = [x for x in map(lambda x: x[1], buy_p)]
# y = [df['Close'][arr[1]] for arr in buy_p]
# plt.scatter(x, y, color='green')

# x_sell = [x for x in map(lambda x: x[1], sell_p)]
# y_sell = [df['Close'][arr[1]] for arr in sell_p]
# df.reset_index().plot(x='index', y='Close', ax=plt.subplot(111))

# df.plot()
# # df.plot(x='index', y='Close')
# # plt.plot([i for i in range(len(df))], df['Close'],  color='black', linewidth=3)

# plt.xlabel("5 Year Spy Data")
# plt.ylabel("Price Movement")
# # points = cumsum(df['Close'], 0.05)
# # x = [x for x in points]
# # y = [df['Close'][x] for x in points]
# # plt.scatter(x, y, color='red')




# # plt.scatter(x_sell, y_sell, color='red')

# # # Predicted Data
# # plt.scatter(buy['index'], [df['Close'][index] for index in buy['index']], color='green')
# # plt.scatter(sell['index'], [df['Close'][index] for index in sell['index']], color='red')


# # Plot outputs


# # plt.xticks(())
# # plt.yticks(())


# plt.show()