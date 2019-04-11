import matplotlib.pyplot as plt
from random import randint
import numpy as np
from joblib import load
from machinelearning.fracdiff import getWeights

mlalgoHasRun = False
clf = None
weights = None
flag = False
previous_data = []
diff_data = []
threshold = 0.1
input_size = 0

ma_10 = []
ma_20 = []

pos = 0
neg = 0



def MA(n=10):
    return sum(previous_data[-n:]) / n

def wurtsAlgorithm(p, cash, stockOwned, ticker=None):
    previous_data.append(p)

    if len(previous_data) >= 10:
        ma_10.append(MA(10))

    if len(previous_data) >= 20:
        ma_20.append(MA(20))

    if len(ma_20) > 2:
        # cross up
        if ma_10[-2] < ma_20[-2] and ma_10[-1] > ma_20[-1]:
            return 'buy', 1000

        # cross down
        if ma_10[-2] > ma_20[-2] and ma_10[-1] < ma_20[-1]:
            return 'sell', 1000


    return 'do_nothing', 0


def alwaysBuy(p, cash, stockOwned, ticker=None):
    if (cash > p):
        return 'buy', 1000000
    else:
        return 'do_nothing', 0


def keepAt50k(p, cash, stockOwned, ticker=None):
    if cash > 50000:
        return 'buy', 100
    elif cash < 50000:
        return 'sell', 100
    else:
        return 'do_nothing', 0
    

def randomBuySell(p, cash, stockOwned, ticker=None):
    choice = randint(0, 2)

    if choice == 0:
        return 'buy', randint(0, 500)
    elif choice == 1:
        return 'sell', randint(0, 500)
    else:
        return 'do nothing', 0


def mlalgo(p, cash, stockOwned, ticker):
    global pos, neg, mlalgoHasRun, clf, previous_data, weights, diff_data, flag, threshold, input_size
    previous_data.append(p)
    if not mlalgoHasRun:
        mlalgoHasRun = True
        weights = getWeights(0.75, threshold=0.01)
        with open(ticker + 'thresh.txt', 'r') as threshFile:
            threshold = float(threshFile.read())
        with open(ticker + 'mlsize.txt', 'r') as mlsizefile:
            input_size = int(mlsizefile.read())
        clf = load('./machinelearning/saved_classifiers/randomforest_' + ticker + '.joblib')

    if len(previous_data) > len(weights):
        val = np.dot(weights.T, previous_data[-len(weights):])
        if not np.isnan(val[0]):
            diff_data.append(val[0])

    flag = False
    # threshold = 0.0395 ## SET ME FROM THE VALUE YOU GOT IN DBDATAPROCESSOR called THRESHOLD
    if len(previous_data) > 1:
        pos = max(0, pos + ((p - previous_data[-2]) / previous_data[-2]))
        neg = min(0, neg + ((p - previous_data[-2]) / previous_data[-2]))
        if pos > threshold:
            flag = True
            pos = 0
        elif neg < -threshold:
            flag = True
            neg = 0

    # ml_input_size = 16
    if len(diff_data) >= input_size and flag:
        result = clf.predict([diff_data[-input_size:]])[0]
        print(result)
        if result == -1 and cash > p:
            return 'buy', int(10000 / p) 
        # else:
        #     return 'sell', 500
        flag = False
    return 'Do Nothing', 10

def reset():
    global previous_data, ma_10, ma_20
    previous_data = []
    ma_10 = []
    ma_20 = []
