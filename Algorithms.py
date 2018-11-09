import matplotlib.pyplot as plt
from random import randint


previous_data = []

ma_10 = []
ma_20 = []

def MA(n=10):
    return sum(previous_data[-n:]) / n

def wurtsAlgorithm(p, cash, stockOwned):
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


def alwaysBuy(p, cash, stockOwned):
    return 'buy', 1000000


def keepAt50k(p, cash, stockOwned):
    if cash > 50000:
        return 'buy', 100
    elif cash < 50000:
        return 'sell', 100
    else:
        return 'do_nothing', 0
    

def randomBuySell(p, cash, stockOwned):
    choice = randint(0, 2)

    if choice == 0:
        return 'buy', randint(0, 500)
    elif choice == 1:
        return 'sell', randint(0, 500)
    else:
        return 'do nothing', 0