import numpy as np
import pandas as pd

# csv = pd.read_csv('SPY.csv')

# close = csv['Close']

def cumsum(raw, p):
    """
    takes in raw pricing and samples based on the price movement.
    p is the percent threshold to activate
    """
    pos = 0
    neg = 0
    diff = raw.diff()
    points = []
    for i, p in enumerate(raw):
        if i == 0: continue
        pos = max(0, pos + (diff[i] / raw[i]))
        neg = min(0, neg + (diff[i] / raw[i]))
        if pos > p:
            points.append(i)
            pos = 0
        elif neg < -p:
            points.append(i)
            neg = 0

    return points



def fractionalDifferentiation(raw, t, d):

    summation = 0
    # raw = [t, ..., t-max]
    raw = raw[t:]

    for k in range(len(raw)):
        product = -raw[k]
        for i in range(len(data) - 1)):
            product *= ((d - i) / (k - i))

        summation += product  

    return summmation      


