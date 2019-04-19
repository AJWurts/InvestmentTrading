import numpy as np
import pandas as pd

# csv = pd.read_csv('SPY.csv')

# close = csv['Close']

def cumsum(data, p, returnI=False):
    """
    takes in raw pricing and samples based on the price movement.
    p is the percent threshold to activate
    """
    pos = 0
    neg = 0
    raw = data['Close']
    diff = raw.diff()
    points = []
    for i, _ in enumerate(raw):
        if i == 0: continue
        pos = max(0, pos + (diff[i] / raw[i]))
        neg = min(0, neg + (diff[i] / raw[i]))
        if pos > p:
            points.append(data.index[i])
            pos = 0
        elif neg < -p:
            points.append(data.index[i])
            neg = 0
    index = pd.DatetimeIndex(points)
    return index




