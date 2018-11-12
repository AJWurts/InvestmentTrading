import numpy as np
import pandas as pd

# csv = pd.read_csv('SPY.csv')

# close = csv['Close']

def cumsum(raw, h):
    pos = 0
    neg = 0
    diff = raw.diff()
    points = []
    for i, p in enumerate(raw):
        if i == 0: continue
        pos = max(0, pos + (diff[i] / raw[i]))
        neg = min(0, neg + (diff[i] / raw[i]))
        if pos > h:
            points.append(i)
            pos = 0
        elif neg < -h:
            points.append(i)
            neg = 0

    return points


