import numpy as np
import pandas as pd

# csv = pd.read_csv('SPY.csv')

# close = csv['Close']

def cumsum(data, p):
    """
    takes in raw pricing and samples based on the price movement.
    p is the percent threshold to activate
    """
    pos = 0
    neg = 0
    print(data)
    raw = data['Close']
    diff = raw.diff()
    points = []
    for i, _ in enumerate(raw):
        if i == 0: continue
        pos = max(0, pos + (diff[i] / raw[i]))
        neg = min(0, neg + (diff[i] / raw[i]))
        if pos > p:
            points.append(data['Date'][i])
            pos = 0
        elif neg < -p:
            points.append(data['Date'][i])
            neg = 0

    return pd.DataFrame(points)



# def fractionalDifferentiation(raw, t, d):

#     summation = 0
#     # raw = [t, ..., t-max]
#     raw = raw[t:]

#     for k in range(len(raw)):
#         product = -raw[k]
#         for i in range(len(data) - 1)):
#             product *= ((d - i) / (k - i))

#         summation += product  

#     return summmation      


# Code straight from the book
def applyTripleBarrierLabeling(close, events, ptSl, molecule):
    """
    Labels every point up, down or neutral
    close: list of close prices
    events: 
        [t1: timestamp of vertical barrier ]
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)
    
    if ptSl[1] > 0:
            sl = ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # Path Prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path prices
        out.loc[loc, 's1'] = df0[df0 < sl[loc]].index.min() # Earlist stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # Earlist profit taking

    return out



# if __name__ == "__main__":
#     data = pd.read_csv("SPY.csv")

#     print(cumsum(data['Close'], 0.01))

