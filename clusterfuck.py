from sampling import cumsum
from bars import dollarBars
import numpy as np
import pandas as pd

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


def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1


if __name__ == "__main__":
    data = pd.read_csv("SPY.csv")

    dbars = dollarBars(data, 2e11)

    events = cumsum(dbars, 0.01)

    t1 = addVerticalBarrier(events, data['Close'], numDays=10)

    print(t1)
    print(events)
    # tEvents = cumsum(data['Close'], 0.01)

