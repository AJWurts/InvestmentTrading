from sampling import cumsum
from bars import dollarBars, Heikin_Ashi, tickBars, volumeBars
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Code straight from the book
def applyTripleBarrierLabeling(close, events, ptSl):
    """
    Labels every point up, down or neutral
    close: list of close prices
    events: 
        [t1: timestamp of vertical barrier ]
    """
    events_ = events
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



def getDailyVol(close,span0=50):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]   
    df0 = (pd.Series(close.index[df0 - 1], 
                   index=close.index[close.shape[0]-df0.shape[0]:]))   
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily rets
    except Exception as e:
        print('error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0
    

def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1


def addStartTime(bins, close, numDays=1):
    tMinus1 = close.index.searchsorted(bins.index - pd.Timedelta(days=numDays))
    tMinus1 = tMinus1[tMinus1 >= 0]
    tMinus1 = pd.Series(close.index[tMinus1], index=bins.index[:tMinus1.shape[0]])
    return tMinus1

def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side. Adjusts so the bin is in terms of the algorithm
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px =  close.reindex(px,method='bfill')
    #2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_:
        out['ret'] *= events_['side'] # meta-labeling
    out['bin'] = np.sign(out['ret'])

    if 'side' in events_:
        out.loc[out['ret'] <= 0,'bin'] = 0  # meta-labeling
    return out

def createTrainingData(bins, close):
    print(close)
    # Bins
    start = close.index.searchsorted(bins['start'].values)
    finish = close.index.searchsorted(bins.index)

    # start = close.index[start]
    # finish = close.index[finish]

    data = []

    for i in range(len(start)):
        s = start[i]
        f = finish[i]

        data.append(close.values[s:f+1])


    data = pd.Series(data, index=bins.index[:t1.shape[0]])

    bins['data'] = data

    return bins



if __name__ == "__main__":
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv("SPY.csv", parse_dates=[0], date_parser=dateparse)
    data = data.set_index('Date')
    bars, raw_bars = dollarBars(data, 1e11, returnBars=True)  
    # dollar bars 1e11
    bars = Heikin_Ashi(raw_bars)

    # print(bars)
    close = bars.Close.copy()
    dailyVol = getDailyVol(close)

    events = cumsum(bars, 0.008)
    
    t1 = addVerticalBarrier(events, data['Close'], numDays=3)
    trgt = dailyVol
    side_ = pd.Series(1.,index=trgt.index)

    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
    
    out = applyTripleBarrierLabeling(data['Close'], events, [1,1] )


    bins = getBins(out, data['Close'])


    tMinus1 = addStartTime(bins, data['Close'], numDays=7)

    bins['start'] = tMinus1

    bins = createTrainingData(bins, data['Close'])    

    bins.to_csv('ml_training_data.csv')
    # print(bins)

    # print(bins.bin.value_counts())
    # print(bins)


    # buying = bins[bins.bin == 1.0]
    # selling = bins[bins.bin == -1]

    # ax = plt.gca()
    
    # buyingX = [i for i in buying.index]
    # buyingY = [data['Close'][x] for x in buying.index]
    # buyingSize = [abs(ret) * 1000 for ret in buying['ret']]


    # plt.scatter(buyingX, buyingY, color='green', zorder=3, s=buyingSize)

    # sellingX = [i for i in selling.index]
    # sellingY = [data['Close'][x] for x in selling.index]
    # sellingSize = [abs(ret) * 1000 for ret in selling['ret']]

    # plt.scatter(sellingX, sellingY, color='red', zorder=2, s=sellingSize)


    # data.plot(y='Close', ax = ax, zorder=1)

    # plt.show()

    


