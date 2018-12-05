from machinelearning.sampling import cumsum
from machinelearning.bars import dollarBars, Heikin_Ashi, tickBars, volumeBars, customBars
from machinelearning.fracdiff import fracDiff
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
            sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # Path Prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path prices
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() # Earlist stop loss
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
    

def addVerticalBarrier(tEvents, close, numDays=1, numMinutes=None):
    if numMinutes is not None:
        offset = pd.Timedelta(minutes=numMinutes)
    else:
        offset = pd.Timedelta(days=numDays)
    t1 = close.index.searchsorted(tEvents + offset)
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1


def addStartTime(bins, close, numDays=1, numMinutes=None):
    if numMinutes is not None:
        offset = pd.Timedelta(minutes=numMinutes)
    else:
        offset = pd.Timedelta(days=numDays)
    tMinusl = close.index.searchsorted(bins.index - offset)
    # tMinusl = tMinusl[tMinusl >= 0]
    tMinusl = pd.Series(close.index[tMinusl], index=bins.index[:tMinusl.shape[0]])
    return tMinusl


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
    # print(events)
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    #2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

   
    if 'side' in events_:
        out['ret'] *= events_['side'] # meta-labeling
    

    if 'side' in events_:
        out.loc[out['ret'] <= 0,'bin'] = 0  # meta-labeling

    # Labels 1, and -1
    out['bin'] = np.sign(out['ret'])
     # out['ret'][abs(out.ret) < 0.008] = 0
    out['bin'][events_.sl.notnull() & ((events_.pt.notnull() & (events_.sl < events_.pt)) | events_.pt.isnull())] = -1
    out['bin'][events_.pt.notnull() & ((events_.sl.notnull() & (events_.pt < events_.sl)) | events_.sl.isnull())] = 1
    out['bin'][events_.pt.isnull() & events_.sl.isnull()] = 0

    return out


def createTrainingData(bins, data, length=50):
    # print(close)
    # Bins
    # diff = fracDiff(data)
    # close = diff['Close']
    close = data['Close']
    start = close.index.searchsorted(bins['start'].values)
    finish = close.index.searchsorted(bins.index)

    # start = close.index[start]
    # finish = close.index[finish]

    arrays = []

    for i in range(len(start)):
        s = start[i]
        f = finish[i]
        arrays.append(close.values[s:f+1])

    training_arrays = pd.Series(arrays, index=bins.index)
    mask = training_arrays.apply(lambda x: len(x) >= length).values
    
    bins['data'] = training_arrays
    bins = bins[mask]
    return bins


    
def processor(filename):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    data = pd.read_csv(filename, parse_dates=[0], date_parser=dateparse)
   
    bars, raw_bars = customBars(data, 1e7, lambda x: x['Volume'] * x['Close'], returnBars=True)  
    data = data.set_index('Date')
    # dollar bars 1e11 for days
    # dollar bar for minutes = 3.6e7
    bars = Heikin_Ashi(raw_bars)
    


    events = cumsum(bars, 0.0001)
    print(data.index)
    
    t1 = addVerticalBarrier(events, data['Close'], numMinutes=120)
    trgt = pd.Series(0.003, index=t1.index)#dailyVol[t1.index]
    side_ = pd.Series(1.,index=t1.index)

    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
    
    out = applyTripleBarrierLabeling(data['Close'], events, [1,1])


    bins = getBins(out, data['Close'])

    bins = bins[bins.bin != 0]


    tMinusl = addStartTime(bins, data['Close'], numMinutes=50)

    bins['start'] = tMinusl

    bins = createTrainingData(bins, data)    
    
    bins.to_csv('ml_training_data.csv')

    return bins

if __name__ == "__main__":
    main()




