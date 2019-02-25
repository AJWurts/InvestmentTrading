from machinelearning.sampling import cumsum
from machinelearning.bars import dollarBars, Heikin_Ashi, tickBars, volumeBars, customBars
from machinelearning.fracdiff import fracDiff
from machinelearning.triplebars import applyTripleBarrierLabeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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


def createTrainingData(bins, data, length=120, fd=False):
    # print(close)
    # Bins
    if fd:
        diff = fracDiff(data, d=0.75)
        data = diff[['Close', 'High', 'Open', 'Low']]
    else:
        data = data[['Close', 'High', 'Open', 'Low']]
    start = data['Close'].index.searchsorted(bins['start'].values)
    finish = data['Close'].index.searchsorted(bins.index)
    arrays = []
    initRet = []


    for i in range(len(start)):
        s = start[i]
        f = finish[i]
        sVal = data['Close'].values[s]
        fVal = data['Close'].values[f]
        initRet.append(fVal - sVal)
        arrays.append(np.array([i for i in np.array(data.values[s:f+1]).flatten()]))


    training_arrays = pd.Series(arrays, index=bins.index)
    training_arrays = training_arrays[training_arrays.map(len) > 8]
    initret = pd.Series(initRet, index=bins.index)
    bins['data'] = training_arrays
    bins['initret'] = initret
    # bins = bins[mask]
    return bins

def calcHyperParams(data, percentile=75, numDays=2, func=lambda x: x['Volume'] * x['Close']):
    """
    Calculates threshold for when cumulative summation activates and calculates the number required so that the bar creator roughly creates bars equal to 2 days each.
    percentile: the percentile to select for the threshold
    numDays: the approximate number of days to include in each bar based on the average of the volume * close
    func: Should be the same function used in the custumBars method
    """
    # Goal: Calculate 75% percentile of daily price movemet assuming its normal
    diff = (data['High'] - data['Low'])
    thresh = np.percentile(diff, percentile)

    # Calculate vol*price for data so that the bars last around 2 days.
    vol_price_data = func(data)
    # print(vol_price_data)
    vol_price_avg = np.percentile(vol_price_data, 50) * numDays


    return vol_price_avg, thresh / 100
    

def createTestData(data, filename, length=45):
    # Creates test data set and saves it under filename_test.csv
    test = data.iloc[-length:]
    lastSlash = filename.rfind('/')
    period = filename.rfind('.')

    test.to_csv('./data/' + filename[lastSlash + 1: period] + '_test.csv')

    return data.iloc[:-length]

def processor(filename):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    dparse2 = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
    dateparse3 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    print("Loading CSV")
    data = pd.read_csv(filename, parse_dates=['Date'],  date_parser=dateparse3)
    vol_price, thresh = calcHyperParams(data, numDays=1, percentile=80)
    print("THRESHOLD: ", thresh)
    data = createTestData(data, filename, length=80)
   
    print("Creating Bars")
    bars, raw_bars = customBars(data, vol_price, lambda x: x['Volume'] * x['Close'], returnBars=True)  
    data = data.set_index('Date')
    # dollar bars 1e11 for days
    # dollar bar for minutes = 3.6e7

    print("Heikin Ashi Bars")
    bars = Heikin_Ashi(raw_bars)

    print("Cumulative Summation Event Selector")
    events = cumsum(bars, thresh)

    print("Vertical Bars")
    t1 = addVerticalBarrier(events, data['Close'], numDays=3)
    trgt = pd.Series(0.0001, index=t1.index)
    side_ = pd.Series(1.,index=t1.index)
    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
    
    out = applyTripleBarrierLabeling(data['Close'], events, [1,1])
    # print("Bins after triple barrier: ", out)
    out = out.sort_index()
    print("Bins")
    bins = getBins(out, data['Close'])

    bins = bins[bins.bin != 0]

    print("Add Start Time")
    tMinusl = addStartTime(bins, data['Close'], numDays=10)

    bins['start'] = tMinusl

    print("Creating Training Data")
    bins = createTrainingData(bins, data, length=8, fd=True)    
    
    print("Saving")
    lastSlash = filename.rfind('/')
    period = filename.rfind('.')
    bins.to_csv('./data/training_' + filename[lastSlash + 1: period] + '.csv')

    return bins

if __name__ == "__main__":
    # altProcess('./data/UNH.csv')
    if len(sys.argv) > 1:
        processor('./data/' + sys.argv[1] + '.csv')
    else:
        processor("./data/UNH.csv")




