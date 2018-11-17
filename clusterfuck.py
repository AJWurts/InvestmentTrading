from sampling import cumsum
from bars import dollarBars
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



def getDailyVol(close,span0=100):
    # daily vol reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]   
    df0=(pd.Series(close.index[df0-1], 
                   index=close.index[close.shape[0]-df0.shape[0]:]))   
    try:
        df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print('error: {e}\nplease confirm no duplicate indices')
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0
    
# # create target series
# ptsl = [1,1]
# target=dailyVol
# # select minRet
# minRet = 0.01

# # Run in single-threaded mode on Windows
# import platform
# if platform.system() == "Windows":
#     cpus = 1
# else:
#     cpus = cpu_count() - 1
    
# events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
# def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
#     #1) get target
#     trgt=trgt.loc[tEvents]
#     trgt=trgt[trgt>minRet] # minRet
#     #2) get t1 (max holding period)
#     if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)
#     #3) form events object, apply stop loss on t1
#     if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
#     else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
#     events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
#             .dropna(subset=['trgt']))
#     df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
#                     numThreads=numThreads,close=close,events=events,
#                     ptSl=ptSl_)
#     events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
#     if side is None:events=events.drop('side',axis=1)
#     return events

def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1

def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

if __name__ == "__main__":
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv("SPY.csv", parse_dates=[0], date_parser=dateparse)
    data = data.set_index('Date')
    print(data)
    dbars = dollarBars(data, 1e11)

    # print(dbars)
    close = dbars.Close.copy()
    dailyVol = getDailyVol(close)

    events = cumsum(dbars, 0.01)
    
    t1 = addVerticalBarrier(events, data['Close'], numDays=10)
    trgt = dailyVol
    side_ = pd.Series(1.,index=trgt.index)

    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
    
    out = applyTripleBarrierLabeling(data['Close'], events, [1,1] )

    bins = getBins(out, data['Close'])


    buying = bins[bins.bin == 1.0]
    selling = bins[bins.bin == -1]

    ax = plt.gca()
    
    buyingX = [i for i in buying.index]
    buyingY = [data['Close'][x] for x in buying.index]
    buyingSize = [abs(ret) * 1000 for ret in buying['ret']]


    plt.scatter(buyingX, buyingY, color='green', zorder=3, s=buyingSize)

    sellingX = [i for i in selling.index]
    sellingY = [data['Close'][x] for x in selling.index]
    sellingSize = [abs(ret) * 1000 for ret in selling['ret']]

    plt.scatter(sellingX, sellingY, color='red', zorder=2, s=sellingSize)


    data.plot(y='Close', ax = ax, zorder=1)

    plt.show()

    


