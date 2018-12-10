import pandas as pd
import numpy as np
import datetime as dt
import multiprocessing as mp
import time
import sys
from copy import copy

def _pickle_method(method):
  func_name = method.im_func.__name__
  obj = method.im_self
  cls = method.im_class
  return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
  for cls in cls.mro():
    try:
      func = cls.__dict__[func_name]
    except KeyError:
      pass
    else:
      break
  return func.__get__(obj, cls)

import copyreg, types, multiprocessing as mp
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def applyPtSlOnT1(close, events, ptSl, molecule):
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
            sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # Path Prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path prices
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() # Earlist stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # Earlist profit taking

    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads=2, t1=False):
  #1) Get Target
  trgt = trgt.loc[tEvents]
  trgt = trgt[trgt > minRet]

  #2) Get t1 (max holding period)
  if t1 is False: 
    t1 = pd.Series(pd.NaT, index=tEvents)
  
  #3) form events object, apply stop loss on t1
  side_ = pd.Series(1., index=trgt.index)
  events = pd.concat({'t1':t1, 'trgt':trgt, 'side':side_}, axis=1).dropna(subset=['trgt'])

  df0 = mpPandasObj(applyPtSlOnT1, ('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=[ptSl, ptSl])
  events['t1'] = df0.dropna(how='all').min(axis=1)
  events = events.drop('side', axis=1)
  return events


def reportProgress(jobNum, numJobs, time0, task):
  # report progress as asynch jobs are completed
  msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
  msg.append(msg[1] * (1 / msg[0] -1 ))

  timeStamp = str(dt.datetime.fromtimestamp(time.time()))
  msg = timeStamp + ' ' + str(round(msg[0]*100,2)) + '% ' + task + ' done after ' + \
        str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
  
  if jobNum < numJobs:
    sys.stderr.write(msg + '\r')
  else:
    sys.stderr.write(msg + '\n')
  return


def linParts(numAtoms, numThreads):
  parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
  parts = np.ceil(parts).astype(int)
  return parts


def processJobs_(jobs):
  # run jobs sequentially, for debugging
  out = []
  for job in jobs:
    out_ = expandCall(job)
    out.append(out_)
  
  return out


def processJobs(jobs, task=None, numThreads=2, redux=None, reduxArgs={}, reduxInPlace=False):
  # Run in parallel
  # jobs must contain a 'func' callback, for expandCall
  if task is None:
    task = jobs[0]['func'].__name__

  pool = mp.Pool(processes=numThreads)
  imap = pool.imap_unordered(expandCall, jobs)
  out = None
  time0 = time.time()

  # Process asynchronous output, report progress
  for i, out_ in enumerate(imap, 1):
    if out is None:
        if redux is None:
          out = [out_]
          redux = list.append
          reduxInPlace = True
        else:
          put = copy(out_)
    else:
      if reduxInPlace:
        redux(out, out_, **reduxArgs)
      else:
        out = redux(out, out_, **reduxArgs)
    reportProgress(i, len(jobs), time0, task)

    # out.append(out_)
    # reportProgress(i, len(jobs), time0, task)
  
  pool.close();pool.join()
  return out


def expandCall(kargs):
  func = kargs['func']
  del kargs['func']
  out = func(**kargs)
  return out


def mpPandasObj(func, pdObj, numThreads=2, mpBatches=1, linMols=True, redux=pd.DataFrame.append, reduxArgs={}, reduxInPlace=False, **kargs):
  """
  Parallelize jobs, return a DataFrame or Series
  + func: function to be parallelized. Returns a DataFrame
  + pdObj[0]: Name of argument used to pass the molecule
  + pdObj[1]: List of atoms that will be grouped into molecules
  + kargs: ano other argument needed by func
  Example: df1 = mpPandasObj(func, ('molecule', df0.index), 24, **kargs)
  """
  parts = linParts(len(pdObj[1]), numThreads * mpBatches)

  jobs = []
  for i in range(1, len(parts)):
    job = {pdObj[0]: pdObj[1][parts[i-1]:parts[i]], 'func': func}
    job.update(kargs)
    jobs.append(job)

  # if numThreads == 1:
  #   out = processJobs_(jobs)
  # else:
  #   out = processJobs(jobs, numThreads=numThreads)

  
  out = processJobs(jobs, redux=redux, reduxArgs=reduxArgs,
                    reduxInPlace=reduxInPlace, numThreads=numThreads)
  

  return out

def mpFunc(array, molecule):
  print(type(array))
  print(molecule)
  array_ = array[molecule]

  out = array_.copy()

  out = array_ + 10

  return out


def applyTripleBarrierLabeling(close, events, ptSl):
    """
    Labels every point up, down or neutral
    close: list of close prices
    events: 
        [t1: timestamp of vertical barrier ]
    """
    result = mpPandasObj(applyPtSlOnT1, ('molecule', events.index), redux=pd.DataFrame.append, close=close, ptSl=ptSl, events=events)
    return result

if __name__ == "__main__":

  
  a = np.arange(1000)#.reshape(100, 10)

  df = pd.DataFrame(a)

  result = mpPandasObj(mpFunc, ('molecule', a), mpBatches=20, array=a, redux=np.concatenate)

  print(result)

