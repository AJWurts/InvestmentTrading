from sklearn.model_selection import KFold as _BaseKFold
from joblib import load
import pandas as pd
import numpy as np


def getTrainTimes(t1, testTimes):
  """
  Given testTimes, find the times of the training observations
  -t1.index: Time when the observation starts.
  -t1.value: Time when the observation ended.
  -testTimes: Times of testing observations.
  """
  trn = t1.copy(deep=True)
  for i, j in testTimes.iteritems():
    df0 = trn[(i <= trn.index) & (trn.index <= j)].index # Train starts within test
    df1 = trn[(i <= trn) & (trn <= j)].index # Train ends within test
    df2 = trn[(trn.index <= i) & (trn >= j)] # Train envelopes test
    trn = trn.drop(df0.union(df1).union(df2))
  
  return trn


def getEmbargoTimes(times, pctEmbargo):
  # Get embargo time for each bar
  step = int(times.shape[0] * pctEmbargo)
  if step == 0:
    mbrg = pd.Series(times, index=times)
  else:
    mbrg = pd.Series(times[step:], index=times[:-step])
    mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))

  return mbrg


class PurgedKFold(_BaseKFold):
  """
  Extend KFold class to work with labels that span intervals
  The train is purged of observations oerlapping test-label intervals
  Test set is asumed contiguous (shuffle = False), w/o training samples in between
  """
  def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
    if not isinstance(t1, pd.Series):
        raise ValueError("Label Through Dates must be a pd.Series")
      
    super(PurgedKFold,self).__init__(n_splits, shuffle=False, random_state=None)
    self.t1 = t1
    self.pctEmbargo = pctEmbargo
  
  def split(self, X, y=None, groups=None):
    if (X.index == self.t1.index).sum() != len(self.t1):
      raise ValueError('X and ThruDateValues must have the same index')
    indices = np.arange(X.shape[0])
    mbrg = int(X.shape[0] * self.pctEmbargo)
    test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
    for i, j in test_starts:
      t0 = self.t1.index[i] # Start of test set
      test_indices = indices[i:j]
      maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
      train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
      if maxT1Idx < X.shape[0]: # Right train (with embargo)
        train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
      yield train_indices, test_indices


def cvScore(clf, X, y, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
  if scoring not in ['neg_log_loss', 'accuracy']:
    raise Exception('wrong scoring method.')
  from sklearn.metrics import log_loss, accuracy_score
  if cvGen is None:
      cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo) # Purged
  score = []
  for train, test in cvGen.split(X=X):
    fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train])#, sample_weight=sample_weight.iloc[train].values)
    if scoring == 'neg_log_loss':
      prob = fit.predict_proba(X.iloc[test, :])
      score_ = -log_loss(y.iloc[test], prob, labels=clf.classes_)#, sample_weight = sample_weight.iloc[test].values, )
    else:
      pred = fit.predict(X.iloc[test, :])
      score_ = accuracy_score(y.iloc[test], pred)#, sample_weight=sample_weight.iloc[test].values)
    score.append(score_)
  return np.array(score)


if __name__ == "__main__":
  clf = load('../machinelearning/randomforest_forex.joblib')


  df = pd.read_csv('../data/ml_training_data_2012-2018_2.csv')
  X = []
  for i in range(len(df)):
    c = df['data'][i]
    for char in '[]\n\r':
      c = c.replace(char, '')
    
    result = np.array(c.split(' '), dtype=str)
    result = result[result != '']
    result = list(map(lambda x: float(x), result))

    X.append(result)

  min_length = min([len(d) for d in X])
  X = [x[:min_length] for x in X]


  split = 0.9
  train_X = X
  train_y = df['bin'].values


  score = cvScore(clf, train_X, train_y, cv=3, pctEmbargo=1, t1=df.index)
  print(score)
