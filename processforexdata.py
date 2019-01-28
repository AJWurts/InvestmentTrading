# from tqdm import tqdm
from os import  listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from datetime import datetime
from machinelearning.triplebars import mpPandasObj

millisecond_parse = lambda x: pd.Timedelta(milliseconds=int(x))
dateParseString = '%Y%m%d'

def loadFile(fileNames, molecule):
  final_df = None
  print(molecule)
  fileNames_ = fileNames[molecule]
  for name in fileNames_:
    
    date = datetime.strptime(name.split('_')[1][-8:], dateParseString)
    df = pd.read_csv(name, header=0, index_col=0, date_parser=millisecond_parse, parse_dates=[0],
                    names=['Date', 'Open', 'High', 'Low', 'Close', 'none', 
                          'ask_open', 'ask_high', 'ask_low', 'ask_close', 'none2'])
    

    df.index = df.index + date

    clipped = df[['Open', 'High', 'Low', 'Close']]

    if final_df is None:
      final_df = clipped
    else:
      final_df = pd.concat([final_df, clipped])
  
  return final_df
    
def reduxFx(df1, df2, **kargs):
  return pd.concat([df1, df2])

if __name__ == "__main__":
  base = './forex_data/'
  fileNames = np.array([join(base, f) for f in listdir(base) if isfile(join(base, f))])

  result = mpPandasObj(loadFile, ('molecule', list(range(len(fileNames)))), numThreads=4, mpBatches=5, fileNames=fileNames, redux=reduxFx)

  print(result)


  # _2015to2018 = df[df.index.year >= 2015]
  result.to_csv('./data/forex_all.csv')
  # final_df.to_csv("../data/forex_data.csv")

