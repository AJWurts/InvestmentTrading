from tqdm import tqdm
from os import  listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from datetime import datetime
base = '../forex_data/'
fileNames = [join(base, f) for f in tqdm(listdir(base)) if isfile(join(base, f))]

dateParseString = '%Y%m%d'
millisecond_parse = lambda x: pd.Timedelta(milliseconds=int(x))
final_df = None

date_parser = []
count = 0
for name in tqdm(fileNames):
  date = datetime.strptime(name.split('_')[1][-8:], dateParseString)
  if date.year >= 2011:
    df = pd.read_csv(name, header=0, index_col=0, date_parser=millisecond_parse, parse_dates=[0],
                    names=['Date', 'Open', 'High', 'Low', 'Close', 'none', 
                          'ask_open', 'ask_high', 'ask_low', 'ask_close', 'none2'])
    
  
    df.index = df.index + date

    clipped = df[['Open', 'High', 'Low', 'Close']]

    if final_df is None:
      final_df = clipped
    else:
      final_df = pd.concat([final_df, clipped])
  elif date.year == 2012:
    break

# _2015to2018 = df[df.index.year >= 2015]
final_df.to_csv('../data/forex_million.csv')
# final_df.to_csv("../data/forex_data.csv")

