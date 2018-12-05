from tqdm import tqdm
from os import  listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
base = '../forex_data/'
fileNames = [join(base, f) for f in tqdm(listdir(base)) if isfile(join(base, f))]

result_data = []

date_parser = 
for name in fileNames:
  
  df = pd.read_csv(name, header=0, names=['time', 'bid_open', 'bid_high', 'bid_low', 'bid_close', ask])
