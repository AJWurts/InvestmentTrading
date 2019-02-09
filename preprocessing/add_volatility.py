import pandas as pd
import numpy as np


csv = pd.read_csv("../data/SPY.csv")
csv['Volatility'] = (csv['High'] - csv['Low']) / csv['Close']
csv.to_csv('../data/SPY_v.csv')