
from DBDataProcessor import processor
from randomforest import forestgenerator
from backtester import start_backtest
import sys

print('Processing csv file')
bins = processor("./data/AAPL_mongo.csv")

clf = forestgenerator(bins)

if len(sys.argv) > 1:
  start_backtest()

