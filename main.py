
from DBDataProcessor import processor
from machinelearning.randomforest import forestgenerator
from backtesting.backtester import start_backtest
import sys

print('Processing csv file')
bins = processor("./data/AAPL_mongo.csv")

clf = forestgenerator(bins)

if len(sys.argv) > 1:
  start_backtest()

