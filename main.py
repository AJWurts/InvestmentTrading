
from DBDataProcessor import processor
from machinelearning.randomforest import forestgenerator
from backtesting.backtester import start_backtest
import sys

print('Processing csv file')
bins = processor("./data/DE.csv")

print("Creating Machine Learning ALgorithm")
clf = forestgenerator(bins)



if len(sys.argv) > 1:
  print("Back Testing")
  start_backtest()

