
import sys
from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DBDataProcessor import processor
from machinelearning.randomforest import forestgenerator
from backtesting.backtester import start_backtest
import os
ticker = sys.argv[1]

# fileName = './data/' + ticker + '.csv'

# bins = processor(fileName)
# forestgenerator(ticker)
# start_backtest(ticker)

os.system('python DBDataProcessor.py ' + ticker)
os.system('python machinelearning/randomforest.py ' + ticker)
os.system('python backtesting/backtester.py ' + ticker)



