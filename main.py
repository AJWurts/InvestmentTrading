
import sys
from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import os
ticker = sys.argv[1]

# fileName = './data/' + ticker + '.csv'

# bins = processor(fileName)
# forestgenerator(ticker)
# start_backtest(ticker)

os.system('python3 DBDataProcessor.py ' + ticker)
os.system('python3 machinelearning/randomforest.py ' + ticker)
os.system('python3 backtesting/backtester.py ' + ticker)





