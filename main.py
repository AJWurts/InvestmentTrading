
import sys
from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import os
ticker = sys.argv[1]
## Run me followed by stock tickers of Yahoo data you have downloaded and placed in the data folder


ticker_string = ""
for ticker in sys.argv[1:]:
    ticker_string += ticker + " "
    os.system('python3 DBDataProcessor.py ' + ticker)
    os.system('python3 machinelearning/randomforest.py ' + ticker)


os.system('python3 backtesting/backtester.py ' + ticker_string)





