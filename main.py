
import sys
from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import os
ticker = sys.argv[1]



for ticker in sys.argv[1:]:
    os.system('python3 DBDataProcessor.py ' + ticker)
    os.system('python3 machinelearning/randomforest.py ' + ticker)
    # os.system('python3 backtesting/backtester.py ' + ticker)





