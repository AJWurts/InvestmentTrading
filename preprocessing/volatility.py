import numpy
import pandas

csv = pandas.read_csv("../data/SPY.csv")


def get_range(csv):
    min_val = min(csv)
    max_val = max(csv)
    return min_val, max_val


def find_volatility():
    vol = []
    for i in range(len(csv)):
    
        high = csv.iloc[i, 2]
        low = csv.iloc[i, 3]
        close = csv.iloc[i, 4]

        volatility = (high - low) / close
        vol.append(volatility)
    return vol


print(find_volatility())
