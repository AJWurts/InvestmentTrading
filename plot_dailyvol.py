import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def millisecond_parse(x): return pd.Timedelta(milliseconds=int(x))


dateParseString = '%d/%m/%Y'


def loadFile(name, molecule=None):
    final_df = None

    # date = datetime.strptime(name.split('_')[1][-8:], dateParseString)
    df = pd.read_csv(name, header=0,  delimiter=',')

    clipped = df[['Open', 'High', 'Low', 'Close']]

    return clipped


df = loadFile('./data/VOO.csv')
delta = df['High'].values - df['Low'].values

#bins every 0.05

points = 28
bins = []
for i in range(points * 20): # 540 / 20 = 27 points 
    bins.append(-(points / 2.0) + i * (1 / (points * 20) * points))

bins = np.array(bins)

# for val in delta:


# print(bins)

print("STDEV: ", np.std(delta))
print("Mean: ", np.mean(delta))
print("Median: ", np.median(delta))
histo, _ = np.histogram(delta, bins='auto')


plt.hist(delta, bins='auto')
# plt.plot(histo)
plt.show()
