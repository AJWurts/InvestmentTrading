from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

client = MongoClient('mongodb://localhost:27017')
db = client.local

def getStock(ticker):
  tickerCol = db[ticker]
  return pd.DataFrame(list(tickerCol.find()))
  
plt.subplots_adjust(hspace=0.57)
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)

aapl = getStock("AAPL")
aapl.plot(x='minute', y='close', ax=ax1, title='AAPL')


amd = getStock('NVDA').dropna()
amd.plot(x='minute', y='close', ax=ax2, title='NVDA')

bac = getStock('BAC').dropna()
bac.plot('minute', 'close', ax=plt.subplot(313), title='BAC')


plt.show()