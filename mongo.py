from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import sys

client = MongoClient('mongodb://localhost:27017')
db = client.local


def getStock(ticker):
    tickerCol = db[ticker]
    
    df = pd.DataFrame(list(tickerCol.find()))
    return parseDate(df)


def plotStocks(stocks):
    _, axs = plt.subplots(len(stocks), 1, sharex=True)
    plt.subplots_adjust(hspace=0.57)
    for i, ticker in enumerate(stocks):
        values = getStock(ticker)
        values.plot(x='minute', y='close', ax=axs[i], title=ticker)
    plt.show()


def getSeconds(time):
    split = time.split(':')

    hours = int(split[0])
    minutes = int(split[1])
    if hours < 9:
        hours += 12

    return pd.Timedelta(minutes=hours * 60 + minutes)

def parseDate(df):
    dateparse = lambda x: pd.datetime.strptime(x, "%Y%m%d")
    df['label'] = df['minute'].apply(getSeconds)
    df['date'] = df['date'].apply(dateparse)
    df['Date'] = df['date'] + df['label']
    return df

def saveColumnsToCSV(stocks):
    
    for ticker in stocks:
        df = getStock(ticker)

        df = df.set_index('Date')
        OHLC = df[['marketHigh', 'marketLow', 'marketClose', 'marketOpen', 'marketVolume']]
        OHLC = OHLC.rename(index=str, columns={'marketHigh': 'High', 'marketLow': "Low", 'marketOpen': 'Open', 'marketClose': 'Close', 'marketVolume': 'Volume'})
        
        mask = (OHLC.Volume >= 100) & (OHLC.Close > 0)
        print(mask.values)
        OHLC = OHLC[mask.values]
        OHLC.to_csv('./data/' + ticker + "_mongo.csv")


if len(sys.argv) > 1:
    saveColumnsToCSV(sys.argv[1:])
else:
    saveColumnsToCSV(['AAPL', 'AMD', "NVDA", "SPY", "AMZN"])

# plotStocks(['AAPL', 'AMD'])