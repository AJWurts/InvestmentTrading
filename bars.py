import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Bar:
    def __init__(self):
        self.high = None
        self.close = None
        self.low = None
        self.open = None
        self.lastTick = None
        self.date = None
        self.volume = 0

    def addTick(self, tick):
        if self.high is None or tick['High'] > self.high:
            self.high = tick['High']
        if self.low is None or tick['Low'] < self.low:
            self.low = tick['Low']
        if self.open is None:
            self.date = tick['Date']
            self.open = tick['Open']

        self.volume += tick['Volume']

        self.lastTick = tick

    def updateLastTick(self, threshold):
        if self.volume > threshold:
            self.volume = threshold
        self.close = self.lastTick['Close']

    def getRow(self):
        return [self.open, self.high, self.low, self.close]

    def __str__(self):
        return "D:{5}O:{0}L:{1}H:{2}C:{3}V:{4}".format(self.open, self.low, self.high, self.close, self.volume, self.date)

    def __repr__(self):
        return self.__str__()


def dollarBars(data, threshold):

    data = data.to_dict('records')
    bars = []
    total = 0
    temp_bar = Bar()
    for d in data:
        temp_bar.addTick(d)
        total += d['Volume'] * d['Close']
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append([temp_bar.date, temp_bar.close])
            temp_bar = Bar()
            total = total - threshold

    temp_bar.updateLastTick(threshold)
    bars.append([temp_bar.date, temp_bar.close])

    df = pd.DataFrame(bars, columns=['Date', "Close"])
    return df


def tickBars(data, threshold):
    bars = []
    total = 0
    temp_bar = Bar()
    for d in data:
        temp_bar.addTick(d)
        total += 1
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append(temp_bar)
            temp_bar = Bar()
            total = 0

    temp_bar.updateLastTick(threshold)
    bars.append(temp_bar)

    return bars


def volumeBars(data, threshold):

    bars = []
    total = 0
    temp_bar = Bar()
    for d in data:
        temp_bar.addTick(d)
        total += d['Volume']
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append(temp_bar)
            temp_bar = Bar()
            total = total - threshold

    temp_bar.updateLastTick(threshold)
    bars.append(temp_bar)

    return bars


def main():
    data = pd.read_csv('SPY.csv', parse_dates=[0])


    dbars = dollarBars(data, 2e11)
    close = dbars.price.copy()

    events = cumsum(dbars, 0.01)



    print(events)

#   index = [i for i in range(len(vBars))]
#   _open = [bar.open for bar in vBars]
#   _close = [bar.close for bar in vBars]
#   _high = [bar.high for bar in vBars]
#   _low = [bar.low for bar in vBars]

#   _, ax = plt.subplots()

#   candlestick_ohlc(ax, zip(index, _open, _high, _low, _close), width=0.6)

#   plt.show()

    # for i, k in enumerate(data_dict.keys()):
    #   print(data[i]['Close'])

    # # for d in data['Close']:
    #   print(d)

    # print(dBars)


if __name__ is "__main__":
    main()

