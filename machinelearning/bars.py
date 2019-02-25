import pandas as pd
import matplotlib.pyplot as plt
# from mpl_finance import candlestick_ohlc
from tqdm import tqdm


class Bar:
    def __init__(self):
        self.high = None
        self.close = None
        self.low = None
        self.open = None
        self.lastTick = None
        self.date = None
        self.volume = 0

    def addTick(self, date, tick):
        if self.high is None or float(tick['High']) > self.high:
            self.high = float(tick['High'])
        if self.low is None or float(tick['Low']) < self.low:
            self.low = float(tick['Low'])
        if self.open is None:
            self.date = date
            self.open = float(tick['Open'])

        if 'Volume' in tick:
            self.volume += float(tick['Volume'])

        self.lastTick = tick

    def updateLastTick(self, threshold):
        if self.volume > threshold:
            self.volume = threshold
        self.close = float(self.lastTick['Close'])

    def getRow(self):
        return [self.open, self.high, self.low, self.close]

    def __str__(self):
        return "D:{5}O:{0}L:{1}H:{2}C:{3}V:{4}".format(self.open, self.low, self.high, self.close, self.volume, self.date)

    def __repr__(self):
        return self.__str__()


def Heikin_Ashi(bars, returnBars=False):
    result = [bars[0]]
    df_result = []
    for i in range(1, len(bars)):
        newBar = Bar()
        newBar.date = bars[i].date
        cur = bars[i]
        prev = result[i-1]
        newBar.close = (cur.open + cur.high + cur.low + cur.close) / 4
        newBar.open = (prev.close + prev.open) / 2
        newBar.high = max([cur.high, newBar.open, newBar.close])
        newBar.low = min([cur.low, newBar.open, newBar.close])
        result.append(newBar)
        df_result.append([newBar.date, newBar.close,
                           newBar.high, newBar.low, newBar.open])

    if returnBars:
        return result
    df = pd.DataFrame(df_result, columns=['Date', "Close", "High", "Low", "Open"])
    return df.set_index('Date')


def customBars(data, threshold, func, returnBars=False):
    dict_data = data.to_dict('records')
    bars = []
    raw_bars = []
    total = 0
    temp_bar = Bar()
    for i, d in tqdm(enumerate(dict_data), total=len(dict_data)):
        temp_bar.addTick(data.Date[i], d)
        total += func(d)
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append([temp_bar.date, temp_bar.close,
                         temp_bar.high, temp_bar.low, temp_bar.open])
            raw_bars.append(temp_bar)
            temp_bar = Bar()
            total = total - threshold

    if temp_bar.high is not None:  # High chosen arbitrarily
        temp_bar.updateLastTick(threshold)
        raw_bars.append(temp_bar)
        bars.append([temp_bar.date, temp_bar.close,
                     temp_bar.high, temp_bar.low, temp_bar.open])

    df = pd.DataFrame(bars, columns=['Date', "Close", "High", "Low", "Open"])
    if returnBars:
        return df, raw_bars
    return df


def dollarBars(data, threshold, returnBars=False):

    dict_data = data.to_dict('records')
    bars = []
    raw_bars = []
    total = 0
    temp_bar = Bar()
    for i, d in enumerate(dict_data):
        temp_bar.addTick(data.Date[i], d)
        total += d['Volume'] * d['Close']
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append([temp_bar.date, temp_bar.close])
            raw_bars.append(temp_bar)
            temp_bar = Bar()
            total = total - threshold

    if temp_bar.high is not None:  # High chosen arbitrarily
        temp_bar.updateLastTick(threshold)
        raw_bars.append(temp_bar)
        bars.append([temp_bar.date, temp_bar.close])

    df = pd.DataFrame(bars, columns=['Date', "Close"])
    if returnBars:
        return df, raw_bars
    return df


def tickBars(data, threshold, returnBars=False):
    dict_data = data.to_dict('records')
    bars = []
    raw_bars = []
    total = 0
    temp_bar = Bar()
    for i, d in enumerate(dict_data):
        temp_bar.addTick(data.Date[i], d)
        total += 1
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append([temp_bar.date, temp_bar.close])
            raw_bars.append(temp_bar)
            temp_bar = Bar()
            total = total - threshold

    if temp_bar.high is not None:  # High chosen arbitrarily
        temp_bar.updateLastTick(threshold)
        raw_bars.append(temp_bar)
        bars.append([temp_bar.date, temp_bar.close])

    df = pd.DataFrame(bars, columns=['Date', "Close"])
    if returnBars:
        return df, raw_bars
    return df


def volumeBars(data, threshold, returnBars=False):

    dict_data = data.to_dict('records')
    bars = []
    raw_bars = []
    total = 0
    temp_bar = Bar()
    for i, d in enumerate(dict_data):
        temp_bar.addTick(data.index[i], d)
        total += d['Volume']
        if total > threshold:
            temp_bar.updateLastTick(threshold)
            bars.append([temp_bar.date, temp_bar.close])
            raw_bars.append(temp_bar)
            temp_bar = Bar()
            total = total - threshold

    temp_bar.updateLastTick(threshold)
    raw_bars.append(temp_bar)
    bars.append([temp_bar.date, temp_bar.close])

    df = pd.DataFrame(bars, columns=['Date', "Close"])
    if returnBars:
        return df.set_index('Date'), raw_bars
    return df.set_index('Date')


# def plotBars(time, bars, ax, color='g'):
#     index = [i for i in range(len(bars))]
#     _open = [bar.open for bar in bars]
#     _close = [bar.close for bar in bars]
#     _high = [bar.high for bar in bars]
#     _low = [bar.low for bar in bars]

#     candlestick_ohlc(ax, zip(index, _open, _high, _low,
#                              _close), colorup='g', width=0.6)


def main():
    data = pd.read_csv('SPY.csv', parse_dates=[0])

    dbars, rawBars = dollarBars(data, 2e11, returnBars=True)

    haBars = Heikin_Ashi(rawBars, returnBars=True)

    ax1 = plt.subplot(211)
    plt.title('Heikin_Ashi Transformation')
    plotBars(dbars, haBars, ax1)
    ax2 = plt.subplot(212)
    ax2.set_title('Standard Bars')
    plotBars(dbars, rawBars, ax2)
    plt.show()
