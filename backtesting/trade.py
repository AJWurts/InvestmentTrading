
class Trade:
    def __init__(self, time, qtn, t_type):
        self.time = time
        self.qtn = qtn
        self.type = t_type


class Trades:
    def __init__(self):
        self.numberOfTrades = 0
        self.tradeLog = []

    def addTrade(self, trade):
        self.numberOfTrades += 1
        self.tradeLog.append(trade)

    def getTradeList(self):
        x = []
        y = []
        for t in self.tradeLog:
            x.append(t.time)
            y.append(t.qtn)

        return x, y

    def getTradeX(self, t_type='buy', offset=0):
        x = []
        for t in self.tradeLog:
            if t.type == t_type:
                x.append(t.time + offset) 

        return x

    def getTradeY(self, t_type='buy'):
        y = []
        for t in self.tradeLog:
            if t.type == t_type:
                y.append(t.qtn)

        return y

    def __len__(self):
        return len(self.tradeLog)

    def __str__(self):
        return str(self.tradeLog)

class Position:
    def __init__(self, time, qtn, tsl=None, sl=None, pt=None, exp=None, buy_price=None, maxVal=None):
        self.start_time = time
        self.sell_time = None
        self.sell_price = None
        self.buy_price = buy_price
        self.qtn = qtn
        self.sl = sl
        self.pt = pt
        self.exp = exp
        self.tsl = tsl
        self.maxVal = buy_price

    def setSellTime(self, sell_time, sell_price):
        self.sell_time = sell_time
        self.sell_price = sell_price

    def newVal(self, p):
        if self.maxVal is None or p > self.maxVal:
            self.maxVal = p
        
        return self

    def __repr__(self):
        return "Return: {2}, Pct return: {0}, Length: {1}\n".format(
            (self.sell_price - self.buy_price) / self.buy_price,
            self.sell_time - self.start_time,
            (self.sell_price * self.qtn) - (self.buy_price * self.qtn)
        )
