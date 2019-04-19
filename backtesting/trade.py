
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
