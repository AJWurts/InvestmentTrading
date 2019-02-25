
class Position:
    def __init__(self, time, qtn, sl=None, pt=None, exp=None, buy_price=None):
        self.start_time = time
        self.sell_time = None
        self.sell_price = None
        self.buy_price = buy_price
        self.qtn = qtn
        self.sl = sl
        self.pt = pt
        self.exp = exp

    def setSellTime(self, sell_time, sell_price):
        self.sell_time = sell_time
        self.sell_price = sell_price

    def __repr__(self):
        return "Return: {2}, Pct return: {0}, Length: {1}\n".format(
            (self.sell_price - self.buy_price) / self.buy_price,
            self.sell_time - self.start_time,
            (self.sell_price * self.qtn) - (self.buy_price * self.qtn)
        )
