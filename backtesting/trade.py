
class Position:
    def __init__(self, time, qtn, sl=None, pt=None, exp=None):
        self.start_time = time
        self.sell_time = None
        self.qtn = qtn
        self.sl = sl
        self.pt = pt
        self.exp = exp
    
    def setSellTime(sell_time):
        self.sell_time = sell_time):
    