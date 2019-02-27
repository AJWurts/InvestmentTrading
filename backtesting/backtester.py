import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from backtesting.trade import Position
from tqdm import tqdm
from machinelearning.fracdiff import fracDiff, getWeights
from backtesting.algorithms.Nicholas_Algorithms import n_algorithm1, n_algorithm2, n_algorithm3
from backtesting.algorithms.Algorithms import wurtsAlgorithm, alwaysBuy, randomBuySell, keepAt50k, mlalgo, reset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


STARTING_MONEY = 100000  # $100,000
COMMISSION = 5  # $5 commission
STOCK = 'SPY'


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

    def getTradeX(self, t_type='buy'):
        x = []
        for t in self.tradeLog:
            if t.type == t_type:
                x.append(t.time)

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


def backTester(algorithm, close_prices, pt=1.01, sl=0.99, exp=5, ticker='UNH'):
    value_history = []
    cash = STARTING_MONEY
    stock_owned = 0
    trades = Trades()
    historical_positions = []
    positions = []
    for i, p in tqdm(enumerate(close_prices), total=len(close_prices)):
        choice, amt = algorithm(p, cash, stock_owned, ticker=ticker)
        keepPositions = []
        for pos in positions:
            if pos.sl >= p or pos.pt <= p or pos.exp <= i or i == len(close_prices) - 1:
                pos.setSellTime(i, p)
                historical_positions.append(pos)
                cash += (pos.qtn * p - COMMISSION)
                trades.addTrade(Trade(i, pos.qtn * p, 'sell'))
                stock_owned -= pos.qtn
            else:
                keepPositions.append(pos)

        positions = keepPositions[:]
        if choice == 'buy':
            if cash < amt * p:
                amt = int(cash / p)
            positions.append(Position(i, amt, p * sl, p * pt, i + exp, p))
            trades.addTrade(Trade(i, amt, 'buy'))
            stock_owned += amt
            cash -= (amt * p + COMMISSION)

        if i % 500 == 0:
            print(str(i), cash + stock_owned * p)
        value_history.append(cash + stock_owned * p)
    cash += stock_owned * close_prices[-1]
    return cash, value_history, trades, historical_positions


def start_backtest(ticker):
    # data = pd.read_csv('../data/forex_all.csv')
    data = pd.read_csv('./data/' + ticker + '_test.csv')
    algorithms = [wurtsAlgorithm, alwaysBuy,  mlalgo]

    names = ["Crossing MA", "Buy and Hold", "Machine Learning"]
    # algorithms = [n_algorithm1, n_algorithm2, n_algorithm3]
    # names = ['algo 1', 'algo 2', 'algo 3']
    _, axs = plt.subplots(len(algorithms), 1, sharex=True)
    for i, algo in enumerate(algorithms):
        result, history, trades, positions = backTester(algo, data['Close'].values, sl=0.97, pt=1.03, exp=10, ticker=ticker)
        with open(names[i] + '.txt', 'w') as output:
            output.write(str(positions))
        roi = "{:.2f}%".format((result - STARTING_MONEY) / STARTING_MONEY * 100)
        axs[i].set_title(names[i] + 'return: ' + roi + ' trades: ' + str(len(trades)))
        axs[i].plot([i for i in range(len(history))], history,  color='blue', linewidth=2)
        axs[i].scatter(trades.getTradeX('sell'), [history[t.time] for t in trades.tradeLog if t.type == 'sell'], color='red', alpha=0.5)
        axs[i].scatter(trades.getTradeX('buy'), [history[t.time] - 10 for t in trades.tradeLog if t.type == 'buy'] , color='green', alpha=0.5)
        reset()
        print(names[i] + " Result: $" + str(result))
        print("ROI: {:.2f}%".format(
            (result - STARTING_MONEY) / STARTING_MONEY * 100))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        start_backtest(sys.argv[1])
    else:
        start_backtest('UNH')
