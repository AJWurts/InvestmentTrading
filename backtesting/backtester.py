import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.Algorithms import wurtsAlgorithm, alwaysBuy, randomBuySell, keepAt50k, mlalgo, reset
from machinelearning.fracdiff import fracDiff, getWeights
from tqdm import tqdm
from trade import Position, Trade, Trades



STARTING_MONEY = 25000  # $100,000
COMMISSION = 5  # $5 commission




def backTester(algorithm, close_prices, config={"tsl": 0.95, "pt": 1.1, "exp": None, 'freq': 1}, ticker='UNH'):
    value_history = []
    cash = STARTING_MONEY
    stock_owned = 0
    trades = Trades()
    historical_positions = []
    positions = []

    if config['exp'] is None:
        config['exp'] = 10000  # Should never happen
    for i, p in tqdm(enumerate(close_prices), total=len(close_prices)):

        choice, amt = algorithm(p, cash, stock_owned, ticker=ticker, last=i >= len(close_prices) - 1)

        keepPositions = []

        for pos in positions:
            # Sell Position if any of the cell criteria are met.
            if pos.pt <= p or pos.exp <= i or i == len(close_prices) - 1 or pos.maxVal * pos.tsl > p:
                pos.setSellTime(i, p)
                historical_positions.append(pos)
                cash += (pos.qtn * p - COMMISSION)
                trades.addTrade(Trade(i, pos.qtn * p, 'sell'))
                stock_owned -= pos.qtn
            else:
                keepPositions.append(pos.newVal(p))

        positions = keepPositions[:]        

        ## Will only initialize trades when on the chosen frequency day
        if i % config['freq'] != 0:
            value_history.append(cash + stock_owned * p)

            continue  # Not allowed to trade this day

        
        if choice == 'buy':
            if cash < amt * p:
                amt = int(cash / p)
            positions.append(Position(
                i, amt, tsl=config['tsl'], pt=p * config['pt'], exp=i + config['exp'], buy_price=p))
            trades.addTrade(Trade(i, amt, 'buy'))
            stock_owned += amt
            cash -= (amt * p + COMMISSION)

        if i % 500 == 0:
            print(str(i), cash + stock_owned * p)
        value_history.append(cash + stock_owned * p)
    cash += stock_owned * close_prices[-1]
    return cash, value_history, trades, historical_positions


def start_backtest(tickers, time='normal2', algo=mlalgo):
    # data = pd.read_csv('../data/forex_all.csv')
    data_set = [pd.read_csv('./data/' + ticker + '_test.csv')
                for ticker in tickers]

    configs = {
        "long": {"tsl": 0.95, "pt": 1.2, "exp": None, "freq": 5},
        "medium": {"tsl": 0.97, "pt": 1.05, "exp": 8, "freq": 2},
        "short": {"tsl": 0.97, "pt": 1.03, "exp": 3, "freq": 1},
        "normal": {"tsl": 0.9, "pt": 1.2, "exp": None, "freq": 1},
        "normal2": {"tsl": 0.95, "pt": 1.05, "exp": None, "freq": 1}
    }

    config = configs[time]
    # names = ["Crossing MA", "Buy and Hold", "Machine Learning"]
    # algorithms = [n_algorithm1, n_algorithm2, n_algorithm3]
    # names = ['algo 1', 'algo 2', 'algo 3']
    if (len(tickers) == 1):
        _, axs = plt.subplots(len(tickers), 1, sharex=True)
        axs = [axs]
    else:
        _, axs = plt.subplots(len(tickers), 1, sharex=True)
    for i, data in enumerate(data_set):
        with open('./mlsize/' + tickers[i] + 'mlsize.txt', 'r') as mlsizefile:
            start_val = int(mlsizefile.read())
        # Run Backtester
        result, history, trades, positions = backTester(
            algo, data['Close'].values, config=config, ticker=tickers[i])

        # Save position history to file
        with open('./tradehistory/' + tickers[i] + time + '.txt', 'w') as output:
            output.write(str(positions))
        # Calculate ROI and turn it into a string
        roi = "{:.2f}%".format(
            (result - STARTING_MONEY) / STARTING_MONEY * 100)

        # Set the title of the graph subsection
        axs[i].set_title(tickers[i] + ' return: ' + roi +
                         ' trades: ' + str(len(trades)) + " Min Value: " + str(min(history)))

        # Plot the algorithms asset history
        axs[i].plot([i  for i in range(len(history[start_val:]))],
                    history[start_val:],  color='blue', linewidth=2)
        
        axs[i].plot([i  for i in range(len(history[start_val:]))], data['Close'].values[start_val:] * (STARTING_MONEY / data['Close'].values[start_val]), color='orange', linewidth=2)
        # Plot the Sell Trades in red
        axs[i].scatter(trades.getTradeX('sell', offset=-start_val), [history[t.time]
                                                  for t in trades.tradeLog if t.type == 'sell'], color='red', alpha=0.5)

        # Plot the Buy Trades in green
        axs[i].scatter(trades.getTradeX('buy', offset=-start_val), [history[t.time]
                       for t in trades.tradeLog if t.type == 'buy'], color='green', alpha=0.5)
        reset()
        # Print the results to terminal
        print(tickers[i] + " Result: $" + str(result))
        print("ROI: {:.2f}%".format(
            (result - STARTING_MONEY) / STARTING_MONEY * 100))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        start_backtest(sys.argv[1:])
    else:
        start_backtest(['UNH'])
