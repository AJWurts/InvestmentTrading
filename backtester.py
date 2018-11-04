import pandas as pd
import matplotlib.pyplot as plt
from Algorithms import wurtsAlgorithm, alwaysBuy, randomBuySell, keepAt50k
STARTING_MONEY = 100000 # $100,000
STOCK = 'SPY'




def backTester(algorithm, close_prices):
    value_history = []
    cash = STARTING_MONEY
    stock_owned = 0
    for p in close_prices:
        choice, amt = algorithm(p, cash, stock_owned)
        if choice == 'sell' and stock_owned != 0:
            if stock_owned - amt < 0:
                amt = stock_owned
            cash += amt * p
            stock_owned -= amt
        elif choice == 'buy':
            if cash < amt * p:
                amt = int(cash / p)
            stock_owned += amt
            cash -= amt * p
        
        value_history.append(cash + stock_owned * p)
    cash += stock_owned * close_prices[-1]
    return cash, value_history
        



def main():
    data = pd.read_csv('SPY.csv')
    

    algorithms = [wurtsAlgorithm, alwaysBuy,  keepAt50k]
    names = ["Wurts", "Always Buy", "Keep At 50k"]
    _, axs = plt.subplots(len(algorithms), 1, sharex=True)
    for i, algo in enumerate(algorithms):
        result, history = backTester(algo, data['Close'].values)
        axs[i].set_title(names[i])
        axs[i].plot([i for i in range(len(history))], history,  color='blue', linewidth=2)
        print("Algo 1 Result: $" + str(result))
        print("ROI: {:.2f}%".format((result - STARTING_MONEY) / STARTING_MONEY * 100))
    plt.show()



  




if __name__ == "__main__":
    main()

