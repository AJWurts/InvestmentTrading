import pandas as pd
import matplotlib.pyplot as plt
from Algorithms import wurtsAlgorithm, alwaysBuy, randomBuySell, keepAt50k, mlalgo, reset
from Nicholas_Algorithms import n_algorithm1, n_algorithm2, n_algorithm3
STARTING_MONEY = 100000 # $100,000
STOCK = 'SPY'




def backTester(algorithm, close_prices):
    value_history = []
    cash = STARTING_MONEY
    stock_owned = 0
    for i, p in enumerate(close_prices):
        # print(str(i) + "/" + str(p))
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
    data = pd.read_csv('./data/SPY.csv')
    


    algorithms = [wurtsAlgorithm, alwaysBuy,  mlalgo]
    # names = ["Wurts", "Always Buy", "Keep At 50k"]
    # algorithms = [n_algorithm1, n_algorithm2, n_algorithm3]
    names = ['algo 1', 'algo 2', 'algo 3']
    _, axs = plt.subplots(len(algorithms), 1, sharex=True)
    for i, algo in enumerate(algorithms):
        result, history = backTester(algo, data['Close'].values)
        # print(result, history)
        axs[i].set_title(names[i])
        axs[i].plot([i for i in range(len(history))], history,  color='blue', linewidth=2)
        reset()
        print(names[i] + " Result: $" + str(result))
        print("ROI: {:.2f}%".format((result - STARTING_MONEY) / STARTING_MONEY * 100))
    plt.show()



  




if __name__ == "__main__":
    main()

