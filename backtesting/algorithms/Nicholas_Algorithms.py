# return sell, buy, do nothing and if buying say how many stocks

previous_prices = []


def n_algorithm1(p, cash, stock_owned):
    previous_prices.append(p)
    percent_change = (p - previous_prices[-1]) / previous_prices[-1] * 100
    if percent_change > 4.5:
        return 'sell', stock_owned
    elif percent_change < 1.5:
        num_buy = int((cash/4) / p)
        return 'buy', num_buy
    else:
        return 'do nothing', 0


def n_algorithm2(p, cash, stock_owned):
    previous_prices.append(p)
    percent_change = (p - previous_prices[-1]) / previous_prices[-1] * 100
    if percent_change > 50:
        return 'sell', stock_owned
    elif percent_change < 1:
        num_buy = int((cash/4) / p)
        return 'buy', num_buy
    else:
        return 'do nothing', 0


def n_algorithm3(p, cash, stock_owned):
    previous_prices.append(p)
    percent_change = (p - previous_prices[-1]) / previous_prices[-1] * 100
    if percent_change < -4.5:
        num_buy = int((cash / 4) / p)
        return 'buy', num_buy
    elif percent_change > 4.5:
        return 'sell', stock_owned
    else:
        return 'do nothing', 0