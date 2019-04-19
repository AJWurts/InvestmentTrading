import pandas as pd
import numpy as np

def find_peaks(prices):
    derivative = []
    for i in range(1, len(prices)):
        derivative.append(prices[i] - prices[i-1])

    valleys = []
    peaks = []

    for d in range(1, len(derivative)):
        if derivative[d - 1] > 0 and derivative[d] < 0:
            peaks.append(d)
        if derivative[d] > derivative[d-1]:
            valleys.append(d)

    return valleys, peaks

spy = pd.read_csv('SPY_withpercent.csv')

close = spy['Close']
percent = spy['%Change']

valleys, peaks = find_peaks(spy['Close'])

print(len(valleys) + len(peaks), len(spy))


close = [0] + close
percent = [0] + percent

buy_points = {}
sell_points = {}

diff_threshold = 0.01
time_length = 100
for p in peaks:
    for v in valleys:
        if  v < p :
            if  close[p] > close[v] and ((close[p] - close[v]) / close[v]) > diff_threshold and p - v <= time_length:
                if v in buy_points:
                    buy_points[v] += 1
                else:
                    buy_points[v] = 1
                
                if p in sell_points:
                    sell_points[p] += 1
                else:
                    sell_points[p] = 1
        else:
            break

data_length = 50

buy_zip = []
for key in buy_points.keys():
    buy_zip.append([key, buy_points[key]])
    
buy_zip = sorted(buy_zip, key=lambda x: x[1], reverse=True)

with open('training_data_buy.csv', 'w') as data:
    [data.write(str(i) + ',') for i in range(data_length)]
    data.write('action,index' + '\n')
    for buy in buy_zip[:500]:
        if buy[0] > data_length:
            output = [a for a in percent[buy[0] - (data_length - 1): buy[0] + 1]]
            output.append(1)
            output.append(buy[0])
            data.write(str(output)[1:-1].replace(' ', '') + '\n')

sell_zip = []
for key in sell_points.keys():
    sell_zip.append([key, sell_points[key]])

sell_zip = sorted(sell_zip, key=lambda x: x[1], reverse=True)

with open('training_data_sell.csv', 'w') as data:
    [data.write(str(i) + ',') for i in range(data_length)]
    data.write('action,index' + '\n')
    for sell in sell_zip[:500]:
        if sell[0] > data_length:
            output = [a for a in percent[sell[0] - (data_length - 1): sell[0] + 1]]
            output.append(0)
            output.append(sell[0])
            data.write(str(output)[1:-1].replace(' ', '') + '\n')

buy_data = pd.read_csv('training_data_buy.csv')
sell_data = pd.read_csv('training_data_sell.csv')

full = pd.concat([buy_data, sell_data])

df = full.sample(frac=1).reset_index(drop=True)
df.to_csv("full_data.csv")
