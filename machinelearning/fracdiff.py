import numpy as np
import pandas as pd


def getWeights(d, threshold=1e-5):
	w = [1.]
	k = 1
	while True:
		w_ = -w[-1] * ( (d-k+1) / k )

		if abs(w_) < threshold:
			break
		w.append(w_)
		k+=1
	
	w = np.array(w[::-1]).reshape(-1,1)
	print(w.size)
	return w


def fracDiff(series, d=0.5, thres=0.01):
	# Compute Weights
	w = getWeights(d, thres)
	width = len(w) - 1

	df = {}
	for name in series.columns:
		seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
		for iloc1 in range(width, seriesF.shape[0]):
			loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
			if not np.isfinite(series.loc[loc1, name]):
				continue

			df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]

		df[name] = df_.copy(deep=True)
	df = pd.concat(df, axis=1)
	return df

def getFracDiffCSV(filename, d=0.5, thres=0.01):
	df = pd.read_csv(filename)
	df = df.set_index('Date')

	df = fracDiff(df, d, thres)

	return df

def plotExample():
	import matplotlib.pyplot as plt
	from fracdiff import fracDiff
	df = pd.read_csv('./data/SPY.csv')
	df = df.set_index('Date')
	df = df.drop(['Volume', 'Adj Close', 'High', 'Open', 'Low'], axis=1)
	ds = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]
	_, axs = plt.subplots(len(ds), 1, sharex=True)
	for i, algo in enumerate(ds):
			df_ = fracDiff(df, d=algo)
			# print(result, history)
			axs[i].set_title("d: " + str(algo))
			df_.plot(ax=axs[i])
	plt.show()


if __name__ == "__main__":
	df = getFracDiffCSV("../data/SPY_2019.csv")
	df.to_csv("../data/SPY_2019_frac.csv")