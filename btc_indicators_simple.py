from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

client = Client('', '')

#get btc historical data from binance api
print("getting historical data...")
data = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2019", "1 July, 2020")
cols = ['time','o','h','l', 'close', 'v','ct','qav','nt','TBBAV','TBQAV','null']
data = pd.DataFrame(data, columns=cols)
data = data[['time', 'close']]

#add indicators to our dataframe to use as features and price differences
data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()
data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
data['close'] = data['close'].astype(np.float64)
data['diff'] = data['close'].shift(-1) - data['close']
data['diffp'] = data['diff']/data['close']
data['diffs'] = np.where(data['diff'] > 0, 1, 0)

#split janurary~june: train, july~december: test
days = 183*24
y_train = data.loc[:days, 'diffs']
X_train = data.loc[:days, ['EMA20', 'MACD']]
y_test = data.loc[days:, ['diffs', 'diffp']]
X_test = data.loc[days:, ['EMA20', 'MACD']]

#train classifiers
random_forest_clf = RandomForestClassifier(max_leaf_nodes=12).fit(X_train, y_train)
gnb_clf = GaussianNB().fit(X_train, y_train)

rf_pred = random_forest_clf.predict(X_test)
gnb_pred = gnb_clf.predict(X_test)

#compute btc average return
btc_start = data.iloc[days][1]
btc_finish = data.iloc[-1][1]
btc_return = 100 * (btc_finish - btc_start) / btc_start
btc_average = 100 + btc_return

#Gaussian Naive Bayes backtest
cash = 100
for i in range(len(gnb_pred)-1):
  if gnb_pred[i] == y_test.iloc[i]['diffs']:
    cash = cash + (cash * abs(y_test.iloc[i]['diffp']))
  else:
    cash = cash - (cash * abs(y_test.iloc[i]['diffp']))
gnb_return = cash

#Random Forest backtest
cash = 100
for i in range(len(rf_pred)-1):
  if rf_pred[i] == y_test.iloc[i]['diffs']:
    cash = cash + (cash * abs(y_test.iloc[i]['diffp']))
  else:
    cash = cash - (cash * abs(y_test.iloc[i]['diffp']))
random_forest_return = cash

#Results
print("BTC 2019 July: %s" % btc_start)
print("BTC 2020 July: %s" % btc_finish)
print("BTC Year return: %s%%" % btc_return)
print("")
print("Random Forest return: %s%%" % (random_forest_return - 100))
print("Random Forest compared to BTC: %s%%" % (random_forest_return - btc_average))
print("")
print("Gaussian Naive Bayes return: %s%%" % (gnb_return - 100))
print("Gaussian Naive Bayes compared to BTC: %s%%" % (gnb_return - btc_average))