## in_sample_test.py
## 2022.11.25
## inputs: signal description, universe, time
## outputs: signal quality metrics (sharpe ratio, pnl, turnover, fitness, drawdown)

import pandas as pd
import pandas_datareader.data as pdr
from signals import five_day_reversion
from math import sqrt

# get S&P 500 tickers list from wikipedia
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500 = table[0]["Symbol"].values.tolist()
universes = {"US10": ["AAPL", "MSFT", "GOOG", "AMZN", "BRK-A", "TSLA", "UNH", "XOM", "JNJ", "V"], "SP500": sp500}

config = {
	'universe': universes["US10"],
	'start': '2017-01-01',
	'end': '2020-12-31',
	'signal': five_day_reversion,
	'neutralization': 'all',
	'investment': 50000
}

def in_sample_test():
	## import data from universe
	df = pdr.DataReader(config['universe'], 'yahoo', config['start'], config['end'])
	df = df[['Close', 'Volume']] # drop unneeded data
	df = df.stack() # let tickers be a column instead of row
	df.index.names = ['date', 'ticker']
	df.columns = ['close', 'volume']
	df, signal_name = config['signal'](df)

	## generate position vectors for each day
	if config['neutralization'] == 'all':
		short_groups = df[df[signal_name] < 0].groupby(level=0)
		short_scale = -config['investment'] / short_groups.sum()
		df[signal_name + "_position"] = short_scale[signal_name] * df[signal_name]

		long_groups = df[df[signal_name] >=0].groupby(level=0)
		long_scale = config['investment'] / long_groups.sum()
		df.loc[df[signal_name] >=0, signal_name + "_position"] = long_scale[signal_name] * df[signal_name]

	## generate pnl vectors from position vectors
	df['pnl'] = df.groupby(['ticker'])['close'].pct_change()
	df[signal_name + "_pnl"] = df[signal_name + '_position'] * df['pnl']

	## generate signal metrics from pnl vectors
	df['trading'] = df[signal_name+'_position'].groupby(['ticker']).diff(periods=1).abs()
	df = df.dropna()
	df = df.reset_index(level=1)
	df.index = pd.to_datetime(df.index)
	metrics = df[signal_name + '_pnl'].groupby(pd.Grouper(freq='Y')).sum().to_frame()
	metrics.columns = ['total_pnl']
	metrics['pnl'] = 100 * metrics['total_pnl'] / (config['investment'] * 2)
	metrics['ir'] = metrics['pnl'] / df[signal_name + '_pnl'].groupby(level=0).sum().groupby(pd.Grouper(freq='Y')).std()
	metrics['sharpe'] = metrics['ir']*sqrt(252)
	metrics['turnover'] = 100 * df['trading'].groupby(pd.Grouper(freq='Y')).sum() / (config['investment'] * 252 * 2)
	metrics['margin'] = 100 * metrics['total_pnl'] / df['trading'].groupby(pd.Grouper(freq='Y')).sum()
	metrics['fitness'] = metrics['sharpe'] * (metrics['pnl'].abs() / metrics['turnover']).apply(sqrt)

	drawdowns = []
	for year, j in df[signal_name + '_pnl'].groupby(pd.Grouper(freq='Y')):
		max_drawdown = 0
		for i in range(len(j.groupby(level=0))):
			val = j.groupby(level=0).sum().iloc[i]
			if val<0:
				curr=i
				drawdown = val
				while(j.groupby(level=0).sum().iloc[curr+1] < 0):
					drawdown += j.groupby(level=0).sum().iloc[curr+1]
					curr += 1
				if drawdown < max_drawdown: 
					max_drawdown = drawdown
		drawdowns.append(max_drawdown)
	
	metrics['drawdown'] = [- 100 * x / config['investment'] for x in drawdowns]

	print(metrics)
	metrics.to_csv("D:\\proj\\trading_system\\metrics.csv")


	#print(df.tail())
	#df.to_csv("D:\\proj\\trading_system\\dat_df.csv")

if __name__ == "__main__":
	in_sample_test()