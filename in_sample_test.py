## in_sample_test.py
## 2022.11.25
## inputs: signal description, universe, time
## outputs: signal quality metrics (sharpe ratio, pnl, turnover, fitness, drawdown)

import pandas as pd
import requests
from signals import five_day_reversion
from math import sqrt

US10 = ["AAPL", "MSFT", "GOOG", "AMZN", "BRK-A", "TSLA", "UNH", "XOM", "JNJ", "V"]

config = {
	'universe': US10,
	'start': '2019-01-01',
	'end': '2022-12-31',
	'signal': five_day_reversion,
	'neutralization': 'all',
	'investment': 50000,
	'API_KEY': '3780b5d18a15253a9e1d33d483ad4dccd49ae20e',
}

def get_data(universe, start, end):
	## import data from universe
	df = pd.DataFrame()
	for ticker in universe:
		url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
		response = requests.get(url, params={"token": config['API_KEY'], 'startDate': start, 'endDate': end})
		data = response.json()
		data = pd.DataFrame(data)
		data = data[['date', 'close', 'volume']]
		data['ticker'] = ticker
		df = pd.concat([df, data], axis=0)
	df['date'] = pd.to_datetime(df['date']).dt.date
	df.set_index(['date', 'ticker'], inplace=True)
	return df

def gen_positions(df, investment):
	## generate position vectors for each day
	df['signal'] = config['signal'](df)
	if config['neutralization'] == 'all':
		short_groups = df[df['signal'] < 0].groupby(level=0)
		short_scale = -investment / short_groups.sum()
		df["position"] = short_scale['signal'] * df['signal']

		long_groups = df[df['signal'] >=0].groupby(level=0)
		long_scale = investment / long_groups.sum()
		df.loc[df['signal'] >=0, "position"] = long_scale['signal'] * df['signal']
	return df

def gen_pnl(df):
	## generate pnl vectors from position vectors
	df['pnl'] = df.groupby(['ticker'])['close'].pct_change()
	df['pnl'] = df['position'] * df['pnl']
	return df

def gen_metrics(df, investment):
	## generate signal metrics from pnl vectors
	df['trading'] = df['position'].groupby(['ticker']).diff(periods=1).abs()
	df = df.dropna()
	df = df.reset_index(level=1)
	df.index = pd.to_datetime(df.index)
	metrics = df['pnl'].groupby(pd.Grouper(freq='Y')).sum().to_frame()
	metrics.columns = ['total_pnl']
	metrics['pnl'] = 100 * metrics['total_pnl'] / (investment * 2) # profit and loss
	metrics['ir'] = metrics['pnl'] / df['pnl'].groupby(level=0).sum().groupby(pd.Grouper(freq='Y')).std() # information ratio
	metrics['sharpe'] = metrics['ir']*sqrt(252) # sharpe ratio
	metrics['turnover'] = 100 * df['trading'].groupby(pd.Grouper(freq='Y')).sum() / (investment * 252 * 2) # turnover
	metrics['margin'] = 100 * metrics['total_pnl'] / df['trading'].groupby(pd.Grouper(freq='Y')).sum() # margin
	metrics['fitness'] = metrics['sharpe'] * (metrics['pnl'].abs() / metrics['turnover']).apply(sqrt) # fitness ratio
	return df, metrics

def gen_drawdown(df, investment):
	drawdowns = []
	for year, j in df['pnl'].groupby(pd.Grouper(freq='Y')):
		max_drawdown = 0
		for i in range(len(j.groupby(level=0))):
			val = j.groupby(level=0).sum().iloc[i]
			if val<0:
				curr=i
				drawdown = val
				if curr == len(j.groupby(level=0))-1: break
				while(j.groupby(level=0).sum().iloc[curr+1] < 0):
					drawdown += j.groupby(level=0).sum().iloc[curr+1]
					curr += 1
					if curr == len(j.groupby(level=0))-1: break
				max_drawdown = min(max_drawdown, drawdown)
		drawdowns.append(max_drawdown)
	
	return [- 100 * x / investment for x in drawdowns]

if __name__ == "__main__":
	df = get_data(config['universe'], config['start'], config['end'])
	df = gen_positions(df, config['investment'])
	df = gen_pnl(df)
	df, metrics = gen_metrics(df, config['investment'])
	metrics['drawdowns'] = gen_drawdown(df, config['investment'])
	print(metrics)
	metrics.to_csv(f"D:\\proj\\trading_system\\{config['signal'].__name__}.csv")