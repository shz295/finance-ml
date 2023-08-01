def five_day_reversion(df):
	df['5dr'] = df.groupby(['ticker'])['close'].diff(periods=5) / df['close']
	df['5dr'] = df['5dr'] * (1 / df.abs().groupby(level=0)['5dr'].max())
	return df['5dr']