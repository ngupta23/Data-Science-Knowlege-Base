import pandas as pd  # type: ignore
import time

def stats_return_df(data, colname='value', mean_offset=0):
	# Imitate slow process
	time.sleep(1)
	returned = pd.DataFrame({
		'mean':  [data[colname].mean() + mean_offset],
		'sum':  [data[colname].sum()],
		'std':  [data[colname].std()],
		}
	)
	return returned
	
def stats_return_series(data, colname='value', mean_offset=0):
	# Imitate slow process
	time.sleep(1)
	returned = pd.Series({
		'mean':  data[colname].mean() + mean_offset,
		'sum':  data[colname].sum(),
		'std':  data[colname].std(),
		}		
	)
	return returned
	
def stats_return_tuple_floats(data, colname='value', mean_offset=0):
	# Imitate slow process
	time.sleep(1)
	return data[colname].mean() + mean_offset, data[colname].sum(), data[colname].std()
	
def stats_return_tuple_mixed1(data, colname='value', mean_offset=0):
	"""
	Mixed Values are returned (DataFrame, Series, Float)
	DataFrame has 1 row per group with multiple columns
	"""
	# Imitate slow process
	time.sleep(1)
	returned1 = pd.DataFrame({
		'mean':  [data[colname].mean() + mean_offset],
		'median': [data[colname].median()]
		}
	)
	returned2 = pd.Series({
		'sum':  data[colname].sum(),
		}		
	)	
	return returned1, returned2, data[colname].std()
	
def stats_return_tuple_mixed2(data, colname='value', mean_offset=0):
	"""
	Mixed Values are returned (DataFrame, Series, Float)
	DataFrame has multiple rows per group with single columns
	"""
	# Imitate slow process
	time.sleep(1)
	returned1 = pd.DataFrame({
		'stats':  [data[colname].mean() + mean_offset, data[colname].median()]		
		},
		index=['mean', 'median']		
	)
	returned2 = pd.Series({
		'sum':  data[colname].sum(),
		}		
	)	
	return returned1, returned2, data[colname].std()

