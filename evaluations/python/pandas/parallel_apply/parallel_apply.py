# Type Checking
from typing import Callable, Union, Tuple, Optional

# Parallelization
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

def apply_grouped_parallel_with_args(
    grouped_data: DataFrameGroupBy,
    func: Callable,
    unpacked:bool=False,
    reset_inner_index:bool=False,
    n_jobs: Optional[int] = None,
    **kwargs
    ) -> Union[Tuple, pd.Series, pd.DataFrame]:       
    
    """Function to parallelize the Pandas `apply` method on a `groupby` object 

    Parameters
    ----------
    grouped_data : DataFrameGroupBy
        Grouped Pandas DataFrame
    func : Callable
        Function to apply on each group
    unpacked : bool, optional
        If the returned values are a tuple of multiple values, should they be separated into a list of values
        (1 entry in list per returned value) or as a pandas Series of tuples (1 row per group). If True,
        it is returned as a list. If False, it is returned as a pandas Series., by default False
    reset_inner_index : bool, optional
        (UNUSED RIGHT NOW) If unpacking, if one of the returned values is a pandas DataFrame,
        should the indices of the returned values be reset or left as is., by default False
    n_jobs : int, optional
        Number of cores to use. If None, uses 90% of cores., by default None
    **kwargs: optional
        Additional arguments to be passed to `func`

    Returns
    -------
    Union[Tuple, pd.Series, pd.DataFrame]
        Depends on the output of `func` and the arguments selected
    """  
      
    # https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
    # https://stackoverflow.com/questions/39686957/parallelize-groupby-with-multiple-arguments
    
    if n_jobs:
        p = Pool(n_jobs)    
    else:
        p = Pool(round(cpu_count() * 0.9))
    ret_list = p.map(partial(func, **kwargs), [group for _, group in grouped_data])
    p.close()
    
    # ret_list will have one object per group
    # If the function is returning a single value, then the object will be a DataFrame per group
    # If the function is returning multiple Dataframes, then the object will be a Tuple of DataFrames
    # We need to handle these situations appropriately
    indices = list(grouped_data.groups.keys())
    
    # Function 'func' is returning 1 DataFrame per group
    if isinstance(ret_list[0], (pd.DataFrame, pd.Series)):
        # Returning Value: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
        return_obj = pd.concat(ret_list, axis = 0)
        return return_obj 
      
    # Function 'func' is returning a tuple of values per group
    # Each value could be a DataFrame, Series or something else
    if isinstance(ret_list[0], tuple):
        return_obj = pd.Series(ret_list)
        return_obj.index = indices
        
        if not unpacked:
            return return_obj  
        
        
        def unpack(col, index):
            # If the initial unpack returns a series of DataFrames (one dataframe for each group)
            # then we need to further unpack the dataframes and concatenate them together
            if isinstance(col, pd.Series) and isinstance(col.iloc[0], pd.DataFrame):
                temp = [col.iloc[i] for i in range(len(col))]
                temp = pd.concat(temp, keys=index, names=['group'], axis=0)
                return temp
            return col 
            
        def return_col(data, col_index):
            # Lambda function was giving this error: https://jira.sonarsource.com/browse/SONARPY-587
            return data[col_index]
            
        initial_unpack = [return_obj.apply(return_col, col_index=j) for j in range(len(return_obj[0]))]
        final_unpack = [unpack(item, index=indices) for item in initial_unpack]
        return final_unpack
    
    raise(TypeError("Currently only supports functions that return a Pandas DataFrame, Pandas Series or a Tuple of values"))  
