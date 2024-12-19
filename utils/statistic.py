import pandas as pd
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def horizon_volatility_log_return(prices, horizon) :
    """
    Input : 
        mid_prices : numpy array containing prices
    Returns : 
        A numpy array containThe log return of the prices
    """

    
    log_returns = np.diff(np.log(prices))
    windows = sliding_window_view(log_returns, horizon)
   
    std = np.std(windows, axis=1) * np.sqrt(horizon)

    return std


