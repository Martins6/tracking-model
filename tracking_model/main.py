from tracking_model.qp import qp_solver
import pandas as pd
import numpy as np
from collections.abc import Iterable


def track(
    data:pd.DataFrame,
    index_name:str,
    P:int,
    cross:float,
    mut:float,
    K:int,
    limits:np.ndarray,
    max_time:float
) -> Iterable[ Iterable[str], Iterable[float], float]:
    """Tracking model using Genetic Algorithm and Quadratic Optimization as shown in Amorim et al. (2020).
    See more in the references. The objective is to imitate a time-series, called 'Index', composed of a linear
    combination of other time-series, called 'stocks', with a fixed quantity of those stocks.
    

    Args:
        data (pd.DataFrame): data
        main_ts (str): the main 
        P (int): initial population size.
        cross (float): probability of crossover.
        mut (float): probability of mutation.
        K (int): the maximum quantity of stocks to invest.
        limits (np.ndarray): the lower and upper boundary to invest of each stock.
        It is a np.ndarray of shape (K, 2).
        max_time (float): maximum time for the algorithm to run. If it takes longer than this,
        returns the current best.

    Returns:
        Iterable[ Iterable[str], Iterable[float], float]: Tuple containing the names,
        the weights and the total time of the operation.
    """
    
    
    return names, weights, time





if __name__ == '__main__':
    T = 1000
    s1=np.random.normal(size=T)
    s2=np.random.normal(size=T)
    s3=np.random.normal(size=T)
    index = s1*0.6 + s2*0.1 + s3*0.3

    df = pd.DataFrame({
        's1': s1,
        's3': s3,
        's2': s2,
        'index': index
    })
    
    df = df.apply(lambda x: (np.diff(x) / x[1:] ))
    print(df.isna().mean())
    
    
    
    