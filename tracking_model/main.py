from tracking_model.qp import qp_solver
import pandas as pd
import numpy as np
from random import randint, choices


def track_index(
    data:pd.DataFrame,
    K:int,
    index_name:str,
    P:int=15,
    cross:float=1,
    mut:float=0.85,
    cut:int=4,
    weight_limits:np.ndarray=None,
    max_time:float=60,
    mse_limit:float=5*(10**(-8))
) -> tuple:
    """Tracking model using Genetic Algorithm and Quadratic Optimization as shown in Amorim et al. (2020).
    See more in the references. The objective is to imitate a time-series, called 'Index', composed of a linear
    combination of other time-series, called 'stocks', with a quantity less than the original of those stocks.
    

    Args:
        data (pd.DataFrame): time-series of returns dataframe containing the stocks
        and the index. They are ordered in ascending order by the date.
        main_ts (str): the column name of the index
        P (int): initial population size.
        cross (float): probability of crossover.
        mut (float): probability of mutation.
        cut (int): where to cut the binary genome of the fathers (e.g. [0, 1, 1, 0, ...]) to construct the child.
        K (int): the maximum quantity of stocks to invest.
        limits (np.ndarray): the lower and upper boundary to invest of each stock.
        It is a np.ndarray of shape (K, 2). Defaults to None.
        max_time (float): maximum time in seconds for the algorithm to run. If it takes longer than this,
        returns the current best. Defaults to 60 seconds or 1 minutes.
        mse_limit (float): miminum value of the objective function to achieve. If achieved, stop the algorithm.

    Returns:
        (tuple): Tuple containing the names, the weights and the total time of the operation, in that order.
    """
    N = data.loc[:, ~data.columns.str.match(index_name)].shape[1]
    
    pop = list(gen_initial_pop(N, P))
    print(choices(pop, k=2))
    stop = False    
    while not stop:
        
        
    
    return names, weights, time


def gen_initial_pop(N:int, P:int):
    for i in range(P):  
        initial_population = np.zeros((N,))
        for i in range(N):
            activated_index = randint(0, N-1)
            initial_population[activated_index] = 1
        yield initial_population


def objective_fun(
    element:np.ndarray,
    index_name:str,
    data:pd.DataFrame
) -> tuple:
    selected_data = data.loc[:, element]
    
    qp = qp_solver(df=element, col_index=index_name)
    sol = qp.solve()
    return sol['x'], qp.weights, sol['cost value']
    


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
    
    track_index(df, K=2, index_name='index')
    
    
    