from tracking_model.qp import qp_solver
from random import randint, choices, uniform, sample
import pandas as pd
import numpy as np
from time import time

def track_index(
    data:pd.DataFrame,
    K:int,
    index_name:str,
    P:int=15,
    cross:float=1,
    mut:float=0.85,
    cut:int=4,
    weight_limits:np.ndarray=None,
    max_time:float=10,
    mse_limit:float=5*(10**(-10))
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
    t0 = time()
    N = data.loc[:, ~data.columns.str.match(index_name)].shape[1]
    pop = list(gen_initial_pop(N, P, K))
    stop = False
    flag_mse_limit = False
    results_df = None
    while not stop:
        children = None
        if uniform(0, 1) <= cross:
            parents = choices(pop, k=2)
            children = crossover(parents, cut, K)
            
            if uniform(0, 1) <= mut:
                children = mutate(children)

            pop.extend(children)
            results_df, flag_mse_limit = select_top(pop, data,
                                                    mse_limit, index_name)
            pop = results_df['pop'].to_numpy().tolist()

        if((time() - t0) >= max_time) or (flag_mse_limit == True):
            stop = True
        
    return results_df.head(1)['obj_values'][0], (time() - t0)


def gen_initial_pop(N:int, P:int, K:int):
    for i in range(P):  
        initial_population = np.zeros((N,), dtype=int)
        sample_values = sample(list(range(N)), k=K)
        for activated_index in sample_values:
            initial_population[activated_index] = 1
        yield initial_population
        

def crossover(parents:list, cut:float, K:int):
    def correct(child:np.array):
        index_array = np.array(range(len(child)))
        
        while child.sum() > K:
            index_of_change = index_array[child == 1]
            i_change = sample(index_of_change.tolist(), 1)
            child[i_change] = 0
        
        while child.sum() < K:
            index_of_change = index_array[child == 0]
            i_change = sample(index_of_change.tolist(), 1)
            child[i_change] = 1
        
        return child
    
    def cut_and_join(indexes:list):
        cut_parent_1 = parents[indexes[0]].tolist()[0:cut]
        cut_parent_2 = parents[indexes[1]].tolist()[cut:]
        child = np.array(cut_parent_1 + cut_parent_2, dtype=int)
        
        if child.sum() != K:
            child = correct(child)
        return child
        
    children = []
    children.append(cut_and_join([0,1]))
    children.append(cut_and_join([1,0]))
    
    return children


def mutate(children:list):
    for child in children:
        index_array = np.array(range(len(child)))
        
        on = index_array[child == 1]
        on_sample = sample(on.tolist(), 1)    
        off = index_array[child == 0]
        off_sample = sample(off.tolist(), 1)
        
        child[on_sample] = 0
        child[off_sample] = 1
    return children


def select_top(
    pop:list,
    data:pd.DataFrame,
    mse_limit:float,
    index_name:str
) -> tuple:
    data = data.copy()
    
    select_data = pd.DataFrame({
        'pop':pop,
        'obj_values':[objective_fun(i, data, index_name) for i in pop]
    }).sort_values(
        by='obj_values', key = lambda col: col.apply(lambda elem: elem['cost value'])).\
    reset_index()

    select_data['cost'] = select_data['obj_values'].apply(lambda x: x['cost value'])
    select_data['check_mse_list'] = select_data['cost'] <= mse_limit
    check_mse = any(select_data['check_mse_list'][:-2].to_numpy())
    
    n = select_data.shape[0]
    select_data = select_data.loc[:, ['pop', 'obj_values']].head(n - 2)
    
    return select_data, check_mse



def objective_fun(
    element:np.ndarray,
    data:pd.DataFrame,
    index_name:str
) -> tuple:
    data = data.copy()
    selected_data = data.loc[:, ~data.columns.str.match(index_name)]
    selected_data = selected_data.loc[:, element == 1]
    selected_data[index_name] = data[index_name]
    
    qp = qp_solver(df=selected_data, col_index=index_name)
    sol = qp.solve()
    
    return {'weights': sol['x'],
            'names': qp.weights,
            'cost value': sol['cost value']}

