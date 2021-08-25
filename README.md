# tracking-model

_tracking_model_ is a Python Package that implements Genetic Algorithm as shown in Amorim et al. (2020) in order to solve the **index-tracking** problem. Consider a time-series that is completly given by a weighted linear combination of other time-series. We shall call this time-series 'index'. The **index-tracking** problem is formed by the challenge of representing this same index with fewer time-series that originated the index.

Say that we have an index formed of 500 economical series, such as the S&P500. We wish to try to imitate the behavior of the S&P500 time-series of returns with only 10 time-series. The reason for that could be because of investment restrictions in general, or because it is harder to track 500 series rathen than just 10, etc.

This problem could be solved using quadratic or linear programming on the possible combinations of sub-collection of time-series, however to find the best weighted linear combination of the original collection of time-series could prove to take a while. This is where the Genetic Metaheuristic comes into play. We can dramatically speed-up this process.

## Installing

You can install it via [pip](https://pip.pypa.io/en/stable/getting-started/) or [poetry](https://python-poetry.org/).

```python
pip install tracking-model
```

or 

```python
poetry add tracking-model
```

## References 

The references for the quadratic programming and the paper that inspired this package is given at the _references/_ folder. 

## Example

```python
import pandas as pd
import numpy as np
# Quadratic Programming 
from tracking_model.qp import qp_solver
# Genetic Metaheuristic Optimization
from tracking_model.model import track_index

T = 1000
s1=np.random.normal(size=T)
s2=np.random.normal(size=T)
s3=np.random.normal(size=T)
s4=np.random.normal(size=T)
s5=np.random.normal(size=T) 
index = s1*0.5 + s2*0.05 + s3*0.05 + s4*0.2 + s5*0.3

df = pd.DataFrame({
    's1': s1,
    's2': s3,
    's3': s2,
    's4': s4,
    's5': s5,
    'index': index
})

# Solution with all time-series
qp = qp_solver(df)
sol = qp.solve()
print('Exact solution with all time-series')
print(sol['x'])
print(qp.weights)
print(sol['cost value'])
print('\n')

# Metaheuristic solution with fewer time-series (K=3)
print('Genetic Algorithm Solution with fewer time-series')
print(track_index(df, K=3, index_name='index', P=5, cut = 2, max_time=3))
```
