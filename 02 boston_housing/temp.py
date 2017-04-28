from sklearn.grid_search import ParameterGrid
import numpy as np
param_grid = {'a': np.linspace(1, 10, 10), 'b': [True, False]}
#list(ParameterGrid(param_grid))

print param_grid



param_grid = {'a': range(1,11), 'b': [True, False]}
print param_grid
