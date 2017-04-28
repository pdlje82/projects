import numpy as np
import pandas as pd
from time import time
#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
#print(data.head(n=1))

#print data.shape
#print data[data.income=='>50K'].shape[0]
#skewed = ['capital-gain', 'capital-loss']

#print data[skewed]
#print data[skewed].apply(lambda x: np.log(x + 1))

#s = pd.Series(list('abca'))
#print s
#print pd.get_dummies(s)

s1 = ['a', 'b', np.nan]
print s1
print pd.get_dummies(s1, dummy_na=True)

#features = pd.get_dummies(features_raw)

income_raw = data['income']
print income_raw
