'''This is the code from the ipynb with the same name (except matplotlib inline, that does not work unfortunately). 
But I added some code to visualize the distributions as bar chart and added code to create a box plot to identify 
the outliers graphically'''



# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns


#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


# Import supplementary visualizations code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Display a description of the dataset
display(data.describe())
display(data.head(10))

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [29, 23, 95]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)




for i in range(0, 6, 1):            # for all product categories (column headers) do
    label_var = list(data)[i]    # take one category

    labels = pd.DataFrame(data[label_var])     # create labels from that category
    new_data = data.copy().drop(label_var, axis = 1)

# TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size = 0.25, random_state = 42)

# TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state = 42)
    regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print "Some of the product categories are correlated with {} with a R2 score of {:4f}".format(label_var, score)

# Produce a scatter matrix for each pair of features in the data
#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

fig = plt.figure(figsize = (11,5));
for i,j in enumerate(data.columns,1):
    ax=fig.add_subplot(2, 3, i)
    ax.hist(data[j], bins = 25, color = '#00A0A0')
    fig.tight_layout()
    ax.set_xlabel(j)
    ax.set_ylabel("Number of Records")


# TODO: Scale the data using the natural logarithm
log_data = data.apply(lambda x: np.log(x + 1))

fig = plt.figure(figsize = (11,5));
for i,j in enumerate(log_data.columns,1):
    ax=fig.add_subplot(2, 3, i)
    ax.hist(log_data[j], bins = 25, color = '#00A0A0')
    fig.tight_layout()
    ax.set_xlabel(j)
    ax.set_ylabel("Number of Records")


# TODO: Scale the sample data using the natural logarithm
log_samples = samples.apply(lambda x: np.log(x + 1))

# Produce a scatter matrix for each pair of newly-transformed features
#pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Display the log-transformed sample data
display(log_samples)


print''
print''
print''
print''

# OPTIONAL: Select the indices for data points you wish to remove
outliers = []                                            # list of outliers that exist in more than 1 feature

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    # print Q1
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    # print Q3
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)

    # Display the outliers
    print''
    print "Data points considered outliers for the feature '{}':".format(feature)

    outl_list = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    print outl_list


    outl_indices = outl_list.index.values                   # get the indices (clients for each outlier)
    for e in outl_indices:                                  # go through indices
        outliers.append(e)                                  # add the corresponding outliers to a list


count_outl = {}                                             # create dict with all outliers and their occurrence number
for i in outliers:
    count_outl[i] = count_outl.get(i, 0) + 1

mult_outl = {}                                              # dict with outliers that appear more than once
for e in count_outl:
    if count_outl[e] > 1:
        mult_outl[e] = count_outl[e]

print "Outliers that occur various times and thus are NOT removed '{}':".format(mult_outl)

outliers_set = list(set(outliers))                          # create list where each outlier appears only once

for e in mult_outl:                                         # remove all outliers that occur more than once
    outliers_set.remove(e)

print "Outliers that occur only once and thus are removed '{}':".format(outliers_set)

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers_set]).reset_index(drop=True)

fig = plt.figure(figsize = (11,5));
for i,j in enumerate(good_data.columns,1):
    ax=fig.add_subplot(2, 3, i)
    ax.hist(good_data[j], bins = 25, color = '#00A0A0')
    fig.tight_layout()
    ax.set_xlabel(j)
    ax.set_ylabel("Number of Records")


fig = plt.figure(figsize = (11,5));
sns.boxplot(x=data)

fig = plt.figure(figsize = (11,5));
sns.boxplot(x=log_data)

fig = plt.figure(figsize = (11,5));
sns.boxplot(x=good_data)

plt.show()



