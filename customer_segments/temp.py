from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


for i in range(0, 5, 1):            # for all product categories (column headers) do
    label_var = list(data)[i]    # take one category

    labels = data[label_var]     # create labels from that category
    new_data = data.drop(label_var, axis = 1)

# TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size = 0.25, random_state = 42)

# TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
    y_pred = regressor.predict(X_test)
    print np.shape(y_test)
    print np.shape(y_pred)
    score = regressor.score(y_pred, y_test)
    print "Some of the product categories depend on {} with a R^2 score of {:4f}".format(label_var, score)
    
    
    
    
    
    columns = list(data.columns)
for w in columns:
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.copy()
    target = pd.DataFrame(new_data[w])
    new_data = new_data.drop(w,axis =1)
    display(target.head(2))
    features = pd.DataFrame(new_data)
    display(features.head(2))
# TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, features, test_size=0.25, random_state=42)
# TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train,y_train)
# TODO: Report the score of the prediction using the testing set
    pred = regressor.predict(X_test)
    score = regressor.score(pred,y_test)
    print np.shape(y_test)
    print np.shape(pred)
    print('R2 score for {} : {}'.format(w, score))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    