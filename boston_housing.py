# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:49:51 2016

@author: Jae

Boston Housing dataset contains aggregated data on various features for houses in Greater 
Boston communities, including the median value of home for each of those areas. An optimal 
model based on a statistical analysis is developed first. Then the model is used to estimate 
the best selling price for a home.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV

# Create client's feature set for which we will be predictiong a selling preice 
CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]

# Load the Boston Housing dataset into the city_data variable
city_data = datasets.load_boston()

# Initialxe the housing prices and housing features
housing_prices = city_data.target
housing_features = city_data.data

print "\nBoston Housing dataset loaded successfully! \n"

# Number of houses in the dataset
total_houses = len(housing_prices)

# Number of features in the dataset
total_features = housing_features.shape[1]

# Minimum house value in the dataset
minimum_price = housing_prices.min()

# Maximun house value in the dataset
maximum_price = housing_prices.max()

# Mean house value in the dataset
mean_price = housing_prices.mean()

# Median house value of the dataset
median_price = np.median(housing_prices)

# Standard deviation of house values of the dataset
std_dev = housing_prices.std()

# Show the calculated statistics
print "Boston Housing dataset statistics (in $1000's): \n"
print "Total number of the houses:", total_houses
print "Total number of features:",total_features
print "Minimum house price:", minimum_price
print "Maximum house price:", maximum_price
print "Mean house price: {0:.3f}".format(mean_price)
print "Median house price:", median_price
print "Standart deviation of house price: {0:.3f}".format(std_dev)

def shuffle_split_data(X,y):
    """ Shuffles and splits data into 70% training and 30% testing subsets,
    then returns the training and testing subsets.  """
    ss = cross_validation.ShuffleSplit(len(y),n_iter=1, test_size=0.3)
    
    # Shuffle and split the data
    for train_index,test_index in ss:
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    return X_train, y_train, X_test, y_test


def performance_metric(y_true,y_predict):
    """ Calculates and returns the total error between true and predicted values based on a 
    a performance metric - mean_squared_error """
    error = mean_squared_error(y_true, y_predict)
    return error

def fit_model(X,y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X
        and target label y and returns this optimal model.  """
        
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
        
    # Set up the parameters we wish to tune
    parameters = {'max_depth': (1,2,3,4,5,6,7,8,9,10)}
        
    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better = False)
        
    # Make the GridSearchCV object
    reg = GridSearchCV(regressor, param_grid = parameters, scoring = scoring_function)
        
    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X,y)
    
    # Return the optimal model
    return reg.best_estimator_
        
def learning_curves(X_train, y_train, X_test, y_test):
    """ Calculates the performance of serveral models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10....."
    
    # Create the figure window
    fig = plt.figure(figsize=(10,8))
    
    # We will vary the training set size so that we have 50 different sizes.
    sizes = np.rint(np.linspace(1, len(X_train),50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))
    
    # Create four different models based on max_depth

    for k, depth in enumerate([1,3,6,10]):
        
        for i,s in enumerate(sizes):
            
            # Setup a decision tree regressor  so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth =depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])
            
            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s],regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test,regressor.predict(X_test))
        
        # Subplot the learning curve graph
        ax = fig.add_subplot(2,2, k+1)
        ax.plot(sizes,test_err, lw = 2, label = "Testing Error")
        ax.plot(sizes, train_err, lw = 2, label = "Training Error")
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel("Number of Data Points in Training Set")
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 18, y=1.03)
    fig.tight_layout()
    fig.show()
    
    
def model_complexity(X_train,y_train,X_test,y_test):
    """ Calculates the performance of the model as model complexity increases.
        The learining and testing errors rates are then plotteed. """
        
    print "Creating a model complexity graph..."
    
    # We will vary the max_depth of a decision tree model form 1 to 14
    max_depth = np.arange(1,14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))
    
    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)
        
        # Fit the learner to the training data
        regressor.fit(X_train,y_train)
        
        # Find the performance on the training set
        train_err[i] = performance_metric(y_train,regressor.predict(X_train))
        
        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test,regressor.predict(X_test))

    # Plot the model complexity graph
    plt.figure(figsize = (7,5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    plt.plot(max_depth, train_err,lw=2, label = 'Training Error')
    plt.legend()
    plt.xlabel('Maximum Depth')
    plt.ylabel('Total Error')
    plt.show()


learning_curves(X_train, y_train, X_test, y_test)   
model_complexity(X_train,y_train,X_test, y_test)

reg = fit_model(X_train,y_train)
print 'Final model has an optimal max_depth parameter of', reg.get_params()['max_depth'] 

sale_price = reg.predict(CLIENT_FEATURES)
print "Predicted value of client's home: {0:.3f}".format(sale_price[0])
