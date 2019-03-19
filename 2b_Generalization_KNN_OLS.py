#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 04:34:17 2019

@author: chase.kusterer
"""

"""
Prof. Chase:
    The machine learning framework:
        * Exploratory Data Analysis (EDA)
        * Break into Train/Test Splits
        * Use KNN on the full dataset to establish a 'learning base'
        * If regression or classification, use statsmodels to develop a model base
        * Test each candidate model's ability to generalize
"""


# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation


# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt


file = 'Housing_Dummies.xlsx'

housing = pd.read_excel(file)


###############################################################################
###############################################################################
# Supervised Modeling Process For Our Course
###############################################################################
###############################################################################


###############################################################################
#  Splitting the Data Using Train/Test Split
###############################################################################

housing_data   = housing.drop(['SalePrice',
                               'Street',
                               'Lot Config',
                               'Neighborhood'],
                                axis = 1)



housing_target = housing.loc[:, 'SalePrice']


X_train, X_test, y_train, y_test = train_test_split(
            housing_data,
            housing_target,
            test_size = 0.25,
            random_state = 508)


# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


###############################################################################
# Building a Base Model with statsmodels
###############################################################################


# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
housing_train = pd.concat([X_train, y_train], axis = 1)


# Step 1: Build the model
lm_price_qual = smf.ols(formula = """SalePrice ~
                                     housing_train['Overall Qual']""",
                         data = housing_train)



# Step 2: Fit the model based on the data
results = lm_price_qual.fit()



# Step 3: Analyze the summary output
print(results.summary())



"""
Prof. Chase:
    The advantage of statsmodels is that it provides a summary output,
    which is something that scikit-learn does not do. This is a classical
    approach to statistical modeling.
    
    Once we are satisfied with our variable selection, we can move on to using
    other modeling techniques with no summary output.
"""



# Let's pull in the optimal model from before, only this time on the training
# set
lm_significant = smf.ols(formula = """SalePrice ~ housing_train['Overall Qual'] +
                                                  housing_train['Overall Cond'] +
                                                  housing_train['Mas Vnr Area'] +
                                                  housing_train['Total Bsmt SF'] +
                                                  housing_train['2nd Flr SF'] +
                                                  housing_train['Fireplaces'] +
                                                  housing_train['Garage Cars'] +
                                                  housing_train['Garage Area'] +
                                                  housing_train['Porch Area'] +
                                                  housing_train['Pool Area'] +
                                                  housing_train['out_Lot_Area'] +
                                                  housing_train['out_ff_SF'] +
                                                  housing_train['out_sf_SF']
                                                  """,
                                                  data = housing_train)


# Fitting Results
results = lm_significant.fit()



# Printing Summary Statistics
print(results.summary())



"""
Prof. Chase:
    Notice how some variables may be insignificant based on the training set.
    This is because we are dealing with less data, and there may not be enough
    observations to establish the same trend.
"""



# Now that we have selected our variables, our next step is to prepare them
# in scikit-learn so that we can see how they predict on new data.


###############################################################################
# Applying Our Optimal Model in scikit-learn
###############################################################################

# Preparing a DataFrame based the the analysis above
housing_data   = housing.loc[ : , ['Overall Qual',
                                   'Overall Cond',
                                   'Mas Vnr Area',
                                   'Total Bsmt SF',
                                   '2nd Flr SF',
                                   'Fireplaces',
                                   'Garage Cars',
                                   'Garage Area',
                                   'Porch Area',
                                   'Pool Area',
                                   'out_Lot_Area',
                                   'out_ff_SF',
                                   'out_sf_SF']]


# Preparing the target variable
housing_target = housing.loc[:, 'SalePrice']


# Now that we have a new set of X_variables, we need to run train/test
# split again

X_train, X_test, y_train, y_test = train_test_split(
            housing_data,
            housing_target,
            test_size = 0.25,
            random_state = 508)




########################
# Using KNN  On Our Optimal Model (same code as our previous script on KNN)
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []



neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)



########################
# The best results occur when k = 15.
########################

# Building a model with k = 15
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 15)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)



########################
## Does OLS predict better than KNN?
########################

from sklearn.linear_model import LinearRegression


# Prepping the Model
lr = LinearRegression()


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))


"""
Prof. Chase:
    These values can be much lower than what we saw before when we didn't
    create a train/test split. However, these results are realistic given we
    have a better understanding as to how well our model will predict on new
    data.
"""


# Printing model results
print(f"""
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")


###############################################################################
# Outputting Model Coefficients, Predictions, and Other Metrics
###############################################################################

# What does our leading model look like?
pd.DataFrame(list(zip(housing_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))



########################
# Some Other Available Metrics
########################

# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)


# Mean Squared Error
lr_mse = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)


# Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = pd.np.sqrt(lr_mse)
print(lr_rmse)

dir(sklearn.metrics)

"""
Prof. Chase:
    More metrics can be found at
    https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
"""



###############################################################################
# Storing Model Predictions and Summary
###############################################################################

# We can store our predictions as a dictionary.
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_reg_optimal_pred,
                                     'OLS_Predicted': lr_pred})



model_predictions_df.to_excel("Ames_Model_Predictions.xlsx")

