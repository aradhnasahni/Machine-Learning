"""
Created on Wed Mar 13 04:52:15 2019

@author: team 4.fillmore

Working Directory:
/Users/aradhnasahni/Desktop/Machine Learning

Purpose:
    This code is meant for creating and comparing various 
    machine learning models to predict birthweight.
"""


###############################################################################
# Importing libraries and base dataset
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

file = 'birthweight_feature_set-1.xlsx'
birthwt = pd.read_excel(file)


###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names
birthwt.columns

# Displaying first rows of the DataFrame
birthwt.head()

# Dimensions of DataFrame
birthwt.shape

# Information about each variable
birthwt.info()

# Descriptive statistics
birthwt.describe().round(2)


###############################################################################
# Checking for missing values
###############################################################################

print(
      birthwt.isnull()
      .any()
      )
# meduc,feduc,npvis

# In absolute numbers- how many missing values in each column?
print(
      birthwt[:]
      .isnull()
      .sum()
      )

#Percentage of missing values in each section
print(
      ((birthwt[:].isnull().sum())
      /
      birthwt[:]
      .count()).round(3)
     )

# Flagging missing values
for col in birthwt:
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    if birthwt[col].isnull().astype(int).sum() > 0: 
        birthwt['m_'+col] = birthwt[col].isnull().astype(int)      
 

      
#Creating new df to plot distributions
df_dropped = birthwt.dropna()

# Distribution of birthweight
sns.distplot(df_dropped['bwght']) 

# Choosing imputation techniques
plt.subplot(2, 2, 1)
sns.distplot(df_dropped['meduc'], color = "red")

plt.subplot(2, 2, 2)
sns.distplot(df_dropped['npvis'], color = 'orange')

plt.subplot(2, 2, 3)
sns.distplot(df_dropped['feduc'], color = 'black')

plt.tight_layout()
plt.show()



# Creating new df to impute missing values with median

df_median = pd.DataFrame.copy(birthwt)

for col in df_median:
    """ Impute missing values using the median of each column """
    if df_median[col].isnull().astype(int).sum() > 0:
        col_median = df_median[col].median()
        df_median[col] = df_median[col].fillna(col_median).round(2)

print(f""" Presence of NAs: {df_median.isnull()
                             .any().any()}
      """) 


###############################################################################
#####PLOTS
###############################################################################
# Boxplots   
for col in birthwt.iloc[:, 0:10]:
    birthwt.boxplot(column = col, vert = False)
    plt.title(f"{col}")
    plt.tight_layout()
    plt.show()

# Histograms
for col in df_median.iloc[:, 0:10]:
    sns.distplot(df_median[col], bins = 'fd')
    plt.tight_layout()
    plt.show()


###############################################################################
#Outlier Detection
###############################################################################
 
# Setting threshold
mage = 54
meduc_lo = 10.5
monpre = 4
npvis_lo = 5
npvis = 16
fage = 53
feduc_lo = 8
omaps_lo = 7
fmaps_lo = 7.5
cigs = 18.5
drink = 10.5
    
#Plotting all variable distributions with threshold

plt.subplot(2, 2, 1)    
sns.distplot(df_median['mage'], bins = 'fd')    
plt.axvline(mage, color = 'red')   

plt.subplot(2, 2, 2)   
sns.distplot(df_median['meduc'], bins = 'fd', color = 'orange') 
plt.axvline(meduc_lo, color='blue') 

plt.subplot(2, 2, 3)    
sns.distplot(df_median['fage'], bins = 'fd', color = 'maroon')    
plt.axvline(fage, color='blue') 

plt.subplot(2, 2, 4)   
sns.distplot(df_median['feduc'], bins = 'fd', color = 'pink') 
plt.axvline(feduc_lo, color='red') 

plt.tight_layout()
plt.show()



plt.subplot(2, 2, 1)    
sns.distplot(df_median['monpre'], bins = 'fd', color = 'purple')    
plt.axvline(monpre,color='red')   

plt.subplot(2, 2, 2)   
sns.distplot(df_median['npvis'], bins = 'fd', color = 'gold') 
plt.axvline(npvis, color = 'blue') 
plt.axvline(npvis_lo, color = 'blue')    

plt.tight_layout()
plt.show()



plt.subplot(2, 2, 1)    
sns.distplot(df_median['omaps'], bins = 'fd', color = 'navy')    
plt.axvline(omaps_lo,color='red')   

plt.subplot(2, 2, 2)   
sns.distplot(df_median['fmaps'], bins = 6, color = 'green', kde = False) 
plt.axvline(fmaps_lo, color = 'purple') 

plt.tight_layout()
plt.show()



plt.subplot(2, 2, 1)   
sns.distplot(df_median['cigs'], bins = 'fd', color = 'black') 
plt.axvline(cigs, color = 'red') 

plt.subplot(2, 2, 2)   
sns.distplot(df_median['drink'], bins = 'fd', color = 'gray') 
plt.axvline(drink, color = 'red') 

plt.tight_layout()
plt.show()


###############################################################################
# FLAG OUTLIERS
###############################################################################

# Creating functions to flag upper and lower limits

def low_out(col, lim):
    df_median['o_'+col] = 0
    for val in enumerate(df_median.loc[ : , col]):   
        if val[1] < lim:
            df_median.loc[val[0], 'o_'+col] = 1

def up_out(col,lim):
    df_median['o_'+col] = 0
    for val in enumerate(df_median.loc[ : , col]):   
        if val[1] > lim:
            df_median.loc[val[0], 'o_'+col] = 1 
                
            
# Flagging upper outliers
up_out('mage', mage)
up_out('monpre', monpre)
up_out('npvis', npvis)
up_out('fage', fage)
up_out('cigs', cigs)
up_out('drink', drink)

# Flagging lower outliers
low_out('meduc', meduc_lo)
low_out('feduc', feduc_lo)
low_out('omaps', omaps_lo)            
low_out('fmaps', fmaps_lo)


# Flagging upper and lower outliers for npvis 
df_median['out_both_npvis'] = 0
for val in enumerate(df_median.loc[ : , 'npvis']):    
    if val[1] < npvis_lo:
        df_median.loc[val[0], 'out_both_npvis'] = -1

for val in enumerate(df_median.loc[ : , 'npvis']):   
    if val[1] > npvis:
        df_median.loc[val[0], 'out_both_npvis'] = 1



# Correlation Matrices
df_corr = birthwt.corr()
df_corr.loc['bwght'].sort_values(ascending = True)

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.show()



# Creating binary variable for bwght 
df_median['bin_bwght'] = 0
 
for val in enumerate(df_median.loc[ : , 'bwght']): 
    """ Flag uncommon birthweight as 1. """
    if val[1] >= 2500 and val[1] <= 4000:
        df_median.loc[val[0], 'bin_bwght'] = 1


###############################################################################
# Plots to detect relationships
###############################################################################

for col in df_median.iloc[ : , 0:10]:
    plt.scatter(x = col,
                y = 'bwght',
                alpha = 0.7,
                cmap = 'bwr',
                data = birthwt)
    plt.title(f"bwght vs {col}")
    plt.tight_layout()
    plt.show()
# mage and bwght have a curvilinear relationship


###############################################################################
# Model 1 - OLS
###############################################################################
        
lm_full = smf.ols(formula = """bwght ~ df_median['mage'] +
                                       df_median['meduc'] +
                                       df_median['monpre'] +
                                       df_median['npvis'] +
                                       df_median['fage'] +
                                       df_median['feduc'] +
                                       df_median['omaps'] +
                                       df_median['fmaps'] +
                                       df_median['cigs'] +
                                       df_median['drink'] +
                                       df_median['male'] +
                                       df_median['mwhte'] +
                                       df_median['mblck'] +
                                       df_median['moth'] +
                                       df_median['fwhte'] +
                                       df_median['fblck'] +
                                       df_median['foth'] +
                                       df_median['m_meduc'] +
                                       df_median['m_npvis'] +
                                       df_median['m_feduc'] +
                                       df_median['o_mage'] +
                                       df_median['o_monpre'] +
                                       df_median['o_npvis'] +
                                       df_median['o_fage'] +
                                       df_median['o_feduc'] +
                                       df_median['o_omaps'] +
                                       df_median['o_fmaps'] +
                                       df_median['o_drink'] 
                                       """,
                    data = df_median)

# Fitting Results
results1 = lm_full.fit()

# Printing Summary Statistics
print(results1.summary())

print(results1.rsquared.round(3))


###############################################################################
# Model 2 - with significant variables
###############################################################################
lm_significant = smf.ols(formula = """bwght ~  df_median['mage'] +
                                               df_median['cigs'] +
                                               df_median['drink'] +
                                               df_median['mwhte'] +
                                               df_median['mblck'] +
                                               df_median['moth'] +
                                               df_median['fwhte'] +
                                               df_median['fblck'] +
                                               df_median['foth'] 
                                               """,
                         data = df_median)

# Fitting Results
results2 = lm_significant.fit()

# Printing Summary Statistics
print(results2.summary())

print(results2.rsquared.round(3))
        

###############################################################################
# Applying OLS Model in scikit-learn
###############################################################################

# Preparing a DataFrame based on the analysis above
bw_data = df_median.loc[ : , ('mage',
                             'cigs',
                             'drink',
                             'mwhte',
                             'mblck',
                             'moth',
                             'fwhte',
                             'fblck',
                             'foth')]

# Preparing the target variable      
bw_target = df_median.loc[ : , 'bwght']

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(
        bw_data,
        bw_target,
        test_size = 0.1,
        random_state = 508)


########################
# Model 3 - Machine Learning using kNN on variables in Model 2
########################

# Creating two lists for train and test accuracy
training_accuracy = []
test_accuracy = []

# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
# Plotting the visualization
fig, ax = plt.subplots(figsize = (12, 9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Printing highest test accuracy
print(max(test_accuracy))

# Optimal number of neighbours
print(test_accuracy.index(max(test_accuracy))+1)

# Building model with k=10
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 10)

# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)

knn_pred = knn_reg.predict(X_test)

# Scoring the model
y_score = knn_reg.score(X_test, y_test)

print(y_score) 


########################
# Model 4 - Machine Learning using Linear Regression on variables in Model 2
########################

# Creating and training model
lm = LinearRegression()
reg_model = lm.fit(X_train, y_train)

# Getting predictions
predictions = reg_model.predict(X_test)

# Comparing the testing score to the training score.
print('Training Score:', lm.score(X_train, y_train).round(4))
print('Testing Score:', lm.score(X_test, y_test).round(4))

# Printing model results
print(f"""
KNN score: {y_score.round(2)}
OLS score: {lm.score(X_test, y_test).round(2)}
""")


###############################################################################
# Model 5 - Decision Trees
###############################################################################

# Building the tree
tree_fit = DecisionTreeRegressor(max_depth = 3,
                                 min_samples_leaf = 17,
                                 criterion = 'mse',
                                 random_state = 508)
tree_fit.fit(X_train, y_train)

print('Training Score:', tree_fit.score(X_train, y_train).round(4))
print('Testing Score:',  tree_fit.score(X_test, y_test).round(4))

# Visualizing the tree
dot_data = StringIO()


export_graphviz(decision_tree = tree_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

# Cigs and drink have strong effect - overpowering other variables.


###############################################################################
# Model 6 - kNN with other variables
###############################################################################

bw_data = df_median.loc[ :, ('feduc',
                             'o_npvis',
                             'o_mage',
                             'o_fage',
                             'drink',
                             'cigs')]    
    
bw_target = df_median.loc[ : ,'bwght']        

X_train, X_test, y_train, y_test = train_test_split(
        bw_data,
        bw_target,
        test_size = 0.1, 
        random_state = 508)

# Creating two lists 
training_accuracy = []
test_accuracy = []

# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
# Plotting the visualization
fig, ax = plt.subplots(figsize = (12, 9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Optimal number of neighbours
print(test_accuracy.index(max(test_accuracy))+1)

# Building model with k=10
knn_reg = KNeighborsRegressor(algorithm = 'auto', #auto runs all distances-
                              n_neighbors = 28)

# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)

knn_pred = knn_reg.predict(X_test).round(2)

# Scoring the model
y_score = knn_reg.score(X_test, y_test)

print(y_score) 


###############################################################################
# Final Model - Optimal Solution with 70% predictive accuracy
###############################################################################
     
bw_data = df_median.loc[ : ,('feduc',
                             'o_npvis',
                             'o_mage',
                             'o_fage',
                             'drink',
                             'cigs')]    
    
bw_target = df_median.loc[ : ,'bwght']        
        
#Train Test split
X_train, X_test, y_train, y_test = train_test_split(
        bw_data,
        bw_target,
        test_size = 0.1, 
        random_state = 508)

#Creating and training model
lm = LinearRegression()
reg_model = lm.fit(X_train, y_train)

#Getting predictions
predictions = reg_model.predict(X_test).round(2)

#Visualizing the predictions
plt.scatter(y_test, predictions) #Similar to a linear model


#Checking R squared
train_score = reg_model.score(X_train, y_train).round(3)
test_score = reg_model.score(X_test, y_test).round(3)

# Maximum test score achieved with small gap between train and test score  
print(f"""
Comparing score:
Train score:             {train_score}
Test score:              {test_score}

Linear Regression Model:
Intercept:               {reg_model.intercept_.round(3)}
feduc:                   {reg_model.coef_[0].round(3)}
o_npvis:                 {reg_model.coef_[1].round(3)}
o_mage:                  {reg_model.coef_[2].round(3)}
o_fage:                  {reg_model.coef_[3].round(3)}
drink:                   {reg_model.coef_[4].round(3)}
cigs:                    {reg_model.coef_[5].round(3)}
""")
    
#Residuals
residuals = y_test - predictions

# Mean Squared Error-put in model
lm_mse = sklearn.metrics.mean_squared_error(y_test, predictions)
print(lm_mse)

# Root Mean Squared Error (how far off are we on each observation?)
lm_rmse = pd.np.sqrt(lm_mse)
print(lm_rmse)


###############################################################################
# Storing Model Predictions and Summary
###############################################################################

# Storing predictions as a dictionary.
model_predictions_df = pd.DataFrame({
        'Actual' : y_test,
        'KNN Predictions': knn_pred,
        'Linear Regression Predictions': predictions,
        'Linear Regression Residuals': residuals})

model_predictions_df.to_excel("Birthweight_Model_Predictions.xlsx")

#End
