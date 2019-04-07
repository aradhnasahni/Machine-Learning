"""
Created on Sun Mar 24 23:35:04 2019

@author: aradhna sahni

Working Directory:
C:\Users\ashes\OneDrive\Desktop\Machine Learning

Purpose:
    This code is meant for creating and comparing various 
    machine learning models to predict characters in the Game of Thrones 
    series will live or die.
"""


###############################################################################
# Importing libraries and base dataset
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve 
from sklearn.ensemble import RandomForestClassifier


###############################################################################
# Loading File into Working Environment 
###############################################################################

got = pd.read_excel('GOT_character_predictions.xlsx')
got_df=pd.DataFrame.copy(got)


###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names
got_df.columns

# Displaying first rows of the DataFrame
got_df.head()

# Information about each variable
got_df.info()

# Descriptive statistics
descriptive_statistics=got_df.describe().round(2)


###############################################################################
# Checking for missing values
###############################################################################

print(
      got_df.isnull()
      .any()
      )
""" Missing variables-title,culture,dateOfBirth,mother,father,heir,house,spouse
isAliveMother,isAliveFather,isAliveHier,isAliveSpouse,age """


#Percentage of missing values in each section
print(
      ((got_df[:].isnull().sum())
      /
      got_df[:]
      .count()).round(2).sort_values(ascending = False)
     )


# Creating a copy of got_df
got_df1 = pd.DataFrame.copy(got_df)
 

# Flagging missing values
for col in got_df1:
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    if got_df1[col].isnull().astype(int).sum() > 0: 
        got_df1['m_'+col] = got_df1[col].isnull().astype(int) 



# Assuming people with missing info are dead
got_df1['isAliveMother'] = got_df1['isAliveMother'].fillna(0)
got_df1['isAliveHeir'] = got_df1['isAliveHeir'].fillna(0)
got_df1['isAliveFather'] = got_df1['isAliveFather'].fillna(0)
got_df1['isAliveSpouse'] = got_df1['isAliveSpouse'].fillna(0)

  
    
# Imputation with median for numeric variables
for col in got_df1:
    """ Impute missing values using the median of each column """
    if (got_df1[col].isnull().astype(int).sum() > 0 and 
        is_numeric_dtype(got_df1[col]) == True) :
        col_median = got_df1[col].median()
        got_df1[col] = got_df1[col].fillna(col_median).round(2)
  
 
    
# Imputation with string for categorical variables
for col in got_df1:
    """ Impute missing values using a string """
    if (got_df1[col].isnull().astype(int).sum() > 0 and 
        is_string_dtype(got_df1[col]) == True) :
        got_df1[col] = got_df1[col].fillna('Missing')
 
        
###############################################################################
# Replacing  the categories of same culture
###############################################################################

got_df1['culture'] = got_df1['culture'].replace(['Astapori'], 'Astapor')
got_df1['culture'] = got_df1['culture'].replace(["Asshai'i"], 'Asshai')
got_df1['culture'] = got_df1['culture'].replace(["Braavosi"], 'Braavos')
got_df1['culture'] = got_df1['culture'].replace(["Lyseni"], 'Lysene')
got_df1['culture'] = got_df1['culture'].replace(["Qartheen"], 'Qarth')
got_df1['culture'] = got_df1['culture'].replace(['Andals'], 'Andal')
got_df1['culture'] = got_df1['culture'].replace(['Dornish', 'Dornishmen'], 
                                                  'Dorne')
got_df1['culture'] = got_df1['culture'].replace(['Free folk', 'free folk',
                                                  'Wildling', 'Wildlings'],
                                                  'Free Folk')
got_df1['culture'] = got_df1['culture'].replace(['Ghiscaricari'], 'Ghiscari')
got_df1['culture'] = got_df1['culture'].replace(['ironborn', 'Ironmen'], 
                                                  'Ironborn')
got_df1['culture'] = got_df1['culture'].replace(['Lhazrene', 'Lhazarene', 
                                                  'Lhazreen'], 'Lhazareen')
got_df1['culture'] = got_df1['culture'].replace(['Meereenese'], 'Meereen')
got_df1['culture'] = got_df1['culture'].replace([ 'northmen', 
                                                  'Northern mountain clans'],
                                                  'Northmen')
got_df1['culture'] = got_df1['culture'].replace(['Norvoshi'], 'Norvos')
got_df1['culture'] = got_df1['culture'].replace(['The Reach', 'Reachmen'], 
                                                  'Reach')
got_df1['culture'] = got_df1['culture'].replace(['Riverlands'], 'Rivermen')
got_df1['culture'] = got_df1['culture'].replace(['Stormlander'], 'Stormlands')
got_df1['culture'] = got_df1['culture'].replace(['Summer Islander', 
                                                 'Summer Islands'], 
                                                 'Summer Isles')
got_df1['culture'] = got_df1['culture'].replace(['Vale', 'Valyrian', 
                                                 'Vale mountain clans'], 
                                                 'Valemen')
got_df1['culture'] = got_df1['culture'].replace(['Westerman', 'Westerlands', 
                                                 'Westermen','westermen'], 
                                                 'Westeros')


# Checking how many are left after grouping together
got_df1['culture'].nunique()


# Making a new column expressing culture into number format and sorting alphabetically
got_df1['culture_num'] = pd.factorize(got_df1['culture'], sort=True)[0] + 1


# Creating culture dictionary
culture_codes = got_df1.filter(['culture_num', 'culture'])        
   
     
###############################################################################
# Replacing  the categories of same culture
###############################################################################

got_df1['house'] = got_df1['house'].replace([
                                           'brotherhood without banners',
                                           'Brotherhood without Banners',
                                           'Brotherhood without banners'], 
                                           'Brotherhood Without Banners')
got_df1['house'] = got_df1['house'].replace([
                                           'House Vance of Atranta',
                                           "House Vance of Wayfarer's Rest"],
                                           'House Vance')
got_df1['house'] = got_df1['house'].replace(['House Bolton of the Dreadfort'],
                                           'House Bolton')
got_df1['house'] = got_df1['house'].replace(['House Dayne of High Hermitage'],
                                           'House Dayne')
got_df1['house'] = got_df1['house'].replace([
                                           'House Brune of Brownhollow',
                                           'House Brune of the Dyre Den'],
                                           'House Brune')
got_df1['house'] = got_df1['house'].replace([
                                           'House Baratheon of Dragonstone',
                                           "House Baratheon of King's Landing"],
                                           'House Baratheon')
got_df1['house'] = got_df1['house'].replace([
                                           'House Farwynd of the Lonely Light'],
                                           'House Farwynd')
got_df1['house'] = got_df1['house'].replace([
                                           'House Harlaw of Grey Garden',
                                           'House Harlaw of Harlaw Hall',
                                           'House Harlaw of Harridan Hill',
                                           'House Harlaw of the Tower of Glimmering'],
                                           'House Harlaw')
got_df1['house'] = got_df1['house'].replace([
                                           'House Fossoway of Cider Hall',
                                           'House Fossoway of New Barrel'],
                                           'House Fossoway')
got_df1['house'] = got_df1['house'].replace([
                                           'House Frey of Riverrun'],
                                           'House Frey')
got_df1['house'] = got_df1['house'].replace([
                                           "House Flint of Widow's Watch"],
                                           'House Flint')
got_df1['house'] = got_df1['house'].replace([
                                            'House Royce of the Gates of the Moon'],
                                            'House Royce')
got_df1['house'] = got_df1['house'].replace([
                                           'House Goodbrother of Shatterstone'],
                                           'House Goodbrother')
got_df1['house'] = got_df1['house'].replace([
                                           'House Lannister of Lannisport',
                                           'House Lannister of Casterly Rock'],
                                           'House Lannister')
got_df1['house'] = got_df1['house'].replace([
                                           'House Tyrell of Brightwater Keep'],
                                           'House Tyrell')
got_df1['house'] = got_df1['house'].replace(['House Kenning of Harlaw',
                                           'House Kenning of Kayce'],
                                           'House Kenning')


# How many unique houses do we have now?

got_df1['house'].nunique()     


# Making new column changing house into numeric & sorting alphabetically
    
got_df1['house_num'] = pd.to_numeric(pd.factorize(got_df1['house'], 
                                                    sort=True)[0] + 1)

# Creating a dictionary for house
house_codes = got_df1.filter(['house_num', 'house'])


###############################################################################
#Outlier Detection
###############################################################################
      
# Setting threshold

numDeadRelations = 2
popularity = 0.2
house_code_l= 95
culture_code_l= 27
dateOfBirth = 260
dateOfBirth_low = -25
age = 28
age_low = -25250


# Boxplots   

for col in got_df1.loc[:, [ 'dateOfBirth',
                            'age',
                            'popularity',
                            'numDeadRelations',
                            'house_num',
                            'culture_num'
                            ]]:
        got_df1.boxplot(column = col, vert = False)
        plt.axvline(f"{col}_out", color = 'blue') 
        plt.title(f"{col}")
        plt.tight_layout()
        plt.show()



# Histograms
        
for col in got_df1.iloc[:, 1:]:
    if is_numeric_dtype(got_df1[col]) == True:
      sns.distplot(got_df1[col], kde = True)
      plt.tight_layout()
      plt.show()
      
        
###############################################################################
# FLAG OUTLIERS
###############################################################################

# Creating functions to flag upper and lower limits

def up_out(col,lim):
    got_df1['o_'+col] = 0
    for val in enumerate(got_df1.loc[ : , col]):   
        if val[1] > lim:
            got_df1.loc[val[0], 'o_'+col] = 1 
                
def low_out(col, lim):
    got_df1['o_'+col] = 0
    for val in enumerate(got_df1.loc[ : , col]):   
        if val[1] < lim:
            got_df1.loc[val[0], 'o_'+col] = 1      
            
# Flagging upper outliers
up_out('numDeadRelations', numDeadRelations)
up_out('popularity', popularity)
low_out('house_num', house_code_l)
low_out('culture_num', culture_code_l)


# Flagging upper and lower outliers for dob and age
got_df1['o_dateOfBirth'] = 0
for val in enumerate(got_df1.loc[ : , 'dateOfBirth']):    
    if val[1] < dateOfBirth_low:
        got_df1.loc[val[0], 'o_dateOfBirth'] = -1
    elif val[1] > dateOfBirth:
        got_df1.loc[val[0], 'o_dateOfBirth'] = 1


got_df1['o_age'] = 0
for val in enumerate(got_df1.loc[ : , 'age']):    
    if val[1] < age_low:
        got_df1.loc[val[0], 'o_age'] = -1
    elif val[1] > age:
        got_df1.loc[val[0], 'o_age'] = 1


# Correlation Matrices
df_corr = got_df1.corr()
df_corr.loc['isAlive'].sort_values(ascending = True)


###############################################################################
# Model 1 - Logistic Regression-with significant variables
###############################################################################
        
logistic_sig = smf.logit(formula = """isAlive ~  male + 
                                                 book1_A_Game_Of_Thrones +
                                                 book3_A_Storm_Of_Swords +
                                                 book4_A_Feast_For_Crows +
                                                 o_popularity +
                                                 o_dateOfBirth + 
                                                 numDeadRelations 
                                                 """,
                         data = got_df1)

#Fitting results
results_logistic_sig = logistic_sig.fit()

#Printing summary statistics
results_logistic_sig.summary()

# AIC and BIC of this model
print('AIC:', results_logistic_sig.aic.round(2))
print('BIC:', results_logistic_sig.bic.round(2))


###############################################################################
# Model 2 - added more variables in it
###############################################################################
lg_significant = smf.logit(formula = """isAlive ~  male  +
                                                   dateOfBirth  +
                                                   isNoble +
                                                   popularity +
                                                   o_house_num +
                                                   o_culture_num +
                                                   m_spouse +
                                                   numDeadRelations +
                                                   book3_A_Storm_Of_Swords +
                                                   book4_A_Feast_For_Crows  +
                                                   book2_A_Clash_Of_Kings +
                                                   book5_A_Dance_with_Dragons +
                                                   book1_A_Game_Of_Thrones 
                                                """,
                           data = got_df1)

# Fitting Results
results2 = lg_significant.fit()

# Printing Summary Statistics
print(results2.summary())
    

###############################################################################
# Applying Model 2 in scikit-learn
###############################################################################

# Preparing a DataFrame based on the analysis above
got_data = got_df1.loc[: , [    'male',
                                'm_spouse',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'dateOfBirth',
                                'book1_A_Game_Of_Thrones',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'o_house_num',
                                'o_culture_num',
                                'book2_A_Clash_Of_Kings',
                                'book5_A_Dance_with_Dragons'
                                ]]

# Preparing the target variable 
got_target =  got_df1.loc[: , 'isAlive']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508
            )

logreg = LogisticRegression(C=0.1)


logreg_fit = logreg.fit(X_train, y_train)


# Predictions
logreg_pred = logreg_fit.predict(X_test)

train_score = logreg_fit.score(X_train, y_train).round(3)
test_score = logreg_fit.score(X_test, y_test).round(3)

print('Training Score', logreg_fit.score(X_train, y_train).round(2))#0.78
print('Testing Score:', logreg_fit.score(X_test, y_test).round(2))#0.77


##############################################################################
# Model 3 - Machine Learning using kNN Classifier on variables in Model 2
##############################################################################

# Creating two lists for train and test accuracy
training_accuracy = []
test_accuracy = []

# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Looking for the highest test accuracy
print(max(test_accuracy))

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 4 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors = 3)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_clf.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
knn_clf_optimal_pred = knn_clf_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(3))#0.86
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(3))#0.81


###############################################################################
# Model 4- Building Random Forest Model Based on Best Parameters
###############################################################################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 600,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

# AUC Score
y_pred_score = rf_optimal.predict_proba(X_test)[:,1]
print("AUC score: {:.2f}".format(roc_auc_score(y_test, y_pred_score)))#0.86

"""Score is coming from RFC-
Training score : 0.824
Testing Score" 0.835
AUC score: 0.86 """
# the model seems to be underfitting by 1.1 gap hence,choosing GBM as below

###############################################################################
# Model 5-Final Model- Gradient Boosted Machines for Model 2 
###############################################################################

gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.267,
                                  n_estimators = 88,
                                  max_depth = 2,
                                  criterion = 'friedman_mse',
                                  warm_start = True,
                                  min_samples_leaf=29,
                                  )

gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(2))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(2))

# AUC Score
y_pred_prob = gbm_basic_fit.predict_proba(X_test)[:,1]
print("AUC score: {:.2f}".format(roc_auc_score(y_test, y_pred_prob)))


"""Best score is coming from GBM- 
Training score : 0.87
Testing Score" 0.86
AUC score: 0.85 """


###############################################################################
# Creating a AUC Curve
###############################################################################

# Compute predicted probabilities: y_pred_prob
y_pred_prob = gbm_basic_fit.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print("AUC score: {:.2f}".format(roc_auc_score(y_test, y_pred_prob)))
roc_auc_score(y_test, y_pred_prob)
print("\n")


###############################################################################
# Creating a confusion matrix
###############################################################################

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = gbm_basic_predict))


# Visualizing a confusion matrix
import seaborn as sns

labels = ['Dead', 'Alive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = gbm_basic_predict)

sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Accent',
            fmt='g')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


##############################################################################
# Creating a classification report
###############################################################################

print(classification_report(y_true = y_test,
                            y_pred = gbm_basic_predict,
                            target_names = labels))

#Our model has predicted alive category better than dead as 0.91


###############################################################################
# Cross Validation with three-folds
###############################################################################

cv = cross_val_score(gbm_3,
                           got_data,
                           got_target,
                           cv = 3)


print(cv)


print(pd.np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))


###############################################################################
# Storing Model Predictions and Summary
###############################################################################

#Residuals
residuals = y_test - gbm_basic_predict

# Storing predictions as a dictionary.
model_predictions_df = pd.DataFrame({
        'Actual' : y_test,
        'KNN Predictions': knn_clf_optimal_pred,
        'Random Forest Model predictions': rf_optimal_pred,
        'Gradient Boost Machine predictions': gbm_basic_predict,
        'GBM Residuals': residuals})

model_predictions_df.to_excel("Game_Of_Thrones_Model_Predictions.xlsx")

#End



